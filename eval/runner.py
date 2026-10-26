"""Run a benchmark: one video per (scene, trajectory) pair and method, then score.

Phase 1 generates every missing video with eval/worker.py, which loads the
model once and then runs the same generation as WorldKV's generate_fast.py
for each video; or with a fake model on a laptop (--fake). Videos that
already exist are skipped, so a stopped run can simply be restarted.
Phase 2 scores all videos and writes, in the output folder:
  results.csv  one row per run (scene, trajectory, method)
  summary.csv  one row per method, like Table 1 of the paper

Usage on a GPU machine:
  python -m eval.runner --benchmark eval/benchmark/dev/benchmark.json \
      --out eval/outputs/dev --ckpt-dir ./lingbot-world-base-cam --nproc 4
Usage on a laptop (no GPU):
  python -m eval.runner --benchmark eval/benchmark/dev/benchmark.json \
      --out eval/outputs/dev_fake --fake --no-fid
"""

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time
import zlib

import imageio.v3 as iio
import numpy as np
import torch

from eval import pairing
from eval.metrics import PairMetrics, SetMetrics, default_device, evaluate_video, read_video, summarize
from eval.telemetry import ENV_VAR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WorldKV with the paper's main settings (Sec. 5.1, Table 1 = Table 2 "3 -> 1.5"):
# retrieve 6 chunks (--retrieval_frames counts uncompressed frames, 6 x 3 = 18),
# each compressed at store time to 1.5 frames. --kv_compression_keep_ratio is
# the fraction kept per non-anchor frame, so 1 + 2 x 0.25 = 1.5 frames, and the
# 6 retrieved chunks fill the paper's 9-frame retrieval budget.
WORLDKV_FLAGS = [
    "--use_retrieval", "--retrieval_frames", "18",
    "--kv_compression_enable", "--kv_compression_keep_ratio", "0.25",
    "--kv_compression_at_store", "--kv_compression_pooled",
]
METHODS = {
    "sliding": ["--sliding_window", "18"],
    "full_kv": ["--full_kv"],
    "gpu_only": WORLDKV_FLAGS + ["--kv_bank_on_gpu"],
    "cpu_only": WORLDKV_FLAGS,
}
RETRIEVAL_METHODS = ("gpu_only", "cpu_only")
# WorldKV's attention window in latent frames: sink (3) + retrieved (18) +
# recent (6), see image2video_fast.py. The code sizes "recent" as what is left
# of the cache after sink and retrieval, and caps the cache at the video
# length, so a shorter video silently loses the previous chunk from "recent".
WORLDKV_WINDOW_LATENT = 3 + 18 + 6
METRIC_GROUPS = ("all", "long")


def check_video_lengths(entries, methods):
    """Refuse WorldKV runs on videos shorter than its attention window."""
    if not any(m in RETRIEVAL_METHODS for m in methods):
        return
    for entry in entries:
        num_poses = len(pairing.load_poses(entry["traj_dir"]))
        frames = pairing.num_video_frames(num_poses, entry["frame_num"])
        latent = pairing.latent_index(frames - 1) + 1
        if latent < WORLDKV_WINDOW_LATENT:
            raise ValueError(
                f"{run_name(entry)}: {latent} latent frames, but WorldKV needs at least "
                f"{WORLDKV_WINDOW_LATENT} (about {4 * WORLDKV_WINDOW_LATENT} video frames). "
                "Use a longer trajectory.")


def load_benchmark(path):
    """Read a benchmark file into a list of runs, one per (scene, trajectory)."""
    with open(path) as f:
        bench = json.load(f)
    root = os.path.dirname(os.path.abspath(path))
    with open(os.path.join(root, bench["prompts"])) as f:
        prompts = json.load(f)
    entries = []
    for p in bench["pairs"]:
        images = glob.glob(os.path.join(root, bench["images_dir"], p["scene"] + ".*"))
        if len(images) != 1:
            raise FileNotFoundError(f"Expected one image for scene {p['scene']}, found {images}.")
        entries.append({
            "scene": p["scene"],
            "trajectory": p["trajectory"],
            "image": images[0],
            "traj_dir": os.path.join(root, bench["traj_dir"], p["trajectory"]),
            # Trajectory names end with their length, e.g. R_L_R_L_R_L90_360.
            "frame_num": int(p["trajectory"].rsplit("_", 1)[1]),
            "prompt": prompts[p["scene"]],
        })
    return entries


def run_name(entry):
    return f"{entry['scene']}_{entry['trajectory']}"


def video_path(out_dir, method, entry):
    return os.path.join(out_dir, "videos", method, run_name(entry) + ".mp4")


def telemetry_path(out_dir, method, entry):
    return os.path.join(out_dir, "runs", method, run_name(entry) + ".telemetry.json")


def meta_path(out_dir, method, entry):
    return os.path.join(out_dir, "runs", method, run_name(entry) + ".json")


def launcher(nproc):
    """How to start a program on `nproc` GPUs with this Python.

    `python -m torch.distributed.run` is torchrun, but with this Python, so it
    uses the same environment as the runner (a bare `torchrun` may belong to
    the system Python instead).
    """
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
    return [sys.executable]


def generate_args(entry, method, video, ckpt_dir, nproc=4, seed=42,
                  task="i2v-A14B", size="480*832"):
    """generate_fast.py's command-line arguments for one run."""
    args = [
        "--task", task,
        "--size", size,
        "--ckpt_dir", ckpt_dir,
        "--image", entry["image"],
        "--action_path", entry["traj_dir"],
        "--frame_num", str(entry["frame_num"]),
        "--prompt", entry["prompt"],
        "--base_seed", str(seed),
        "--save_dir", os.path.dirname(video),
        "--save_file", video,
    ] + METHODS[method]
    if nproc > 1:
        # Same multi-GPU setup as run_retrieval_compress.sh: split heads across GPUs.
        args += ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(nproc)]
    return args


def build_command(entry, method, video, ckpt_dir, nproc=4, seed=42):
    """The equivalent stand-alone generate_fast.py command for one run.

    Real runs go through eval/worker.py instead (same arguments, model loaded
    once); this command is saved with each run so a single video can be
    reproduced by hand from the repo root.
    """
    return (launcher(nproc) + ["generate_fast.py"]
            + generate_args(entry, method, video, ckpt_dir, nproc, seed))


def fake_generate(entry, video, seed=42):
    """Write a stand-in video: the input picture plus a little noise per frame.

    It has the same number of frames as a real WorldKV video, so the rest of
    the pipeline (pairing, metrics, tables) can be tested without a GPU.
    """
    num_poses = len(pairing.load_poses(entry["traj_dir"]))
    frames = pairing.num_video_frames(num_poses, entry["frame_num"])
    image = iio.imread(entry["image"])[..., :3].astype(np.int16)
    rng = np.random.default_rng(seed + zlib.crc32(run_name(entry).encode()))
    noise = rng.integers(-8, 9, size=(frames,) + image.shape, dtype=np.int16)
    iio.imwrite(video, np.clip(image + noise, 0, 255).astype(np.uint8), fps=16)


def make_job(entry, method, out_dir, args):
    """One video for eval/worker.py: generate_fast.py's arguments and output paths."""
    video = video_path(out_dir, method, entry)
    return {
        "name": f"{method} {run_name(entry)}",
        "argv": generate_args(entry, method, video, args.ckpt_dir, args.nproc, args.seed),
        "save_file": video,
        "telemetry": telemetry_path(out_dir, method, entry),
        "meta": meta_path(out_dir, method, entry),
        "meta_fields": {
            "scene": entry["scene"],
            "trajectory": entry["trajectory"],
            "method": method,
            "fake": False,
            "seed": args.seed,
            "nccl_p2p_disabled": bool(args.no_p2p),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF),
            "command": build_command(entry, method, video, args.ckpt_dir, args.nproc, args.seed),
        },
    }


# Lets PyTorch grow its GPU memory blocks instead of leaving unusable gaps
# between them. It changes only how memory is organized, not what is
# computed, so videos are the same; without it Full KV ran out of memory on
# 46 GB A40s with 1.4 GB stuck in such gaps.
ALLOC_CONF = "expandable_segments:True"


def worker_env(args):
    """Environment for the worker: GPU memory setting, and NCCL P2P if disabled."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    if args.no_p2p:
        env["NCCL_P2P_DISABLE"] = "1"
    return env


def launch_worker(jobs_file, log_path, args):
    """Run eval/worker.py on the jobs file; copy its output to the log, echo progress."""
    cmd = launcher(args.nproc) + ["-m", "eval.worker", jobs_file]
    env = worker_env(args)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()  # keep the log current, so progress can be watched live
            if "[job]" in line or "Model loaded once" in line:
                print("   " + line.strip().split("INFO: ")[-1], flush=True)
        return proc.wait()


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_jobs(jobs, out_dir, args, launch=launch_worker):
    """Generate all jobs with as few model loads as possible.

    One worker loads the model and runs every job. If it stops early (e.g. a
    video runs out of GPU memory), the job it was on is marked failed and a
    new worker runs the rest.
    """
    failed = set()
    attempt = 0
    while True:
        todo = [j for j in jobs if not os.path.exists(j["save_file"]) and j["name"] not in failed]
        if not todo:
            break
        attempt += 1
        jobs_file = os.path.join(out_dir, "runs", f"jobs_{attempt}.json")
        log_path = os.path.join(out_dir, "runs", f"worker_{attempt}.log")
        os.makedirs(os.path.dirname(jobs_file), exist_ok=True)
        with open(jobs_file, "w") as f:
            json.dump(todo, f, indent=1)
        print(f"[worker] {len(todo)} videos, model loaded once (log: {log_path})", flush=True)
        returncode = launch(jobs_file, log_path, args)

        progress = False
        for job in todo:
            meta = read_json(job["meta"]) or {}
            if os.path.exists(job["save_file"]):
                progress = True
            elif meta.get("status") == "running":
                # The worker stopped during this job.
                meta.update(status="failed", returncode=returncode, log=log_path)
                with open(job["meta"], "w") as f:
                    json.dump(meta, f, indent=1)
                failed.add(job["name"])
                progress = True
                print(f"[fail] {job['name']}: worker exit code {returncode}, see {log_path}")
        if not progress:
            print(f"[fail] worker exit code {returncode} before finishing any video, "
                  f"see {log_path}")
            break


def generate_all(entries, methods, out_dir, args, launch=launch_worker):
    """Phase 1: make every missing video. Each run's settings go to runs/*.json."""
    jobs = []
    for method in methods:
        for entry in entries:
            video = video_path(out_dir, method, entry)
            if os.path.exists(video):
                print(f"[skip] {method} {run_name(entry)}: video exists")
                continue
            os.makedirs(os.path.dirname(video), exist_ok=True)
            os.makedirs(os.path.dirname(meta_path(out_dir, method, entry)), exist_ok=True)

            if args.fake:
                if args.dry_run:
                    print(f"[fake] {method} {run_name(entry)}")
                    continue
                print(f"[run]  {method} {run_name(entry)}", flush=True)
                start = time.time()
                fake_generate(entry, video, args.seed)
                with open(meta_path(out_dir, method, entry), "w") as f:
                    json.dump({"scene": entry["scene"], "trajectory": entry["trajectory"],
                               "method": method, "fake": True, "seed": args.seed,
                               "status": "done",
                               "wall_time_s": round(time.time() - start, 1)}, f, indent=1)
                continue

            job = make_job(entry, method, out_dir, args)
            if args.dry_run:
                print(shlex.join(job["meta_fields"]["command"]))
            jobs.append(job)

    if args.dry_run and jobs:
        print(f"(real runs use one worker for all {len(jobs)} videos: "
              f"{shlex.join(launcher(args.nproc) + ['-m', 'eval.worker', 'JOBS.json'])})")
    if jobs and not args.dry_run:
        run_jobs(jobs, out_dir, args, launch)


def read_telemetry(path):
    """FPS and memory columns from a run's telemetry file (None if missing)."""
    row = {"fps_last_chunk": None, "fps_mean": None, "peak_gpu_mem_gb": None,
           "peak_gpu_mem_reserved_gb": None, "peak_cpu_rss_gb": None}
    if not os.path.exists(path):
        return row
    with open(path) as f:
        t = json.load(f)

    def peak(values):
        values = [v for v in values or [] if v is not None]
        return max(values) if values else None

    row["fps_last_chunk"] = t.get("fps_last_chunk")
    row["fps_mean"] = t.get("fps_mean")
    # Largest of the GPUs, since the run fails if any one GPU runs out.
    row["peak_gpu_mem_gb"] = peak(t.get("peak_gpu_mem_allocated_gb"))
    row["peak_gpu_mem_reserved_gb"] = peak(t.get("peak_gpu_mem_reserved_gb"))
    row["peak_cpu_rss_gb"] = peak(t.get("cpu_rss_gb"))
    return row


def score_all(entries, methods, out_dir, device, use_fid=True):
    """Phase 2: score every video. Returns (results rows, summary rows)."""
    pair_metrics = PairMetrics(device)
    results, summary = [], []
    for method in methods:
        set_metrics = {g: SetMetrics(device) for g in METRIC_GROUPS} if use_fid else {}
        ok_rows = []
        for entry in entries:
            row = {"scene": entry["scene"], "trajectory": entry["trajectory"], "method": method}
            meta = read_json(meta_path(out_dir, method, entry)) or {}
            # Generation time of this video; the model load is counted once per worker.
            row["wall_time_s"] = meta.get("wall_time_s")
            row["model_load_s"] = meta.get("model_load_s")
            video = video_path(out_dir, method, entry)
            if not os.path.exists(video):
                row["status"] = meta.get("status", "missing")
                results.append(row)
                continue

            print(f"[score] {method} {run_name(entry)}", flush=True)
            pairs = pairing.build_pairs(entry["traj_dir"], entry["frame_num"])
            pair_rows = evaluate_video(read_video(video), pairs, pair_metrics, set_metrics)
            groups = summarize(pair_rows)
            row["status"] = "ok"
            row["n_pairs"] = groups["all"]["n"]
            row["n_pairs_long"] = groups["long"]["n"]
            for g in METRIC_GROUPS:
                suffix = "" if g == "all" else f"_{g}"
                for name in ("lpips", "psnr", "ssim"):
                    row[name + suffix] = groups[g][name]
            row.update(read_telemetry(telemetry_path(out_dir, method, entry)))
            results.append(row)
            ok_rows.append(row)

            detail = os.path.join(out_dir, "metrics", method, run_name(entry) + ".json")
            os.makedirs(os.path.dirname(detail), exist_ok=True)
            with open(detail, "w") as f:
                json.dump({"summary": groups, "pairs": pair_rows}, f, indent=1)

        # Paired metrics and FPS: mean over videos, so every scene counts equally.
        srow = {"method": method, "n_videos": len(ok_rows)}
        for key in ("lpips", "psnr", "ssim", "lpips_long", "psnr_long", "ssim_long",
                    "fps_last_chunk", "fps_mean"):
            vals = [r[key] for r in ok_rows if r.get(key) is not None]
            srow[key] = float(np.mean(vals)) if vals else None
        # Memory: the largest peak over all videos (what the method needs).
        for key in ("peak_gpu_mem_gb", "peak_gpu_mem_reserved_gb", "peak_cpu_rss_gb"):
            vals = [r[key] for r in ok_rows if r.get(key) is not None]
            srow[key] = float(np.max(vals)) if vals else None
        # Set metrics: pooled over all videos of the method.
        for g, sm in set_metrics.items():
            suffix = "" if g == "all" else f"_{g}"
            res = sm.compute()
            srow["fid" + suffix] = res["fid"]
            srow["kid" + suffix] = res["kid_mean"]
        summary.append(srow)
    return results, summary


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    def fmt(v, spec):
        return format(v, spec) if v is not None else format("-", ">" + spec.split(".")[0])
    print(f"\n{'method':<10} {'videos':>6} {'LPIPS':>7} {'PSNR':>7} {'SSIM':>6} {'FID':>8}"
          f" | {'LPIPS*':>7} {'PSNR*':>7} {'SSIM*':>6} | {'FPS':>6} {'GPU GB':>7}")
    for s in summary:
        print(f"{s['method']:<10} {s['n_videos']:>6} {fmt(s['lpips'], '7.3f')} "
              f"{fmt(s['psnr'], '7.3f')} {fmt(s['ssim'], '6.3f')} {fmt(s.get('fid'), '8.2f')}"
              f" | {fmt(s['lpips_long'], '7.3f')} {fmt(s['psnr_long'], '7.3f')} "
              f"{fmt(s['ssim_long'], '6.3f')} | {fmt(s['fps_last_chunk'], '6.2f')} "
              f"{fmt(s['peak_gpu_mem_gb'], '7.1f')}")
    print("(* = long-gap pairs only; FPS = last chunk; GPU GB = peak allocated, largest GPU)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file.")
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help=f"Comma-separated, from: {', '.join(METHODS)}.")
    parser.add_argument("--ckpt-dir", help="LingBot checkpoint folder (real runs).")
    parser.add_argument("--nproc", type=int, default=4, help="GPUs per run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-p2p", action="store_true",
                        help="Set NCCL_P2P_DISABLE=1 for runs. Needed on machines where "
                             "direct GPU-to-GPU copies over PCIe hang (e.g. RunPod 4x A40). "
                             "Leave off on NVLink machines (H200): it slows them down.")
    parser.add_argument("--fake", action="store_true",
                        help="Use a fake model instead of WorldKV (no GPU needed).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the generation commands and stop.")
    parser.add_argument("--no-fid", action="store_true", help="Skip FID/KID (faster).")
    parser.add_argument("--device", help="cuda, mps or cpu for scoring.")
    args = parser.parse_args()

    methods = args.methods.split(",")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"Unknown methods: {unknown}")
    if not args.fake and not args.ckpt_dir:
        parser.error("--ckpt-dir is required unless --fake is set.")
    args.ckpt_dir = os.path.abspath(args.ckpt_dir) if args.ckpt_dir else None
    out_dir = os.path.abspath(args.out)

    entries = load_benchmark(args.benchmark)
    check_video_lengths(entries, methods)
    print(f"{len(entries)} scene-trajectory pairs x {len(methods)} methods "
          f"= {len(entries) * len(methods)} runs")
    generate_all(entries, methods, out_dir, args)
    if args.dry_run:
        return

    device = torch.device(args.device) if args.device else default_device()
    results, summary = score_all(entries, methods, out_dir, device, use_fid=not args.no_fid)
    write_csv(os.path.join(out_dir, "results.csv"), results)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)
    print_summary(summary)
    print(f"\nWrote {os.path.join(out_dir, 'results.csv')} and summary.csv")


if __name__ == "__main__":
    main()
