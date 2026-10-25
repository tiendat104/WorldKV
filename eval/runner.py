"""Run a benchmark: one video per (scene, trajectory) pair and method, then score.

Phase 1 generates every missing video with WorldKV's generate_fast.py, or
with a fake model on a laptop (--fake). Videos that already exist are
skipped, so a stopped run can simply be restarted.
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
METRIC_GROUPS = ("all", "long")


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


def build_command(entry, method, video, ckpt_dir, nproc=4, seed=42,
                  task="i2v-A14B", size="480*832"):
    """The generate_fast.py command for one run (run it from the repo root)."""
    cmd = [
        "generate_fast.py",
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
        return (["torchrun", f"--nproc_per_node={nproc}"] + cmd
                + ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(nproc)])
    return [sys.executable] + cmd


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


def generate_all(entries, methods, out_dir, args):
    """Phase 1: make every missing video. Each run's settings go to runs/*.json."""
    for method in methods:
        for entry in entries:
            video = video_path(out_dir, method, entry)
            meta_path = os.path.join(out_dir, "runs", method, run_name(entry) + ".json")
            if os.path.exists(video):
                print(f"[skip] {method} {run_name(entry)}: video exists")
                continue
            os.makedirs(os.path.dirname(video), exist_ok=True)
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)

            cmd = None if args.fake else build_command(
                entry, method, video, args.ckpt_dir, args.nproc, args.seed)
            if args.dry_run:
                print(shlex.join(cmd) if cmd else f"[fake] {method} {run_name(entry)}")
                continue

            print(f"[run]  {method} {run_name(entry)}", flush=True)
            start = time.time()
            if args.fake:
                fake_generate(entry, video, args.seed)
                returncode = 0
            else:
                log_path = os.path.join(out_dir, "runs", method, run_name(entry) + ".log")
                # Tells eval/telemetry.py (inside generate_fast.py) where to save timing/memory.
                env = dict(os.environ, **{ENV_VAR: telemetry_path(out_dir, method, entry)})
                with open(log_path, "w") as log:
                    returncode = subprocess.call(cmd, cwd=REPO_ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT)
            meta = {
                "scene": entry["scene"],
                "trajectory": entry["trajectory"],
                "method": method,
                "fake": args.fake,
                "seed": args.seed,
                "command": cmd,
                "returncode": returncode,
                "wall_time_s": round(time.time() - start, 1),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)
            if returncode != 0:
                print(f"[fail] {method} {run_name(entry)}: exit code {returncode}, "
                      f"see {meta_path.replace('.json', '.log')}")


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
            meta_path = os.path.join(out_dir, "runs", method, run_name(entry) + ".json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    row["wall_time_s"] = json.load(f).get("wall_time_s")
            video = video_path(out_dir, method, entry)
            if not os.path.exists(video):
                row["status"] = "missing"
                results.append(row)
                continue

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
