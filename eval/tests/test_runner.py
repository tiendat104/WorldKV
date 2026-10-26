import argparse
import ast
import importlib.util
import json
import os
import re

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from eval import runner
from generate_traj import build_yaw_series, make_poses

# Load wan/modules/kv_modes.py on its own: importing the `wan` package pulls in
# the full model code and its GPU-only dependencies.
_spec = importlib.util.spec_from_file_location(
    "kv_modes", os.path.join(runner.REPO_ROOT, "wan", "modules", "kv_modes.py"))
_kv_modes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kv_modes)
validate_kv_mode_flags = _kv_modes.validate_kv_mode_flags

DEV_BENCHMARK = os.path.join(runner.REPO_ROOT, "eval", "tests", "fixtures", "dev", "benchmark.json")


def flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_dev_benchmark_loads():
    entries = runner.load_benchmark(DEV_BENCHMARK)
    assert [e["scene"] for e in entries] == ["00", "15", "13", "uni01", "uni03"]
    for e in entries:
        assert os.path.exists(e["image"])
        assert os.path.isdir(e["traj_dir"])
        assert e["prompt"]
    assert entries[0]["frame_num"] == 360


def test_commands_use_known_flags_and_one_kv_mode():
    # Every flag we pass must exist in generate_fast.py, and each method must
    # pick at most one KV mode (the check generate_fast.py itself runs).
    with open(os.path.join(runner.REPO_ROOT, "generate_fast.py")) as f:
        known = set(re.findall(r'"(--[a-z_0-9]+)"', f.read()))
    entry = runner.load_benchmark(DEV_BENCHMARK)[0]
    for method in runner.METHODS:
        cmd = runner.build_command(entry, method, "/tmp/x.mp4", "/ckpt", nproc=4)
        assert cmd[1:4] == ["-m", "torch.distributed.run", "--nproc_per_node=4"]
        flags = {a for a in cmd if a.startswith("--") and not a.startswith("--nproc")}
        assert flags <= known, flags - known
        validate_kv_mode_flags(
            full_kv="--full_kv" in cmd,
            sliding_window=int(flag_value(cmd, "--sliding_window") or 0),
            use_retrieval="--use_retrieval" in cmd)


def test_worldkv_settings_match_paper():
    gpu = runner.build_command({"image": "a.png", "traj_dir": "t", "frame_num": 360,
                                "prompt": "p"}, "gpu_only", "v.mp4", "/ckpt", nproc=1)
    cpu = runner.build_command({"image": "a.png", "traj_dir": "t", "frame_num": 360,
                                "prompt": "p"}, "cpu_only", "v.mp4", "/ckpt", nproc=1)
    assert flag_value(gpu, "--retrieval_frames") == "18"  # 6 chunks
    assert flag_value(gpu, "--kv_compression_keep_ratio") == "0.25"  # 3 -> 1.5 frames
    # GPU-only and CPU-only differ only in where the KV bank is stored.
    assert [a for a in gpu if a != "--kv_bank_on_gpu"] == cpu


def test_short_videos_rejected_for_worldkv(tmp_path):
    entries = runner.load_benchmark(make_tiny_benchmark(str(tmp_path / "bench")))
    # 100 frames = 24 latent frames, below WorldKV's 27-frame window.
    with pytest.raises(ValueError, match="latent frames"):
        runner.check_video_lengths(entries, ["sliding", "gpu_only"])
    runner.check_video_lengths(entries, ["sliding", "full_kv"])  # no retrieval: fine
    # The dev and smoke benchmarks are long enough.
    runner.check_video_lengths(runner.load_benchmark(DEV_BENCHMARK), ["gpu_only"])
    smoke = os.path.join(runner.REPO_ROOT, "eval", "tests", "fixtures", "smoke", "benchmark.json")
    runner.check_video_lengths(runner.load_benchmark(smoke), ["gpu_only"])


def generate_call_keywords(path):
    """Keyword names of the `.generate(...)` call in a source file."""
    with open(path) as f:
        tree = ast.parse(f.read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"]
    assert len(calls) == 1, path
    return {k.arg for k in calls[0].keywords}


def test_worker_calls_generate_like_generate_fast():
    # eval/worker.py must pass the same settings to generate() as
    # generate_fast.py, so both make the same videos.
    worker = generate_call_keywords(os.path.join(runner.REPO_ROOT, "eval", "worker.py"))
    reference = generate_call_keywords(os.path.join(runner.REPO_ROOT, "generate_fast.py"))
    assert worker == reference


def test_worker_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    env = runner.worker_env(argparse.Namespace(no_p2p=False))
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "NCCL_P2P_DISABLE" not in env
    assert runner.worker_env(argparse.Namespace(no_p2p=True))["NCCL_P2P_DISABLE"] == "1"
    # A value set by the user is kept.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    assert runner.worker_env(argparse.Namespace(no_p2p=False))["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"


def test_worker_restarted_after_a_crash(tmp_path):
    entries = runner.load_benchmark(DEV_BENCHMARK)[:3]
    out = str(tmp_path / "out")
    args = argparse.Namespace(fake=False, dry_run=False, seed=42, ckpt_dir="/ckpt",
                              nproc=4, no_p2p=True)
    jobs = [runner.make_job(e, "sliding", out, args) for e in entries]
    calls = []

    def fake_worker(jobs_file, log_path, args):
        # Finishes the first job it gets and stops (crashes) on the second.
        with open(jobs_file) as f:
            todo = json.load(f)
        calls.append([j["name"] for j in todo])
        for k, job in enumerate(todo):
            os.makedirs(os.path.dirname(job["meta"]), exist_ok=True)
            with open(job["meta"], "w") as f:
                json.dump({"status": "running"}, f)
            if k == 1 and len(calls) == 1:
                return 1
            os.makedirs(os.path.dirname(job["save_file"]), exist_ok=True)
            open(job["save_file"], "w").close()
            with open(job["meta"], "w") as f:
                json.dump({"status": "done"}, f)
        return 0

    runner.run_jobs(jobs, out, args, launch=fake_worker)
    names = [j["name"] for j in jobs]
    # First worker: all 3 jobs; it crashes on job 2. Second worker: only job 3.
    assert calls == [names, names[2:]]
    assert os.path.exists(jobs[0]["save_file"]) and os.path.exists(jobs[2]["save_file"])
    with open(jobs[1]["meta"]) as f:
        assert json.load(f)["status"] == "failed"
    # Each job carries the same arguments a stand-alone generate_fast.py run would get.
    assert jobs[0]["meta_fields"]["command"][-len(jobs[0]["argv"]):] == jobs[0]["argv"]


def make_tiny_benchmark(root):
    """One scene (a 32x48 picture) and one short turn-right-then-left trajectory."""
    os.makedirs(os.path.join(root, "images"))
    rng = np.random.default_rng(0)
    iio.imwrite(os.path.join(root, "images", "s0.png"),
                rng.integers(0, 256, (32, 48, 3), dtype=np.uint8))
    traj = "R_L90_100"
    yaw, xyz, *_ = build_yaw_series(["R", "L"], 90.0, [], translation_distance=1.0)
    yaw = np.concatenate([yaw, np.full(10, yaw[-1])])
    xyz = np.concatenate([xyz, np.repeat(xyz[-1:], 10, axis=0)])
    os.makedirs(os.path.join(root, "traj", traj))
    np.save(os.path.join(root, "traj", traj, f"{traj}_poses.npy"), make_poses(yaw, xyz))
    with open(os.path.join(root, "prompts.json"), "w") as f:
        json.dump({"s0": "a test scene"}, f)
    path = os.path.join(root, "benchmark.json")
    with open(path, "w") as f:
        json.dump({"name": "tiny", "images_dir": "images", "prompts": "prompts.json",
                   "traj_dir": "traj", "pairs": [{"scene": "s0", "trajectory": traj}]}, f)
    return path


def test_fake_run_end_to_end(tmp_path):
    entries = runner.load_benchmark(make_tiny_benchmark(str(tmp_path / "bench")))
    out = str(tmp_path / "out")
    args = argparse.Namespace(fake=True, dry_run=False, seed=42, ckpt_dir=None, nproc=1,
                              no_p2p=False)
    methods = ["sliding", "gpu_only"]

    runner.generate_all(entries, methods, out, args)
    video = runner.video_path(out, "sliding", entries[0])
    assert len(iio.imread(video)) == 93  # 100 frames -> 24 latent -> 93 video frames
    mtime = os.path.getmtime(video)

    runner.generate_all(entries, methods, out, args)  # second call skips finished videos
    assert os.path.getmtime(video) == mtime

    # Stand-in telemetry for one method, as generate_fast.py would write it.
    with open(runner.telemetry_path(out, "gpu_only", entries[0]), "w") as f:
        json.dump({"fps_last_chunk": 4.8, "fps_mean": 5.0,
                   "peak_gpu_mem_allocated_gb": [50.0, 52.5, 51.0, 50.5],
                   "peak_gpu_mem_reserved_gb": [55.0, 56.0, 55.5, 55.0],
                   "cpu_rss_gb": [10.0, 12.0]}, f)

    results, summary = runner.score_all(entries, methods, out, torch.device("cpu"),
                                        use_fid=False)
    assert [r["method"] for r in results] == methods
    assert all(r["status"] == "ok" and r["n_pairs"] > 0 for r in results)
    assert [s["n_videos"] for s in summary] == [1, 1]

    sliding, gpu_only = results
    assert sliding["fps_last_chunk"] is None  # fake runs have no telemetry
    assert gpu_only["fps_last_chunk"] == 4.8
    assert gpu_only["peak_gpu_mem_gb"] == 52.5  # the largest of the 4 GPUs
    assert gpu_only["peak_cpu_rss_gb"] == 12.0
    assert summary[1]["fps_last_chunk"] == 4.8
