"""Check a run of the offline schedule against WorldKV (CPU) on the same input.

Reads the telemetry and videos that eval/runner.py wrote for one run name in
an output folder, for methods "cpu_only" and "offline", and checks:

  1. both retrieve the same chunks, in the same order, at every chunk;
  2. both videos are identical (and equal to --reference, e.g. an earlier
     WorldKV video of the same input);
  3. the offline bank after each chunk is exactly the chunks still needed
     later, at WorldKV's measured size per chunk;
and prints memory, copies and speed of both side by side. Without a cpu_only
run (e.g. a long video run with offline only), it prints the offline numbers.

Usage, from the repo root:
  python -m eval.check_offline OUT_DIR RUN_NAME [--reference VIDEO]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from eval.metrics import read_video


def load(out_dir, method, run):
    path = os.path.join(out_dir, "runs", method, run + ".telemetry.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def video(out_dir, method, run):
    return os.path.join(out_dir, "videos", method, run + ".mp4")


def same_video(path_a, path_b):
    """(identical, detail) for two video files."""
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        if hashlib.sha256(fa.read()).digest() == hashlib.sha256(fb.read()).digest():
            return True, "same file bytes"
    a, b = read_video(path_a), read_video(path_b)
    if a.shape != b.shape:
        return False, f"different shapes {a.shape} vs {b.shape}"
    if np.array_equal(a, b):
        return True, "same decoded frames"
    diff = np.abs(a.astype(int) - b.astype(int))
    frames = int((diff.reshape(len(diff), -1).max(axis=1) > 0).sum())
    return False, f"{frames}/{len(a)} frames differ, max pixel difference {diff.max()}"


def live_chunks(retrieved, num_chunks):
    """Chunks the offline bank holds right after each chunk is stored."""
    last = [-1] * num_chunks
    for t, ids in enumerate(retrieved):
        for c in ids:
            last[c] = max(last[c], t)
    return [sum(1 for c in range(t + 1) if last[c] >= t) for t in range(num_chunks)]


def mean(values):
    values = [v for v in values or [] if v is not None]
    return sum(values) / len(values) if values else None


def numbers(t):
    """The side-by-side numbers of one run's telemetry."""
    peak = lambda values: max(v for v in values if v is not None) if values else None
    return {
        "chunks": t.get("num_chunks"),
        "peak bank (GB, all GPUs)": t.get("peak_bank_gb"),
        "peak CPU RSS total (GB)": t.get("peak_cpu_rss_total_gb"),
        "CPU RSS at start (GB)": t.get("cpu_rss_start_total_gb"),
        "peak GPU reserved (GB, largest GPU)": peak(t.get("peak_gpu_mem_reserved_gb")),
        "copied CPU->GPU (GB, all chunks)": t.get("copied_total_gb"),
        "retrieval time per chunk (s)": mean(t.get("retrieval_s")),
        "time per chunk (s)": mean(t.get("chunk_times_s")),
        "FPS": t.get("fps_mean"),
    }


def check(out_dir, run, reference=None):
    """Returns (list of (ok, message), table rows)."""
    cpu, off = load(out_dir, "cpu_only", run), load(out_dir, "offline", run)
    if off is None:
        return [(False, f"no offline telemetry for {run} in {out_dir}")], []
    results = []
    if cpu is not None:
        same = cpu["retrieved_chunks"] == off["retrieved_chunks"]
        results.append((same, "same retrieved chunks at every chunk"))
        ok, detail = same_video(video(out_dir, "cpu_only", run), video(out_dir, "offline", run))
        results.append((ok, f"offline video equals cpu_only video ({detail})"))
        # WorldKV stores every chunk, so its bank gives the size of one chunk.
        chunk_gb = cpu["bank_gb"][-1] / len(cpu["bank_gb"])
        expected = [n * chunk_gb for n in live_chunks(off["retrieved_chunks"], len(off["bank_gb"]))]
        exact = all(abs(b - e) < 1e-6 for b, e in zip(off["bank_gb"], expected))
        results.append((exact, f"offline bank = chunks still needed x {chunk_gb:.3f} GB per chunk"))
    if reference:
        ok, detail = same_video(reference, video(out_dir, "offline", run))
        results.append((ok, f"offline video equals {os.path.basename(reference)} ({detail})"))
    rows = [(name, numbers(cpu)[name] if cpu else None, value)
            for name, value in numbers(off).items()]
    return results, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", help="The runner's output folder.")
    parser.add_argument("run", help="Run name, e.g. 04_RLWSRL90_245_RLWSRL90_245.")
    parser.add_argument("--reference", help="Another video the offline video must equal.")
    args = parser.parse_args()
    results, rows = check(args.out_dir, args.run, args.reference)
    for ok, message in results:
        print(("PASS " if ok else "FAIL ") + message)
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"\n{'':38} {'cpu_only':>10} {'offline':>10}")
    for name, cpu_value, off_value in rows:
        print(f"{name:38} {fmt(cpu_value):>10} {fmt(off_value):>10}")
    sys.exit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()
