"""Predict the experiments' memory and copy numbers before running them.

WorldKV picks the chunks to retrieve from camera poses alone, so for each run
the retrieval schedule, and with it the bank and the CPU -> GPU copies, follow
from the camera path. This uses the same pose processing and the same schedule
code as generation (wan/retrieval_schedule.py):

  - bank right after each chunk is stored: WorldKV (CPU) holds every chunk; ours
    holds the stored chunks a later chunk still retrieves;
  - copies: WorldKV (CPU) copies all retrieved chunks every chunk; ours copies
    only the ones the previous chunk did not retrieve (the rest are reused on
    the GPU).

GPU memory (E2) does not depend on the path: it is predicted from straight
lines through measured runs on 4x A40.

Run from the repo root:
  python -m eval.experiments.predict
"""

import importlib.util
import os

import numpy as np
import torch

from eval import pairing
from eval.experiments import natural_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_TRAJ = os.path.join(REPO_ROOT, "eval", "benchmark", "traj")

# WorldKV's window in our runs (eval/runner.py WORLDKV_FLAGS, model sink 3 frames):
# sink 1 chunk, recent 2 chunks, 6 retrieved chunks; one chunk = 3 latent frames.
CHUNK, SINK_CHUNKS, RECENT_CHUNKS, NUM_RETRIEVE = 3, 1, 2, 6
# One stored chunk over the 4 GPUs, measured in the smoke test (2026-11-17).
CHUNK_GB = 1.760
FPS = 16


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, *path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded from their files: importing the wan package would load the model code.
cam_utils = _load("cam_utils", "wan", "utils", "cam_utils.py")
schedule_lib = _load("retrieval_schedule", "wan", "retrieval_schedule.py")


def latent_poses(poses, frame_num):
    """Absolute latent-frame poses, with the same steps as wan/image2video_fast.py."""
    len_c2ws = ((len(poses) - 1) // 4) * 4 + 1
    c2ws = poses[:min(((frame_num - 1) // 4) * 4 + 1, len_c2ws)]
    lat_f = (len(c2ws) - 1) // 4 + 1
    lat_f -= lat_f % CHUNK
    n = len(c2ws)
    return cam_utils.interpolate_camera_poses(
        src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, n - 1, lat_f))


def predict(traj_dir, frame_num):
    """Bank and copy numbers of one run (frame_num as the runner passes it)."""
    c2ws = latent_poses(pairing.load_poses(traj_dir), frame_num)
    num_chunks = len(c2ws) // CHUNK
    schedule = schedule_lib.retrieval_schedule(c2ws, num_chunks, CHUNK, SINK_CHUNKS,
                                               RECENT_CHUNKS, NUM_RETRIEVE)
    last = schedule_lib.last_uses(schedule, num_chunks)
    ours_bank = [sum(1 for c in range(t + 1) if last[c] >= t) for t in range(num_chunks)]
    copies_cpu = sum(len(ids) for ids in schedule)
    copies_ours = sum(len(set(ids) - set(schedule[t - 1] if t else []))
                      for t, ids in enumerate(schedule))
    frames = 4 * (num_chunks * CHUNK - 1) + 1
    return {
        "frames": frames, "seconds": frames / FPS, "chunks": num_chunks,
        "never_retrieved": last.count(-1),
        "bank_cpu_chunks": num_chunks, "bank_ours_chunks": max(ours_bank),
        "bank_cpu_gb": num_chunks * CHUNK_GB, "bank_ours_gb": max(ours_bank) * CHUNK_GB,
        "bank_saving": 1 - max(ours_bank) / num_chunks,
        "copies_cpu_chunks": copies_cpu, "copies_ours_chunks": copies_ours,
        "copies_cpu_gb": copies_cpu * CHUNK_GB, "copies_ours_gb": copies_ours * CHUNK_GB,
        "bank_ours_per_chunk": ours_bank,
    }


# E1: 4 benchmark entries (trajectory name ends with the frame_num the runner uses).
E1 = ["RLWSRL90_245", "WSRLRL90_245", "R_L_W_S_R_L90_318", "L_R_W_S_L_R90_318"]
# E3: the 4 natural paths, 40 s each.
E3 = list(natural_paths.SCENARIOS)

# E2: peak GPU memory per GPU (reserved, largest GPU), measured on 4x A40:
# (chunks, GB) points; methods whose memory does not grow use a flat level.
MEASURED_GPU = {
    "Full KV": [(20, 37.60), (26, 42.98)],          # replication, 12 runs
    "WorldKV (GPU)": [(20, 39.26), (26, 41.95)],    # replication, 12 runs
    "WorldKV (CPU)": [(20, 30.86)],                 # smoke test
    # Flat within each run (two fixed-size GPU buffers); the 60 s smoke run
    # peaked at 34.39 GB, the short one at 33.42 GB: the higher one is used.
    "Ours": [(80, 34.39)],
}
GPU_LIMIT_GB = 44.5  # usable on one A40 (45.0 GiB total, minus the CUDA context)


def gpu_line(points):
    """(slope per chunk, value at 0 chunks) through measured points."""
    if len(points) == 1:
        return 0.0, points[0][1]
    (x0, y0), (x1, y1) = points[0], points[-1]
    slope = (y1 - y0) / (x1 - x0)
    return slope, y0 - slope * x0


def predict_gpu(chunks):
    """Predicted peak GPU memory per GPU for each method at a video length."""
    out = {}
    for method, points in MEASURED_GPU.items():
        slope, base = gpu_line(points)
        out[method] = base + slope * chunks
    return out


def first_out_of_memory(method, limit=GPU_LIMIT_GB):
    """First video length (chunks) whose predicted peak exceeds the GPU, or None."""
    slope, base = gpu_line(MEASURED_GPU[method])
    if slope <= 0:
        return None
    return int(np.floor((limit - base) / slope)) + 1


def main():
    print("E1: short benchmark videos (ours vs WorldKV (CPU); WorldKV (GPU) has no CPU bank)")
    print(f"{'trajectory':20} {'s':>5} {'chunks':>6} {'bank CPU':>14} {'bank ours':>14} {'saving':>6}"
          f" {'copies CPU':>16} {'copies ours':>16}")
    for name in E1:
        r = predict(os.path.join(BENCH_TRAJ, name), int(name.rsplit("_", 1)[1]))
        print(f"{name:20} {r['seconds']:5.1f} {r['chunks']:6d} "
              f"{r['bank_cpu_chunks']:3d} ({r['bank_cpu_gb']:5.1f} GB) "
              f"{r['bank_ours_chunks']:3d} ({r['bank_ours_gb']:5.1f} GB) {r['bank_saving']:6.0%} "
              f"{r['copies_cpu_chunks']:4d} ({r['copies_cpu_gb']:6.1f} GB) "
              f"{r['copies_ours_chunks']:4d} ({r['copies_ours_gb']:6.1f} GB)")

    print("\nE2: peak GPU memory per GPU (GB), from measured runs; A40 usable ~44.5 GB")
    lengths = [9, 18, 26, 27, 28, 29, 30, 32, 34, 40]
    print(f"{'method':15}" + "".join(f"{n:>7d}" for n in lengths) + "   first out of memory")
    for method in MEASURED_GPU:
        row = "".join(f"{predict_gpu(n)[method]:7.1f}" for n in lengths)
        oom = first_out_of_memory(method)
        oom_text = f"{oom} chunks ({(12 * oom - 3) / FPS:.1f} s)" if oom else "none"
        print(f"{method:15}{row}   {oom_text}")
    print(f"{'(chunks)':15}" + "".join(f"{(12 * n - 3) / FPS:6.1f}s" for n in lengths))

    print("\nE3: 40 s natural paths (ours vs WorldKV (CPU))")
    for name in E3:
        folder = os.path.join(natural_paths.PATHS_DIR, f"{name}_{natural_paths.FRAMES}")
        r = predict(folder, natural_paths.FRAMES)
        print(f"{name:8} {r['seconds']:5.1f} s {r['chunks']:3d} chunks | bank "
              f"{r['bank_cpu_chunks']:3d} -> {r['bank_ours_chunks']:3d} chunks "
              f"({r['bank_cpu_gb']:5.1f} -> {r['bank_ours_gb']:5.1f} GB, saving {r['bank_saving']:.0%}) | "
              f"copies {r['copies_cpu_chunks']:3d} -> {r['copies_ours_chunks']:3d} chunks "
              f"({r['copies_cpu_gb']:5.1f} -> {r['copies_ours_gb']:5.1f} GB)")
    print(f"(GB over the 4 GPUs at {CHUNK_GB} GB per stored chunk, measured)")


if __name__ == "__main__":
    main()
