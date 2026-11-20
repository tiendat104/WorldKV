"""Write the experiments' camera paths and run lists.

Run from the repo root (rewrites eval/experiments/paths/ and the run lists):
  python -m eval.experiments.prepare

Run lists (benchmark files for eval/runner.py; scenes and prompts come from
eval/benchmark/), see Documents/experiment_design.md for the designs:
  e1.json          E1: 4 benchmark entries, one per trajectory type
  e2_long.json     E2: one 40-chunk video (WorldKV GPU / CPU, ours)
  e2_full_kv.json  E2: the same path at several lengths (Full KV)
  e3.json          E3: 4 natural 40 s paths
"""

import json
import os

import numpy as np

import generate_traj as gt
from eval.experiments import natural_paths

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_DIR = natural_paths.PATHS_DIR

E1_PAIRS = [
    ("04_RLWSRL90_245", "RLWSRL90_245"),
    ("24_WSRLRL90_245", "WSRLRL90_245"),
    ("20_R_L_W_S_R_L90_318", "R_L_W_S_R_L90_318"),
    ("21_L_R_W_S_L_R90_318", "L_R_W_S_L_R90_318"),
]
# One scene for E2 and E3: the authors' own input picture (memory does not
# depend on the picture).
LONG_SCENE = "00_R_L_R_L_R_L90_360_compressed"
# E2: look right 90 deg and back, 4 times, with 16-frame pauses (generate_traj.py).
E2_TRAJECTORY, E2_NOOP_FRAMES = "R_L_R_L_R_L_R_L90", 16
E2_CHUNKS = 40
FULL_KV_CHUNKS = [9, 18, 26, 27, 28, 29, 30]


def frames_for(chunks):
    """Video frames of a video of `chunks` chunks (3 latent frames each)."""
    return 12 * chunks - 3


def e2_poses(frames):
    """The E2 path exactly as `generate_traj.py --frames <frames> --trajectory
    R_L_R_L_R_L_R_L90 --noop-frames 16` writes it."""
    ops, degree = gt.parse_sequence(E2_TRAJECTORY)
    noops = gt.parse_noop_frames(str(E2_NOOP_FRAMES), ops.count("_"))
    yaw, xyz, *_ = gt.build_yaw_series(ops, degree, noops, translation_distance=1.0)
    pad = frames - len(yaw)
    yaw = np.concatenate([yaw, np.full(pad, yaw[-1], dtype=np.float32)])
    xyz = np.concatenate([xyz, np.tile(xyz[-1:], (pad, 1)).astype(np.float32)])
    return gt.make_poses(yaw, xyz)


def write_e2_paths():
    """The 40-chunk path, and its first 12n - 3 frames for each Full KV length
    (generate_traj.py refuses lengths shorter than the moves, so they are cut)."""
    full = e2_poses(frames_for(E2_CHUNKS))
    for chunks in sorted(set(FULL_KV_CHUNKS + [E2_CHUNKS])):
        frames = frames_for(chunks)
        natural_paths.write(f"{E2_TRAJECTORY}_{frames}", full[:frames])


def write_run_list(file_name, pairs, traj_dir="paths"):
    run_list = {"name": os.path.splitext(file_name)[0],
                "images_dir": "../benchmark/final_images",
                "prompts": "../benchmark/prompts.json", "traj_dir": traj_dir,
                "pairs": [{"scene": scene, "trajectory": traj} for scene, traj in pairs]}
    with open(os.path.join(HERE, file_name), "w") as f:
        json.dump(run_list, f, indent=1)
        f.write("\n")


def main():
    natural_paths.main()  # E3 paths
    write_e2_paths()
    write_run_list("e1.json", E1_PAIRS, traj_dir="../benchmark/traj")
    write_run_list("e2_long.json",
                   [(LONG_SCENE, f"{E2_TRAJECTORY}_{frames_for(E2_CHUNKS)}")])
    write_run_list("e2_full_kv.json",
                   [(LONG_SCENE, f"{E2_TRAJECTORY}_{frames_for(n)}") for n in FULL_KV_CHUNKS])
    write_run_list("e3.json", [(LONG_SCENE, f"{name}_{natural_paths.FRAMES}")
                               for name in natural_paths.SCENARIOS])
    print("wrote", ", ".join(sorted(f for f in os.listdir(HERE) if f.endswith(".json"))))


if __name__ == "__main__":
    main()
