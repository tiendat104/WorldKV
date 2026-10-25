"""Find (first visit, revisit) frame pairs in a camera trajectory.

Two video frames form a pair when the camera has the same position and
viewing direction in both. Each revisit is paired with the earliest frame
at that pose. Only the trajectory's poses file is needed: no video, no GPU.

Frame numbering follows WorldKV (wan/image2video_fast.py):
  - the trajectory is trimmed to F = 4 * (lat_f - 1) + 1 video frames,
    where lat_f is a multiple of the chunk size (3);
  - video frame j uses camera pose j;
  - latent frame 0 is video frame 0, latent k covers video frames 4k-3 .. 4k.

Usage:
  python -m eval.pairing traj/R_L_R_L_R_L90_360 --frame-num 360 -o pairs.json
"""

import argparse
import glob
import json
import os

import numpy as np

VAE_STRIDE_T = 4  # video frames per latent frame (after the first)
CHUNK_SIZE = 3  # latent frames per generated chunk
SINK_LATENT_FRAMES = 3  # first latent frames kept in attention (paper, Sec. 5.1)


def load_poses(traj_dir):
    """Load the (N, 4, 4) camera-to-world poses from a trajectory folder."""
    matches = glob.glob(os.path.join(traj_dir, "*poses.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one *poses.npy in {traj_dir}, found {len(matches)}.")
    return np.load(matches[0]).astype(np.float64)


def num_video_frames(num_poses, frame_num, chunk_size=CHUNK_SIZE):
    """Number of frames in the generated video, matching WorldKV's trimming."""
    len_c2ws = ((num_poses - 1) // VAE_STRIDE_T) * VAE_STRIDE_T + 1
    frames = ((frame_num - 1) // VAE_STRIDE_T) * VAE_STRIDE_T + 1
    frames = min(frames, len_c2ws)
    lat_f = (frames - 1) // VAE_STRIDE_T + 1
    lat_f -= lat_f % chunk_size
    return (lat_f - 1) * VAE_STRIDE_T + 1


def latent_index(frame):
    """Latent frame that contains the given video frame."""
    return 0 if frame == 0 else (frame - 1) // VAE_STRIDE_T + 1


def rotation_distance_deg(rots, rot):
    """Angle in degrees between each rotation in `rots` (M, 3, 3) and `rot`."""
    cos = (np.einsum("mij,ij->m", rots, rot) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def find_pairs(poses, rot_tol_deg=0.5, trans_tol=1e-3,
               sink_latent_frames=SINK_LATENT_FRAMES):
    """Pair every frame with the earliest earlier frame at the same pose.

    Returns a list of dicts with keys:
      first, revisit:  video frame indices (first < revisit)
      gap_frames:      revisit - first, in video frames
      gap_latent:      the same gap in latent frames
      in_sink:         True if the first visit is inside the sink, which the
                       model never evicts (an easy revisit)
    """
    rots = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    pairs = []
    for j in range(1, len(poses)):
        rot_dist = rotation_distance_deg(rots[:j], rots[j])
        trans_dist = np.linalg.norm(trans[:j] - trans[j], axis=1)
        same = np.flatnonzero((rot_dist <= rot_tol_deg) & (trans_dist <= trans_tol))
        if len(same) == 0:
            continue
        i = int(same[0])
        pairs.append({
            "first": i,
            "revisit": j,
            "gap_frames": j - i,
            "gap_latent": latent_index(j) - latent_index(i),
            "in_sink": latent_index(i) < sink_latent_frames,
        })
    return pairs


def build_pairs(traj_dir, frame_num, **kwargs):
    """Load a trajectory, trim it like WorldKV, and find its pairs."""
    poses = load_poses(traj_dir)
    frames = num_video_frames(len(poses), frame_num)
    return {
        "trajectory": os.path.basename(os.path.normpath(traj_dir)),
        "num_video_frames": frames,
        "sink_latent_frames": kwargs.get("sink_latent_frames", SINK_LATENT_FRAMES),
        "pairs": find_pairs(poses[:frames], **kwargs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traj_dir", help="Trajectory folder with a *poses.npy file.")
    parser.add_argument("--frame-num", type=int, required=True,
                        help="The --frame_num passed to generate_fast.py.")
    parser.add_argument("--rot-tol-deg", type=float, default=0.5)
    parser.add_argument("--trans-tol", type=float, default=1e-3)
    parser.add_argument("-o", "--output", help="Write the pairs to this JSON file.")
    args = parser.parse_args()

    result = build_pairs(args.traj_dir, args.frame_num,
                         rot_tol_deg=args.rot_tol_deg, trans_tol=args.trans_tol)
    pairs = result["pairs"]
    print(f"{result['trajectory']}: {result['num_video_frames']} video frames, "
          f"{len(pairs)} pairs, {sum(p['in_sink'] for p in pairs)} in sink")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=1)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
