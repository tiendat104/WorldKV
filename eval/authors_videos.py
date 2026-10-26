"""Score the WorldKV authors' released demo videos with our metric code.

The authors' videos are their own outputs, so scoring them checks our
measuring tools against a known answer: Table 1's order (Sliding Window
worst, WorldKV ~ Full KV) should come out.

Steps:
  1. download the videos from the authors' repo (pinned to one commit);
  2. recreate each trajectory from the file name with generate_traj.py
     (16-frame pauses for "_", as in the authors' own trajectory file);
  3. black out the on-screen W/A/S/D key overlay (bottom-left corner);
  4. score every video: PSNR, SSIM, LPIPS per pair, FID over all pairs;
  5. shift check: score the pairs again with the revisit frame moved by
     -4..+4 frames. If the recreated trajectory lines up with the video,
     the unshifted pairs (shift 0) score best.

Usage:
  python -m eval.authors_videos --out eval/outputs/authors_videos --workers 3
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "315be40cfe601b724d05bc2d4ba4ff676f10426d"  # cvlab-kaist/WorldKV, main branch
RAW = f"https://raw.githubusercontent.com/cvlab-kaist/WorldKV/{COMMIT}/world_videos/"
GITHUB = f"https://github.com/cvlab-kaist/WorldKV/blob/{COMMIT}/world_videos/"
# Region (x0, y0, x1, y1) covering the key overlay; the same mask is applied to
# both frames of every pair, so it removes the overlay without favoring a method.
OVERLAY = (0, 300, 200, 464)
SHIFTS = range(-4, 5)

# (path under world_videos/, scene, trajectory, method, has key overlay)
VIDEOS = [
    ("worldkvfullkv/00_R_L_R_L_R_L90_360_full.mp4", "00", "R_L_R_L_R_L90_360", "full_kv", False),
    ("worldkvfullkv/00_R_L_R_L_R_L90_360_compressed.mp4", "00", "R_L_R_L_R_L90_360", "worldkv", False),
    ("worldkvfullkv/15_R_L_R_L_R_L90_360_Full_form.mp4", "15", "R_L_R_L_R_L90_360", "full_kv", False),
    ("worldkvfullkv/15_R_L_R_L_R_L90_360_compressed.mp4", "15", "R_L_R_L_R_L90_360", "worldkv", False),
    ("dynamic/04_RLWSRL90_245_sliding.mp4", "04", "RLWSRL90_245", "sliding", True),
    ("dynamic/04_RLWSRL90_245_full.mp4", "04", "RLWSRL90_245", "full_kv", True),
    ("dynamic/04_RLWSRL90_245.mp4", "04", "RLWSRL90_245", "worldkv", True),
    ("dynamic/04_WSRLRL90_245_full.mp4", "04", "WSRLRL90_245", "full_kv", True),
    ("dynamic/04_WSRLRL90_245.mp4", "04", "WSRLRL90_245", "worldkv", True),
    # Single-method videos (WorldKV only)
    ("dynamic/01_R_L_W_S_R_L90_318.mp4", "01", "R_L_W_S_R_L90_318", "worldkv", True),
    ("dynamic/20_R_L_W_S_R_L90_318.mp4", "20", "R_L_W_S_R_L90_318", "worldkv", True),
    ("dynamic/21_L_R_W_S_L_R90_318.mp4", "21", "L_R_W_S_L_R90_318", "worldkv", True),
    ("dynamic/23_L_R_W_S_L_R90_318.mp4", "23", "L_R_W_S_L_R90_318", "worldkv", True),
    ("dynamic/23_RLWSRL90_245.mp4", "23", "RLWSRL90_245", "worldkv", True),
    ("dynamic/24_WSRLRL90_245.mp4", "24", "WSRLRL90_245", "worldkv", True),
    ("dynamic/26_L_R_W_S_L_R90_318.mp4", "26", "L_R_W_S_L_R90_318", "worldkv", True),
    ("dynamic/29_L_R_W_S_L_R90_318.mp4", "29", "L_R_W_S_L_R90_318", "worldkv", True),
    ("dynamic/31_WSRLRL90_245.mp4", "31", "WSRLRL90_245", "worldkv", True),
    ("dynamic/37_RLWSRL90_245.mp4", "37", "RLWSRL90_245", "worldkv", True),
    ("dynamic/41_RLWSRL90_245.mp4", "41", "RLWSRL90_245", "worldkv", True),
    ("dynamic/uni01_L_R_W_S_L_R90_318.mp4", "uni01", "L_R_W_S_L_R90_318", "worldkv", False),
]


def make_trajectory(name, traj_dir):
    """Recreate a trajectory folder from its name with the authors' generate_traj.py."""
    if os.path.isdir(os.path.join(traj_dir, name)):
        return
    pattern, frames = name.rsplit("_", 1)
    cmd = [sys.executable, "generate_traj.py", "--frames", frames, "--trajectory", pattern,
           "--output-dir", traj_dir]
    if "_" in pattern:
        cmd += ["--noop-frames", "16"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)


def shift_check(video, pairs, shifts=SHIFTS, scale=4):
    """Mean PSNR of the pairs with the revisit moved by each shift (frames downscaled)."""
    small = video[:, ::scale, ::scale].astype(np.float32) / 255.0
    out = {}
    for k in shifts:
        vals = []
        for p in pairs:
            j = p["revisit"] + k
            if 0 <= j < len(small):
                mse = np.mean((small[p["first"]] - small[j]) ** 2)
                vals.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        out[k] = float(np.mean(vals))
    return out


def score_one(item, out_dir):
    import torch
    from eval import pairing
    from eval.metrics import PairMetrics, SetMetrics, evaluate_video, read_video, summarize

    path, scene, traj, method, overlay = item
    name = os.path.basename(path)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    video = read_video(os.path.join(out_dir, "videos", name)).copy()
    if overlay:
        x0, y0, x1, y1 = OVERLAY
        video[:, y0:y1, x0:x1] = 0
    result = pairing.build_pairs(os.path.join(out_dir, "traj", traj), int(traj.rsplit("_", 1)[1]))
    if len(video) != result["num_video_frames"]:
        raise ValueError(f"{name}: {len(video)} frames, trajectory gives {result['num_video_frames']}")
    sm = {"all": SetMetrics(device)}
    rows = evaluate_video(video, result, PairMetrics(device), sm)
    summary = summarize(rows)
    summary["all"].update({k: v for k, v in sm["all"].compute().items() if k != "n"})
    shifts = shift_check(video, result["pairs"])
    return {"video": name, "path": path, "github": GITHUB + path, "scene": scene,
            "trajectory": traj, "method": method, "overlay_masked": overlay,
            "summary": summary, "shift_psnr": shifts,
            "best_shift": max(shifts, key=shifts.get)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--workers", type=int, default=1, help="Videos scored in parallel.")
    parser.add_argument("--only", help="Comma-separated scenes to score (default: all).")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    items = [v for v in VIDEOS if not args.only or v[1] in args.only.split(",")]

    for path, _, traj, _, _ in items:
        target = os.path.join(out, "videos", os.path.basename(path))
        if not os.path.exists(target):
            urllib.request.urlretrieve(RAW + path, target)
        make_trajectory(traj, os.path.join(out, "traj"))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_one, items, [out] * len(items)))

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "trajectory", "method", "lpips", "psnr", "ssim", "fid",
                    "lpips_long", "best_shift", "video"])
        for r in results:
            a, l = r["summary"]["all"], r["summary"]["long"]
            w.writerow([r["scene"], r["trajectory"], r["method"], a["lpips"], a["psnr"], a["ssim"],
                        a["fid"], l["lpips"], r["best_shift"], r["video"]])
    for r in results:
        a, l = r["summary"]["all"], r["summary"]["long"]
        print(f"{r['scene']:6s} {r['trajectory']:18s} {r['method']:8s} | LPIPS {a['lpips']:.3f} "
              f"PSNR {a['psnr']:6.2f} SSIM {a['ssim']:.3f} FID {a['fid']:6.1f} | long LPIPS "
              f"{l['lpips']:.3f} | best shift {r['best_shift']:+d}")
    print(f"Wrote {os.path.join(out, 'results.json')} and results.csv")


if __name__ == "__main__":
    main()
