"""Check a benchmark before spending GPU time on it.

For every (scene, trajectory) pair it checks that:
  - the picture exists and makes 464 x 848 videos, like every other scene
    (the model keeps the picture's shape, so the shape decides the video size);
  - the scene has a description in prompts.json;
  - the trajectory exists and its video is long enough for WorldKV;
  - the trajectory has enough long-gap revisits to test long-term memory.
It then prints a summary and exits with an error if anything failed.

Usage:
  python -m eval.check_benchmark eval/benchmark/full/benchmark.json
"""

import argparse
import collections
import math
import sys

from PIL import Image

from eval import pairing, runner
from eval.metrics import pair_group
from eval.prepare_images import HEIGHT, WIDTH

MIN_LONG_PAIRS = 50
MAX_AREA = 480 * 832  # --size 480*832


def video_size(width, height, max_area=MAX_AREA):
    """(height, width) of the video the model makes from a picture of this size.

    Same calculation as wan/image2video_fast.py: keep the picture's shape at
    about `max_area` pixels, rounded to the latent grid (VAE stride 8, patch 2).
    """
    aspect = height / width
    lat_h = round(math.sqrt(max_area * aspect) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / aspect) // 8 // 2 * 2)
    return lat_h * 8, lat_w * 8


def check(benchmark_path, min_long_pairs=MIN_LONG_PAIRS):
    """Returns (problems, rows): a list of problem messages and one row per pair."""
    problems, rows = [], []
    try:
        entries = runner.load_benchmark(benchmark_path)
    except (FileNotFoundError, KeyError) as e:
        return [f"cannot load benchmark: {e!r}"], rows

    for entry in entries:
        name = runner.run_name(entry)
        size = Image.open(entry["image"]).size
        video = video_size(*size)
        if video != (HEIGHT, WIDTH):
            problems.append(f"{name}: picture {size[0]}x{size[1]} makes {video[0]}x{video[1]} "
                            f"videos, expected {HEIGHT}x{WIDTH}")
        if not entry["prompt"].strip():
            problems.append(f"{name}: empty description")
        result = pairing.build_pairs(entry["traj_dir"], entry["frame_num"])
        groups = collections.Counter(pair_group(p) for p in result["pairs"])
        latent = pairing.latent_index(result["num_video_frames"] - 1) + 1
        if latent < runner.WORLDKV_WINDOW_LATENT:
            problems.append(f"{name}: {latent} latent frames, below WorldKV's "
                            f"{runner.WORLDKV_WINDOW_LATENT}")
        if groups["long"] < min_long_pairs:
            problems.append(f"{name}: only {groups['long']} long-gap pairs "
                            f"(need {min_long_pairs})")
        rows.append({"run": name, "frames": result["num_video_frames"], "latent": latent,
                     "long": groups["long"], "short": groups["short"], "sink": groups["sink"]})
    return problems, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark", help="Benchmark JSON file.")
    parser.add_argument("--min-long-pairs", type=int, default=MIN_LONG_PAIRS)
    args = parser.parse_args()

    problems, rows = check(args.benchmark, args.min_long_pairs)
    for r in rows:
        print(f"{r['run']:36s} {r['frames']:4d} frames ({r['frames'] / 16:4.1f} s) | "
              f"long {r['long']:3d}  short {r['short']:3d}  sink {r['sink']:3d}")
    if rows:
        total = sum(r["frames"] for r in rows)
        print(f"\n{len(rows)} pairs, {total} video frames per method "
              f"({total / 16 / 60:.1f} min of video)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
