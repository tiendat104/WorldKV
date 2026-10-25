"""Memory-fidelity metrics: compare each revisit frame with its first visit.

Per pair (paired metrics, as in WorldKV Sec. 5.1):
  PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better).
Per set (distribution metrics):
  FID and KID between all revisit frames and all first-visit frames
  (lower is better). These need many frames, so the runner should
  accumulate them over all videos of a method, not one video.

Pairs come from eval.pairing and are put in groups:
  long:  gap beyond the attention window, so it tests long-term memory
  short: gap within the window, so the first visit is still in context
  sink:  the first visit is in the sink, which is never evicted
  all:   every pair (what the paper most likely reports)

Usage:
  python -m eval.metrics VIDEO.mp4 --pairs pairs.json -o result.json
"""

import argparse
import json

import imageio.v3 as iio
import lpips
import numpy as np
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# Largest plain attention window among the baselines (sliding window of 18
# latent frames). A pair farther apart than this tests long-term memory.
DEFAULT_WINDOW_LATENT = 18
GROUPS = ("all", "long", "short", "sink")


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_video(path):
    """Read a video into a (T, H, W, 3) uint8 array."""
    return iio.imread(path)


def to_tensor(frames, device):
    """(N, H, W, 3) uint8 array -> (N, 3, H, W) float tensor in [0, 1]."""
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0


def pair_group(pair, window_latent=DEFAULT_WINDOW_LATENT):
    if pair["in_sink"]:
        return "sink"
    return "long" if pair["gap_latent"] > window_latent else "short"


class PairMetrics:
    """PSNR, SSIM and LPIPS for a batch of (first visit, revisit) frames."""

    def __init__(self, device=None):
        self.device = device or default_device()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, first, revisit):
        """first, revisit: (N, 3, H, W) in [0, 1]. Returns per-pair numpy arrays."""
        psnr = peak_signal_noise_ratio(revisit, first, data_range=1.0,
                                       reduction="none", dim=(1, 2, 3))
        ssim = structural_similarity_index_measure(revisit, first, data_range=1.0,
                                                   reduction="none")
        # LPIPS expects inputs in [-1, 1].
        lp = self.lpips(revisit * 2 - 1, first * 2 - 1).flatten()
        return {
            "psnr": psnr.cpu().numpy(),
            "ssim": ssim.cpu().numpy(),
            "lpips": lp.cpu().numpy(),
        }


class SetMetrics:
    """FID and KID between the set of first visits and the set of revisits.

    First visits are treated as the "real" set and revisits as the
    "generated" set. Call update() on many videos, then compute() once.
    """

    def __init__(self, device=None, kid_subset_size=1000):
        device = device or default_device()
        # torchmetrics keeps FID statistics in float64, which MPS doesn't support.
        self.device = torch.device("cpu") if device.type == "mps" else device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.kid = KernelInceptionDistance(subset_size=kid_subset_size,
                                           normalize=True).to(self.device)
        self.count = 0

    @torch.no_grad()
    def update(self, first, revisit):
        first, revisit = first.to(self.device), revisit.to(self.device)
        for metric in (self.fid, self.kid):
            metric.update(first, real=True)
            metric.update(revisit, real=False)
        self.count += len(first)

    def compute(self):
        if self.count < 2:
            return {"n": self.count, "fid": None, "kid_mean": None, "kid_std": None}
        self.kid.subset_size = min(self.kid.subset_size, self.count)
        kid_mean, kid_std = self.kid.compute()
        return {
            "n": self.count,
            "fid": float(self.fid.compute()),
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
        }


def evaluate_video(video, pairs_result, pair_metrics, set_metrics=None,
                   window_latent=DEFAULT_WINDOW_LATENT, batch_size=16):
    """Score every pair of one video.

    video:        (T, H, W, 3) uint8 array
    pairs_result: output of eval.pairing.build_pairs
    set_metrics:  optional {group: SetMetrics}; their update() is called with
                  the pairs of that group (use "all" for every pair)
    Returns one row per pair: the pair fields plus group, psnr, ssim, lpips.
    """
    expected = pairs_result["num_video_frames"]
    if len(video) != expected:
        raise ValueError(
            f"Video has {len(video)} frames but the trajectory gives {expected}. "
            "Check that the pairs were built with the same --frame-num.")
    set_metrics = set_metrics or {}

    rows = [dict(p, group=pair_group(p, window_latent)) for p in pairs_result["pairs"]]
    device = pair_metrics.device
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        first = to_tensor(video[[r["first"] for r in batch]], device)
        revisit = to_tensor(video[[r["revisit"] for r in batch]], device)
        scores = pair_metrics(first, revisit)
        for k, row in enumerate(batch):
            for name, values in scores.items():
                row[name] = float(values[k])
        for group, sm in set_metrics.items():
            keep = [k for k, r in enumerate(batch) if group in ("all", r["group"])]
            if keep:
                sm.update(first[keep], revisit[keep])
    return rows


def summarize(rows):
    """Mean PSNR, SSIM and LPIPS per group."""
    summary = {}
    for group in GROUPS:
        sel = [r for r in rows if group in ("all", r["group"])]
        summary[group] = {"n": len(sel)}
        for name in ("psnr", "ssim", "lpips"):
            summary[group][name] = float(np.mean([r[name] for r in sel])) if sel else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", help="Generated video (.mp4).")
    parser.add_argument("--pairs", required=True, help="Pairs JSON from eval.pairing.")
    parser.add_argument("--window-latent", type=int, default=DEFAULT_WINDOW_LATENT)
    parser.add_argument("--no-fid", action="store_true", help="Skip FID/KID (faster).")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available).")
    parser.add_argument("-o", "--output", help="Write summary and per-pair rows to JSON.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()
    with open(args.pairs) as f:
        pairs_result = json.load(f)
    video = read_video(args.video)

    set_metrics = {} if args.no_fid else {g: SetMetrics(device) for g in ("all", "long")}
    rows = evaluate_video(video, pairs_result, PairMetrics(device), set_metrics,
                          window_latent=args.window_latent)
    summary = summarize(rows)
    for group, sm in set_metrics.items():
        summary[group].update({k: v for k, v in sm.compute().items() if k != "n"})

    print(f"{args.video}")
    print(f"{'group':<6} {'n':>5} {'LPIPS':>7} {'PSNR':>7} {'SSIM':>6} {'FID':>8}")
    for group in GROUPS:
        s = summary[group]
        if not s["n"]:
            continue
        fid = f"{s['fid']:8.2f}" if s.get("fid") is not None else f"{'-':>8}"
        print(f"{group:<6} {s['n']:>5} {s['lpips']:7.3f} {s['psnr']:7.3f} {s['ssim']:6.3f} {fid}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"video": args.video, "trajectory": pairs_result["trajectory"],
                       "window_latent": args.window_latent, "summary": summary,
                       "pairs": rows}, f, indent=1)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
