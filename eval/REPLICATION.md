# Replicating WorldKV's Table 1 (LingBot-World-Fast)

This document describes how we replicated the LingBot-World-Fast rows of
WorldKV's Table 1 (Sliding Window, Full KV, WorldKV): how the videos were
generated from the benchmark, how they were scored, and the results.

- The benchmark (16 entries built from the authors' demo videos) is described
  in [`benchmark/README.md`](benchmark/README.md).
- The metrics (pairs of revisit and first-visit frames; PSNR, SSIM, LPIPS,
  FID) are described in [`README.md`](README.md).

## Summary

- **Done:** memory quality (LPIPS, PSNR, SSIM, FID) of all three methods on
  12 of the 16 benchmark entries, on 4x A40 GPUs.
- **Result:** the paper's order is reproduced. Sliding Window is worst on
  every metric; Full KV is slightly ahead of WorldKV on LPIPS, PSNR and SSIM,
  and WorldKV slightly ahead on FID, as in Table 1. The gaps between methods
  are smaller than in the paper.
- **Not done:** speed (FPS), which needs the paper's hardware (4x H200) or at
  least GPUs linked by NVLink; and 4 of the 16 entries (see
  [What is left](#what-is-left)).

## Setup

All videos were generated on 4x NVIDIA A40 GPUs (46 GB each) on RunPod.

## Process

### 1. Generating the videos

[`runner.py`](runner.py) takes a benchmark file (a list of scene + trajectory
entries) and, for every entry and every method, generates one video with
WorldKV's generation code:

```bash
python -m eval.runner --benchmark eval/benchmark/benchmark.json \
    --out /workspace/outputs/<name> --methods gpu_only,full_kv,sliding \
    --ckpt-dir /workspace/models/lingbot-world-base-cam --nproc 4 --no-p2p
```

To run only some entries, give it a benchmark file in the same format as
`benchmark/benchmark.json` with fewer entries. Add `--dry-run` to print the
generation commands without running them.

Methods and their WorldKV flags:

| Method | WorldKV flags |
|---|---|
| Sliding Window (`sliding`) | `--sliding_window 18` |
| Full KV (`full_kv`) | `--full_kv` |
| WorldKV (`gpu_only`) | `--use_retrieval --retrieval_frames 18 --kv_compression_enable --kv_compression_keep_ratio 0.25 --kv_compression_at_store --kv_compression_pooled --kv_bank_on_gpu` |
| WorldKV, stored chunks in CPU memory (`cpu_only`) | the same without `--kv_bank_on_gpu` |

WorldKV uses the paper's main settings: 6 retrieved chunks, each compressed to
1.5 frames. (The repo's own example uses `--kv_compression_keep_ratio 0.5`,
which is the "3 -> 2.0" row of the paper's Table 2, not Table 1.) `cpu_only`
makes the same video as `gpu_only` and differs only in speed and memory, so it
was not run for the quality results. All runs use seed 42 and 4 GPUs per video
(`--dit_fsdp --t5_fsdp --ulysses_size 4`).

How the runner works:

- **The model is loaded once.** Loading LingBot takes several minutes, so the
  runner writes the list of missing videos to `runs/jobs_<n>.json` and starts
  one [`worker.py`](worker.py) on the GPUs, which loads the model once and
  generates the videos one after another. Each job holds exactly the arguments
  `generate_fast.py` would get and makes the same `generate()` call (a test
  checks this). If the worker stops during a video (e.g. out of GPU memory),
  that video is marked `failed` and a new worker runs the rest.
- **Stopped runs can be restarted.** Videos that already exist are skipped, so
  the same command continues where it stopped.
- **Timing and memory are recorded per chunk.** The runner sets
  `WORLDKV_TELEMETRY` for every run, and the hooks in
  `wan/image2video_fast.py` (through [`telemetry.py`](telemetry.py)) record
  the time of every chunk (with CUDA events on the generation stream), FPS
  per chunk, peak GPU memory of every GPU, and CPU memory. Model loading and
  VAE decoding are not counted.

On 4x A40, a video took about 12-15 minutes for 309 frames and 9-11 minutes
for 237 frames, plus 6-8 minutes to load the model once per worker.

### 2. Scoring

After generating, the runner scores every video with the metric code (see
[`README.md`](README.md)):

- every (first visit, revisit) pair of the video's trajectory, found on the
  camera poses generation gave each frame, is scored with PSNR, SSIM and
  LPIPS; a video's score is the mean over its pairs, and a method's score the
  mean over its videos, so every entry counts equally;
- FID and KID are computed once per method over all pairs of all its videos;
- "long gaps" are the pairs whose first visit has left the attention window
  (more than 18 latent frames earlier): they test long-term memory.

### 3. Outputs

Everything goes in the `--out` folder (not in git):

| Path | What |
|---|---|
| `videos/<method>/<run>.mp4` | The generated videos (`<run>` = `<scene>_<trajectory>`) |
| `runs/<method>/<run>.json` | The run's status, time, settings and the equivalent stand-alone `generate_fast.py` command |
| `runs/<method>/<run>.telemetry.json` | Per-chunk time, FPS and memory |
| `runs/jobs_<n>.json`, `runs/worker_<n>.log` | The job list given to each worker, and its output |
| `metrics/<method>/<run>.json` | The score of every pair, and per-group means |
| `results.csv` | One row per run: LPIPS, PSNR, SSIM (all pairs and long-gap pairs), FPS, peak memory |
| `summary.csv` | One row per method, like Table 1 |

## Results

Because of our budget, we ran only 12 of the benchmark's 16 entries (see
[What is left](#what-is-left)). All values below are from our own generated
videos.

### All 12 entries

| Method | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| Sliding Window | 0.540 | 12.05 | 0.386 | 96.1 | 0.626 |
| Full KV | **0.482** | **13.30** | **0.404** | 65.0 | **0.545** |
| WorldKV | 0.505 | 12.85 | 0.391 | **64.6** | 0.566 |
| *Paper, Sliding Window (60 scenes)* | *0.581* | *12.18* | *0.375* | *144.0* | |
| *Paper, Full KV (60 scenes)* | *0.441* | *15.90* | *0.472* | *85.7* | |
| *Paper, WorldKV (60 scenes)* | *0.455* | *15.66* | *0.463* | *75.6* | |

LPIPS, PSNR and SSIM are means over the 12 entries; FID is computed once over
all pairs of the 12 entries.

### Per entry

**Sliding Window**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.476 | 12.47 | 0.342 | 197.0 | 0.576 |
| `21_L_R_W_S_L_R90_318` | 0.551 | 11.50 | 0.450 | 131.7 | 0.680 |
| `23_L_R_W_S_L_R90_318` | 0.514 | 11.47 | 0.306 | 144.5 | 0.598 |
| `04_RLWSRL90_245` | 0.573 | 11.46 | 0.351 | 118.6 | 0.673 |
| `04_WSRLRL90_245` | 0.610 | 11.17 | 0.315 | 147.1 | 0.646 |
| `31_WSRLRL90_245` | 0.563 | 12.11 | 0.364 | 206.7 | 0.578 |
| `37_RLWSRL90_245` | 0.621 | 10.24 | 0.392 | 134.5 | 0.740 |
| `23_RLWSRL90_245` | 0.512 | 11.14 | 0.358 | 149.8 | 0.618 |
| `uni01_L_R_W_S_L_R90_318` | 0.446 | 13.81 | 0.488 | 102.5 | 0.613 |
| `26_L_R_W_S_L_R90_318` | 0.358 | 14.59 | 0.494 | 102.1 | 0.459 |
| `41_RLWSRL90_245` | 0.669 | 9.29 | 0.275 | 217.5 | 0.720 |
| `24_WSRLRL90_245` | 0.584 | 15.29 | 0.497 | 183.1 | 0.610 |
| **Mean of 12** | **0.540** | **12.05** | **0.386** | **152.9** | **0.626** |

**Full KV**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.453 | 12.53 | 0.300 | 164.4 | 0.528 |
| `21_L_R_W_S_L_R90_318` | 0.447 | 13.77 | 0.495 | 57.3 | 0.559 |
| `23_L_R_W_S_L_R90_318` | 0.430 | 13.87 | 0.393 | 103.3 | 0.523 |
| `04_RLWSRL90_245` | 0.514 | 13.28 | 0.366 | 129.9 | 0.567 |
| `04_WSRLRL90_245` | 0.566 | 11.92 | 0.310 | 114.5 | 0.590 |
| `31_WSRLRL90_245` | 0.468 | 13.72 | 0.400 | 121.1 | 0.476 |
| `37_RLWSRL90_245` | 0.565 | 11.16 | 0.413 | 112.3 | 0.643 |
| `23_RLWSRL90_245` | 0.470 | 11.71 | 0.342 | 102.0 | 0.551 |
| `uni01_L_R_W_S_L_R90_318` | 0.357 | 15.15 | 0.506 | 39.5 | 0.455 |
| `26_L_R_W_S_L_R90_318` | 0.329 | 16.58 | 0.497 | 45.8 | 0.407 |
| `41_RLWSRL90_245` | 0.652 | 9.08 | 0.270 | 198.0 | 0.678 |
| `24_WSRLRL90_245` | 0.535 | 16.83 | 0.556 | 107.2 | 0.559 |
| **Mean of 12** | **0.482** | **13.30** | **0.404** | **107.9** | **0.545** |

**WorldKV**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.454 | 12.59 | 0.298 | 154.3 | 0.517 |
| `21_L_R_W_S_L_R90_318` | 0.476 | 13.50 | 0.486 | 61.5 | 0.588 |
| `23_L_R_W_S_L_R90_318` | 0.453 | 12.83 | 0.364 | 104.7 | 0.537 |
| `04_RLWSRL90_245` | 0.523 | 12.88 | 0.357 | 110.7 | 0.570 |
| `04_WSRLRL90_245` | 0.595 | 11.61 | 0.312 | 125.9 | 0.619 |
| `31_WSRLRL90_245` | 0.608 | 11.61 | 0.317 | 156.1 | 0.645 |
| `37_RLWSRL90_245` | 0.579 | 10.52 | 0.400 | 99.0 | 0.649 |
| `23_RLWSRL90_245` | 0.467 | 11.71 | 0.330 | 106.2 | 0.545 |
| `uni01_L_R_W_S_L_R90_318` | 0.367 | 15.16 | 0.500 | 38.9 | 0.468 |
| `26_L_R_W_S_L_R90_318` | 0.336 | 16.74 | 0.498 | 38.5 | 0.402 |
| `41_RLWSRL90_245` | 0.651 | 9.28 | 0.287 | 185.9 | 0.682 |
| `24_WSRLRL90_245` | 0.548 | 15.79 | 0.536 | 129.8 | 0.575 |
| **Mean of 12** | **0.505** | **12.85** | **0.391** | **109.3** | **0.566** |

### Memory

Peak GPU memory allocated, on the GPU with the largest peak, over all 12
entries: Sliding Window 23.6 GB, WorldKV 39.9 GB, Full KV 41.2 GB (of 46 GB).
Full KV fits the 309-frame videos on A40 but not the 357-frame ones.

### What matches the paper, and what does not

- **The order matches.** Sliding Window is worst on every metric. Full KV is
  slightly ahead of WorldKV on LPIPS, PSNR and SSIM, and WorldKV slightly
  ahead on FID (64.6 vs. 65.0), both as in Table 1. The difference to Sliding
  Window is largest on long-gap pairs, where memory matters.
- **It holds entry by entry.** On LPIPS, Full KV beats Sliding Window on all
  12 entries and WorldKV on 11 of 12 (the exception is `31_WSRLRL90_245`);
  WorldKV beats Full KV on 2 of 12. On per-entry FID, WorldKV beats Sliding
  Window on all 12 entries and Full KV on 11 of 12. SSIM is the least
  consistent metric: on 3 of 12 entries Sliding Window has the highest SSIM,
  even though its LPIPS is worse.
- **One entry is hard for every method.** On `41_RLWSRL90_245` all three have
  LPIPS about 0.65-0.67 and PSNR about 9.1-9.3; Sliding Window even has the
  highest PSNR there, by a hair.
- **The gaps are smaller than in the paper.** WorldKV is 0.035 LPIPS better
  than Sliding Window here, vs. 0.126 in the paper. Our Sliding Window is close
  to the paper's PSNR (12.05 vs. 12.18), but our Full KV and WorldKV are about
  2.6-2.8 dB below the paper's.
- **Why the numbers can differ:** the paper's 60 scenes are not public, so we
  use different scenes (built from the authors' demo videos) and our own
  scene descriptions; 12 entries are far fewer than 60, so the absolute
  numbers are less reliable than the order; and the paper does not publish
  its exact scoring protocol (e.g. which frames count as revisits, or how
  close two camera poses must be). FID also depends on the number of frames,
  so our FID is not directly comparable with the paper's.

## Machine notes

- **`--no-p2p`:** on RunPod's 4x A40 the GPUs are linked only by PCIe, and
  they hang when copying to each other directly. `--no-p2p` sets
  `NCCL_P2P_DISABLE=1`, so the data passes through the CPU. Do not use it on
  NVLink machines (e.g. H200).
- **GPU memory:** the runner sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for every run. Together
  with freeing the KV cache and stored chunks before the VAE decode (the
  12-line change in `wan/image2video_fast.py`), this is what makes the runs
  fit on 46 GB GPUs. Both were checked to leave the videos pixel-identical.
- **FPS on A40 is not comparable with the paper's:** with `--no-p2p`, the data
  between GPUs passes through the CPU, which takes most of each chunk's time
  and hides the differences between methods (Full KV runs at about 0.87x the
  speed of Sliding Window there, vs. 0.47x in the paper).
- **Minimum video length:** WorldKV needs videos of at least 27 latent frames
  (about 110 video frames); with shorter ones its attention window silently
  loses the recent frames, so the runner refuses them.

## What is left

- **4 entries:** `29_L_R_W_S_L_R90_318` and `01_R_L_W_S_R_L90_318` fit on A40;
  the two 357-frame entries, `00_R_L_R_L_R_L90_360_compressed` and
  `15_R_L_R_L_R_L90_360_compressed`, need more or larger GPUs for Full KV.
- **Speed (FPS):** not measured. The paper reports, on 4x H200, 5.05 FPS
  (Sliding Window), 2.36 (Full KV) and 4.78 (WorldKV), measured at the last
  chunk of each video; it reports no CPU-only number. Measuring it needs GPUs
  linked by NVLink.
