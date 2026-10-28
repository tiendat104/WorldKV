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
  the time of every chunk (after waiting for the GPUs), FPS per chunk, peak
  GPU memory of every GPU, and CPU memory. Model loading and VAE decoding are
  not counted.

On 4x A40, a video took about 12-15 minutes for 309 frames and 9-11 minutes
for 237 frames, plus 6-8 minutes to load the model once per worker.

### 2. Scoring

After generating, the runner scores every video with the metric code (see
[`README.md`](README.md)):

- every (first visit, revisit) pair of the video's trajectory is scored with
  PSNR, SSIM and LPIPS; a video's score is the mean over its pairs, and a
  method's score the mean over its videos, so every entry counts equally;
- FID and KID are computed once per method over all pairs of all its videos;
- "long gaps" are the pairs whose first visit has left the attention window
  (more than 18 latent frames earlier): they test long-term memory;
- our videos are scored on whole frames, without any mask.

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
videos, scored without masks.

### All 12 entries

| Method | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| Sliding Window | 0.538 | 12.17 | 0.393 | 92.3 | 0.627 |
| Full KV | **0.487** | **13.31** | **0.410** | 62.9 | **0.549** |
| WorldKV | 0.501 | 13.08 | 0.401 | **61.4** | 0.566 |
| *Paper, Sliding Window (60 scenes)* | *0.581* | *12.18* | *0.375* | *144.0* | |
| *Paper, Full KV (60 scenes)* | *0.441* | *15.90* | *0.472* | *85.7* | |
| *Paper, WorldKV (60 scenes)* | *0.455* | *15.66* | *0.463* | *75.6* | |

LPIPS, PSNR and SSIM are means over the 12 entries; FID is computed once over
all pairs of the 12 entries.

### Per entry

FID per entry uses only one video's pairs (167 or 239), which is few for FID:
these values are higher and noisier than the FID over all 12 entries, so
compare methods within an entry rather than reading the absolute numbers.

**Sliding Window**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.473 | 12.56 | 0.348 | 199.3 | 0.577 |
| `21_L_R_W_S_L_R90_318` | 0.546 | 11.71 | 0.457 | 124.6 | 0.676 |
| `23_L_R_W_S_L_R90_318` | 0.498 | 11.87 | 0.319 | 146.8 | 0.593 |
| `04_RLWSRL90_245` | 0.598 | 11.01 | 0.339 | 119.7 | 0.685 |
| `04_WSRLRL90_245` | 0.579 | 11.87 | 0.338 | 133.3 | 0.635 |
| `31_WSRLRL90_245` | 0.553 | 12.25 | 0.374 | 185.5 | 0.609 |
| `37_RLWSRL90_245` | 0.629 | 9.99 | 0.388 | 134.1 | 0.740 |
| `23_RLWSRL90_245` | 0.537 | 10.60 | 0.343 | 150.6 | 0.621 |
| `uni01_L_R_W_S_L_R90_318` | 0.438 | 14.70 | 0.510 | 98.3 | 0.607 |
| `26_L_R_W_S_L_R90_318` | 0.360 | 14.74 | 0.505 | 100.0 | 0.463 |
| `41_RLWSRL90_245` | 0.704 | 8.63 | 0.254 | 233.7 | 0.725 |
| `24_WSRLRL90_245` | 0.539 | 16.12 | 0.537 | 155.7 | 0.596 |
| **Mean of 12** | **0.538** | **12.17** | **0.393** | **148.5** | **0.627** |

**Full KV**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.444 | 12.58 | 0.314 | 165.5 | 0.523 |
| `21_L_R_W_S_L_R90_318` | 0.427 | 14.18 | 0.521 | 55.7 | 0.538 |
| `23_L_R_W_S_L_R90_318` | 0.418 | 14.52 | 0.416 | 100.4 | 0.523 |
| `04_RLWSRL90_245` | 0.544 | 12.63 | 0.348 | 125.8 | 0.594 |
| `04_WSRLRL90_245` | 0.535 | 12.60 | 0.335 | 105.1 | 0.569 |
| `31_WSRLRL90_245` | 0.520 | 12.28 | 0.373 | 117.9 | 0.544 |
| `37_RLWSRL90_245` | 0.577 | 11.05 | 0.404 | 113.0 | 0.651 |
| `23_RLWSRL90_245` | 0.500 | 11.11 | 0.323 | 102.1 | 0.547 |
| `uni01_L_R_W_S_L_R90_318` | 0.361 | 16.14 | 0.529 | 35.5 | 0.451 |
| `26_L_R_W_S_L_R90_318` | 0.330 | 16.77 | 0.515 | 42.6 | 0.409 |
| `41_RLWSRL90_245` | 0.686 | 8.43 | 0.248 | 211.7 | 0.689 |
| `24_WSRLRL90_245` | 0.497 | 17.41 | 0.591 | 95.2 | 0.546 |
| **Mean of 12** | **0.487** | **13.31** | **0.410** | **105.9** | **0.549** |

**WorldKV**

| Entry | LPIPS ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ (long gaps) |
|---|---|---|---|---|---|
| `20_R_L_W_S_R_L90_318` | 0.448 | 12.67 | 0.313 | 152.5 | 0.522 |
| `21_L_R_W_S_L_R90_318` | 0.454 | 14.16 | 0.518 | 57.8 | 0.575 |
| `23_L_R_W_S_L_R90_318` | 0.446 | 13.39 | 0.381 | 97.1 | 0.541 |
| `04_RLWSRL90_245` | 0.540 | 12.64 | 0.348 | 110.7 | 0.583 |
| `04_WSRLRL90_245` | 0.574 | 12.02 | 0.329 | 115.4 | 0.614 |
| `31_WSRLRL90_245` | 0.575 | 12.13 | 0.336 | 143.0 | 0.643 |
| `37_RLWSRL90_245` | 0.587 | 10.37 | 0.395 | 100.1 | 0.659 |
| `23_RLWSRL90_245` | 0.499 | 11.11 | 0.314 | 105.8 | 0.539 |
| `uni01_L_R_W_S_L_R90_318` | 0.368 | 16.21 | 0.526 | 35.2 | 0.463 |
| `26_L_R_W_S_L_R90_318` | 0.337 | 16.89 | 0.512 | 36.6 | 0.405 |
| `41_RLWSRL90_245` | 0.681 | 8.59 | 0.269 | 198.7 | 0.683 |
| `24_WSRLRL90_245` | 0.507 | 16.71 | 0.575 | 107.6 | 0.565 |
| **Mean of 12** | **0.501** | **13.08** | **0.401** | **105.0** | **0.566** |

### Memory

Peak GPU memory allocated, on the GPU with the largest peak, over all 12
entries: Sliding Window 23.6 GB, WorldKV 39.9 GB, Full KV 41.2 GB (of 46 GB).
Full KV fits the 309-frame videos on A40 but not the 357-frame ones.

### What matches the paper, and what does not

- **The order matches.** Sliding Window is worst on every metric. Full KV is
  slightly ahead of WorldKV on LPIPS, PSNR and SSIM, and WorldKV slightly
  ahead on FID (61.4 vs. 62.9), both as in Table 1. The difference to Sliding
  Window is largest on long-gap pairs, where memory matters.
- **It holds entry by entry.** On LPIPS, Full KV beats Sliding Window on all
  12 entries and WorldKV on 11 of 12 (the exception is `31_WSRLRL90_245`);
  WorldKV beats Full KV on 3 of 12. On per-entry FID, WorldKV beats Sliding
  Window on all 12 entries and Full KV on 11 of 12. SSIM is the least
  consistent metric: on 4 of 12 entries Sliding Window has the highest SSIM,
  even though its LPIPS is worse.
- **One entry is hard for every method.** On `41_RLWSRL90_245` all three have
  LPIPS about 0.69 and PSNR about 8.5; Sliding Window even has the highest
  PSNR there.
- **The gaps are smaller than in the paper.** WorldKV is 0.037 LPIPS better
  than Sliding Window here, vs. 0.126 in the paper. Our Sliding Window matches
  the paper's PSNR (12.17 vs. 12.18), but our Full KV and WorldKV are about
  2.6 dB below the paper's.
- **Why the numbers can differ:** the paper's 60 scenes are not public, so we
  use different scenes (built from the authors' demo videos) and our own
  scene descriptions; 12 entries are far fewer than 60, so the absolute
  numbers are less reliable than the order; and the paper does not publish
  its exact scoring protocol (e.g. which frames count as revisits). FID also
  depends on the number of frames, so our FID is not directly comparable with
  the paper's.

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
