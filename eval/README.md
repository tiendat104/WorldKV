# Evaluation harness

Evaluation harness for the LTWM project (CS854). It measures:

- **Memory quality:** LPIPS, PSNR, SSIM, FID/KID between each revisit frame and the first-visit frame at the same camera pose.
- **Speed:** FPS per chunk (the paper reports the last chunk).
- **Memory:** peak GPU memory per GPU, and CPU memory.

## How it works

```
benchmark.json ─► runner.py ─► worker.py (on the GPUs) ─► videos/<method>/<run>.mp4
 (picture,                       loads the model once,          │
  description,                   generates each video           ▼
  trajectory)                    like generate_fast.py    pairing.py: which frames show
                                                          the same camera pose
                                                                │
                                                                ▼
                                                          metrics.py: score every pair
                                                                │
                                                                ▼
                                                    results.csv (per run), summary.csv (per method)
```

1. **Generate.** For every (scene, trajectory) entry of a benchmark and every
   method, the runner generates one video, unless it already exists.
2. **Pair.** From the trajectory's camera poses alone (no video needed), find
   every frame where the camera returns to an earlier position and viewing
   direction (see [Pairs](#pairs)).
3. **Score.** Compare each revisit frame with its first-visit frame:
   PSNR, SSIM and LPIPS per pair; FID and KID over all pairs of a method.
4. **Summarize.** Write one row per run and one row per method (see
   [Outputs](#outputs)).

### Pairs

- Two frames pair up when the camera's position (within 0.001) and viewing
  direction (within 0.5°) are the same. Each revisit is paired with the
  **earliest** frame at that pose.
- Frame numbering and trimming follow WorldKV's code, so pair indices match
  the generated video frame by frame.
- Every pair is put in a group, by its gap in latent frames (1 latent frame
  = 4 video frames):

  | Group | Meaning |
  |---|---|
  | `long` | gap larger than 18 latent frames (the sliding window): the first visit has left the attention window, so this tests long-term memory |
  | `short` | gap within the window: the first visit is still attended |
  | `sink` | the first visit is in the first 3 latent frames (the "sink"), which the model never evicts |
  | `all` | every pair; this is what the paper most likely reports |

### Metrics

- **PSNR, SSIM** (torchmetrics) and **LPIPS** (the official `lpips` package,
  AlexNet) are computed per pair on whole frames, in [0, 1].
- **FID and KID** (torchmetrics) need many frames, so they are computed once
  per method, over all pairs of all its videos: first visits are the "real"
  set and revisits the "generated" set. On a Mac they run on the CPU (MPS has
  no float64).

## Setup

Python 3.10 (the repo requires >= 3.10, and `numpy < 2`).

```bash
conda activate cs854
pip install -r eval/requirements.txt
```

On a laptop this is enough to run the pairing and metric code. Generating
videos needs the full WorldKV requirements and GPUs.

The first metric run downloads the LPIPS (AlexNet) and FID (Inception)
networks, about 320 MB, into `~/.cache/torch`.

## Layout

Code lives in this folder so that changes to WorldKV's own files stay small
(see [Changes to WorldKV's code](#changes-to-worldkvs-code)).

| File | What it does |
|---|---|
| `pairing.py` | Finds (first visit, revisit) frame pairs from a trajectory's poses |
| `metrics.py` | PSNR, SSIM, LPIPS per pair; FID/KID over sets of frames |
| `runner.py` | Generates one video per (scene, trajectory) pair and method, then scores them all |
| `worker.py` | Loads the model once and generates a list of videos (started by `runner.py`) |
| `telemetry.py` | Per-chunk timing (FPS) and GPU/CPU memory, called from WorldKV's generation loop |
| `prepare_images.py` | Crops raw benchmark pictures to 848 x 464 and names them with scene IDs |
| `check_benchmark.py` | Checks a benchmark (pictures, descriptions, video lengths, long-gap revisits) before GPU runs |
| `benchmark/` | The benchmark: 16 entries built from the WorldKV authors' demo videos (see its README) |
| `authors_videos.py` | Scores the WorldKV authors' released demo videos with our metric code; their scores should show Table 1's order and range |
| `setup_pod.sh` | Prepares a fresh RunPod pod in one command: code, environment, `flash_attn`, model files, checks |
| `nccl_check.py` | Checks that the GPUs can talk to each other (tells you whether `--no-p2p` is needed) |
| `requirements.txt` | Packages the harness needs on top of WorldKV's |

## Running a benchmark

```bash
# Laptop: fake model, checks the pipeline end to end
python -m eval.runner --benchmark eval/tests/fixtures/dev/benchmark.json \
    --out eval/outputs/dev_fake --fake --no-fid

# GPU machine: check the benchmark, print the commands, then run
python -m eval.check_benchmark eval/benchmark/benchmark.json
python -m eval.runner --benchmark eval/benchmark/benchmark.json \
    --out eval/outputs/full --ckpt-dir ./lingbot-world-base-cam --nproc 4 --dry-run
python -m eval.runner --benchmark eval/benchmark/benchmark.json \
    --out eval/outputs/full --ckpt-dir ./lingbot-world-base-cam --nproc 4
```

Options:

| Option | Meaning |
|---|---|
| `--methods` | Comma-separated subset of the methods below (default: all four) |
| `--nproc` | GPUs per run (default 4) |
| `--seed` | Generation seed (default 42) |
| `--no-p2p` | Turn off direct GPU-to-GPU copies (needed where GPUs are linked only by PCIe, e.g. 4x A40; not on NVLink machines) |
| `--fake` | Fake model, for checking the pipeline without GPUs |
| `--dry-run` | Print the generation commands without running them |
| `--no-fid` | Skip FID/KID (faster) |
| `--device` | `cuda`, `mps` or `cpu` for scoring |

Methods:

| Method | WorldKV flags |
|---|---|
| `sliding` | `--sliding_window 18` |
| `full_kv` | `--full_kv` |
| `gpu_only` | WorldKV, stored chunks kept on the GPU (`--kv_bank_on_gpu`) |
| `cpu_only` | WorldKV, stored chunks kept in CPU memory |

The two WorldKV methods use the paper's main settings: 6 retrieved chunks,
each compressed to 1.5 frames (`--use_retrieval --retrieval_frames 18
--kv_compression_enable --kv_compression_keep_ratio 0.25
--kv_compression_at_store --kv_compression_pooled`). Note that the repo's own
example uses `--kv_compression_keep_ratio 0.5` (2.0 frames), which is the
"3 -> 2.0" row of the paper's Table 2, not Table 1. `gpu_only` and `cpu_only`
make the same video; they differ only in where the stored chunks live, so
only their speed and memory differ.

## Outputs

Everything goes in the `--out` folder:

| Path | What |
|---|---|
| `videos/<method>/<run>.mp4` | The generated videos (`<run>` = `<scene>_<trajectory>`) |
| `runs/<method>/<run>.json` | The run's status, timing, and the equivalent stand-alone `generate_fast.py` command |
| `runs/<method>/<run>.telemetry.json` | Per-chunk time, FPS and memory |
| `runs/jobs_<n>.json`, `runs/worker_<n>.log` | The job list given to each worker, and its output |
| `metrics/<method>/<run>.json` | The score of every pair, and per-group means |
| `results.csv` | One row per run: per-group means of LPIPS, PSNR and SSIM (`all`, and `long` as `*_long`), FPS, peak memory |
| `summary.csv` | One row per method, like Table 1 |

How `summary.csv` combines the runs of a method:

- LPIPS, PSNR, SSIM and FPS: mean over videos, so every scene counts equally;
- peak GPU and CPU memory: the largest value over videos (what the method needs);
- FID and KID: computed once over all pairs of all videos (`fid`, and `fid_long` for long-gap pairs only).

Videos that already exist are skipped, so a stopped run can be restarted with
the same command.

## Loading the model once

Loading LingBot takes minutes, so the runner does not start `generate_fast.py`
once per video. It writes the list of missing videos to
`<out>/runs/jobs_<n>.json` and starts one `eval/worker.py`, which loads the
model once and generates them one after another. Each job holds the same
arguments `generate_fast.py` would get, parsed by `generate_fast.py`'s own
parser and passed to the same `generate()` call (a test checks that the two
calls match). If the worker stops during a video (e.g. out of GPU memory),
that video is marked `failed` and a new worker runs the rest.

## Changes to WorldKV's code

Outside this folder, the harness changes only `wan/image2video_fast.py`
(12 added lines) and `.gitignore`:

- **Timing hooks (5 lines):** one import of `eval/telemetry.py` and four calls
  around the chunk loop. They do nothing unless `WORLDKV_TELEMETRY` is set.
- **Freeing memory before decoding (4 lines and a comment):** after the last
  chunk, the KV cache and stored chunks are released before the VAE decodes
  the video, which needs only the generated latents. Without this, decoding
  ran out of memory on 46 GB GPUs. The videos are pixel-identical with and
  without it.
