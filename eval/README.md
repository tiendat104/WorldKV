# Evaluation harness

Evaluation harness for the LTWM project (CS854). It measures:

- **Memory quality:** LPIPS, PSNR, SSIM, FID/KID between each revisit frame and the first-visit frame at the same camera pose.
- **Speed:** FPS per chunk (the paper reports the last chunk).
- **Memory:** peak GPU memory per GPU.

The first goal is to reproduce the LingBot-World-Fast rows of WorldKV's Table 1
(Sliding Window, Full KV, WorldKV GPU-only, WorldKV CPU-only).

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
(only the timing and memory logging).

| File | What it does |
|---|---|
| `pairing.py` | Finds (first visit, revisit) frame pairs from a trajectory's poses |
| `metrics.py` | PSNR, SSIM, LPIPS per pair; FID/KID over sets of frames |
| `runner.py` | Generates one video per (scene, trajectory) pair and method, then scores them all |
| `worker.py` | Loads the model once and generates a list of videos (started by `runner.py`) |
| `telemetry.py` | Per-chunk timing (FPS) and GPU/CPU memory, called from WorldKV's generation loop |
| `benchmark/dev/` | Dev test set: 5 scenes, 3 trajectories, `benchmark.json` |
| `tests/` | Tests (`python -m pytest eval/tests`) |

## Running a benchmark

```bash
# Laptop: fake model, checks the pipeline end to end
python -m eval.runner --benchmark eval/benchmark/dev/benchmark.json \
    --out eval/outputs/dev_fake --fake --no-fid

# GPU machine: print the commands first, then run
python -m eval.runner --benchmark eval/benchmark/dev/benchmark.json \
    --out eval/outputs/dev --ckpt-dir ./lingbot-world-base-cam --nproc 4 --dry-run
python -m eval.runner --benchmark eval/benchmark/dev/benchmark.json \
    --out eval/outputs/dev --ckpt-dir ./lingbot-world-base-cam --nproc 4
```

Methods (`--methods`, default all): `sliding` (`--sliding_window 18`),
`full_kv`, `gpu_only` and `cpu_only`. The two WorldKV methods use the paper's
main settings: 6 retrieved chunks, each compressed to 1.5 frames
(`--retrieval_frames 18 --kv_compression_keep_ratio 0.25`). Note that the
repo's own example uses `--kv_compression_keep_ratio 0.5` (2.0 frames), which
is the "3 -> 2.0" row of the paper's Table 2, not Table 1.

## Timing and memory

`wan/image2video_fast.py` has five added lines that call `eval/telemetry.py`
(one import and four calls around the chunk loop). When the runner launches a
run, it sets `WORLDKV_TELEMETRY` to `<out>/runs/<method>/<run>.telemetry.json`,
and the run writes there:

- the time of every chunk (after waiting for the GPU to finish), and FPS per
  chunk = video frames / time. A chunk is 3 latent frames = 12 video frames
  (9 for the first chunk). Model loading and VAE decoding are not counted.
- peak GPU memory of every GPU (allocated and reserved), and memory per chunk;
- CPU memory (RSS) per chunk.

Without the environment variable, nothing is recorded.

In the tables, `fps_last_chunk` matches the paper's FPS, and
`peak_gpu_mem_gb` is the largest peak among the GPUs.

## Loading the model once

Loading LingBot takes minutes, so the runner does not start `generate_fast.py`
once per video. It writes the list of missing videos to
`<out>/runs/jobs_<n>.json` and starts one `eval/worker.py`, which loads the
model once and generates them one after another. Each job holds the same
arguments `generate_fast.py` would get, parsed by `generate_fast.py`'s own
parser and passed to the same `generate()` call (a test checks that the two
calls match). The equivalent stand-alone command is saved in each run's
`<out>/runs/<method>/<run>.json`. If the worker stops during a video (e.g.
out of GPU memory), that video is marked `failed` and a new worker runs the
rest. Worker output goes to `<out>/runs/worker_<n>.log`.

## Machine notes

- On machines where GPUs are linked only by PCIe (e.g. RunPod 4x A40), the
  GPUs can hang when copying to each other directly. Pass `--no-p2p`
  (sets `NCCL_P2P_DISABLE=1`). Do not use it on NVLink machines (H200).
- WorldKV needs videos of at least 27 latent frames (about 110 video frames);
  the runner refuses shorter ones.

## Restarting

Videos that already exist are skipped, so a stopped run can be restarted.
Results go to `<out>/results.csv` (one row per run) and `<out>/summary.csv`
(one row per method). Every run's exact command is saved in `<out>/runs/`.
