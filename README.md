# Evaluation harness for the LTWM project (CS854)

Tools to evaluate world-model memory, a benchmark, and our replication of
WorldKV's Table 1 (LingBot-World-Fast rows). The code is built on
[WorldKV](https://github.com/cvlab-kaist/WorldKV) (Apache 2.0, see
`LICENSE.txt`); everything we added is in [`eval/`](eval/).

## What is here

| What | Where |
|---|---|
| **Metrics:** memory quality of a generated video, comparing each revisit frame with the first visit at the same camera pose (PSNR, SSIM, LPIPS, FID). How they work and how to score any video. | [`eval/`](eval/) |
| **Benchmark:** 16 entries (scene picture, description, camera trajectory) built from the WorldKV authors' demo videos. | [`eval/benchmark/`](eval/benchmark/) |
| **Replication:** how we generated and scored the benchmark with Sliding Window, Full KV and WorldKV, and the results. | [`eval/REPLICATION.md`](eval/REPLICATION.md) |

## Installation

Scoring videos (no GPU needed):

```bash
pip install -r eval/requirements.txt
```

Generating videos with WorldKV also needs its requirements, `flash_attn`, and
the model files:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
huggingface-cli download robbyant/lingbot-world-fast --local-dir ./lingbot-world-base-cam/lingbot_world_fast
```

## Changes to WorldKV's code

Outside `eval/`, the only code change is 12 added lines in
`wan/image2video_fast.py`:

- **Timing hooks:** one import of `eval/telemetry.py` and four calls around
  the chunk loop, recording time and memory per chunk. They do nothing unless
  the `WORLDKV_TELEMETRY` environment variable is set.
- **Freeing memory before decoding:** after the last chunk, the KV cache and
  stored chunks are released before the VAE decodes the video. Without this,
  decoding ran out of memory on 46 GB GPUs. The videos are pixel-identical
  with and without it.
