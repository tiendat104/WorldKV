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
