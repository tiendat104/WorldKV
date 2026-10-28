# Memory-quality metrics

Tools to measure how well a generated video remembers places it has already
shown. Whenever the camera returns to an earlier position and viewing
direction, the frame at the return (the **revisit**) is compared with the
frame from the first time (the **first visit**). A model with good memory
shows the same place both times.

Metrics, as in the WorldKV paper (Sec. 5.1):

| Metric | Computed | Better |
|---|---|---|
| PSNR | per pair | higher |
| SSIM | per pair | higher |
| LPIPS | per pair | lower |
| FID, KID | over a set of pairs (first visits vs. revisits) | lower |

They work on any video, from any generation code, as long as you have the
camera trajectory it was generated from.

## Setup

Python 3.10 or newer, with `numpy < 2`.

```bash
pip install -r eval/requirements.txt
```

The first run downloads the LPIPS (AlexNet) and FID (Inception) networks,
about 320 MB, into `~/.cache/torch`. A GPU is not needed: scoring runs on
CUDA, Apple MPS or the CPU.

## Files

| File | What it does |
|---|---|
| `pairing.py` | Finds the (first visit, revisit) frame pairs from a trajectory's camera poses |
| `metrics.py` | Scores a video on those pairs: PSNR, SSIM, LPIPS per pair; FID/KID over sets of pairs |
| `requirements.txt` | Packages these tools need |

## How it works

### 1. Pairs

`pairing.py` reads only the trajectory (its `*poses.npy` file: one 4x4
camera-to-world matrix per video frame); it does not need the video.

- Two frames pair up when the camera position (within 0.001) and viewing
  direction (within 0.5°) are the same. Each revisit is paired with the
  **earliest** frame at that pose.
- The trajectory is trimmed exactly as WorldKV's generation code trims it,
  so pair indices match the generated video frame by frame: video frame `j`
  uses camera pose `j`, and the video has `4 * (lat_f - 1) + 1` frames, where
  `lat_f` (latent frames) is a multiple of 3.
- Every pair is put in a group by its gap in latent frames (1 latent frame =
  4 video frames):

  | Group | Meaning |
  |---|---|
  | `long` | gap larger than 18 latent frames (the sliding window): the first visit has left the attention window, so this tests long-term memory |
  | `short` | gap within the window: the first visit is still attended |
  | `sink` | the first visit is in the first 3 latent frames (the "sink"), which the model never evicts |
  | `all` | every pair; this is what the paper most likely reports |

### 2. Scores

- **PSNR, SSIM** (torchmetrics) and **LPIPS** (the official `lpips` package,
  AlexNet) compare the two frames of each pair, on whole frames with values
  in [0, 1]. A group's score is the mean over its pairs.
- **FID and KID** (torchmetrics) compare two sets of frames: the first visits
  (the "real" set) and the revisits (the "generated" set). They need many
  frames to be reliable, so compute them over all videos of a method together,
  not per video (see [Scoring several videos](#scoring-several-videos)).
  On a Mac they run on the CPU, because MPS has no float64.

## Scoring one video

```bash
# 1. Pairs, from the trajectory folder and the --frame_num used for generation
python -m eval.pairing path/to/traj/R_L_W_S_R_L90_318 --frame-num 318 -o pairs.json

# 2. Scores of the video on those pairs
python -m eval.metrics video.mp4 --pairs pairs.json -o scores.json
```

The second command prints one line per group:

```
group      n   LPIPS    PSNR   SSIM      FID
all      239   0.449  12.666  0.311   152.48
long     113   0.523  11.589  0.270   197.74
short     99   0.328  14.392  0.386        -
sink      27   0.586  10.850  0.207        -
```

Options:

| Command | Option | Meaning |
|---|---|---|
| `eval.pairing` | `--frame-num` | The `--frame_num` given to the generator (required) |
| | `--rot-tol-deg`, `--trans-tol` | How close two poses must be to count as the same (defaults 0.5 and 0.001) |
| | `-o` | Where to write the pairs |
| `eval.metrics` | `--pairs` | Pairs file from `eval.pairing` (required) |
| | `--window-latent` | Gap, in latent frames, above which a pair counts as `long` (default 18) |
| | `--no-fid` | Skip FID/KID (faster) |
| | `--device` | `cuda`, `mps` or `cpu` (default: the best available) |
| | `-o` | Where to write the scores |

The video must have exactly the number of frames `eval.pairing` reports;
otherwise `eval.metrics` stops with an error, because the pairs would point
to the wrong frames.

### Output files

`pairs.json`:

```
{"trajectory": ..., "num_video_frames": ..., "sink_latent_frames": 3,
 "pairs": [{"first": 136, "revisit": 170, "gap_frames": 34, "gap_latent": 9, "in_sink": false}, ...]}
```

`scores.json`:

```
{"video": ..., "trajectory": ..., "window_latent": 18,
 "summary": {"all":  {"n": ..., "psnr": ..., "ssim": ..., "lpips": ..., "fid": ..., "kid_mean": ..., "kid_std": ...},
             "long": {...}, "short": {...}, "sink": {...}},
 "pairs": [{"first": ..., "revisit": ..., "gap_latent": ..., "group": "long",
            "psnr": ..., "ssim": ..., "lpips": ...}, ...]}
```

`fid` and `kid_*` are given for `all` and `long` only.

## Scoring several videos

To compare methods, score all videos of a method with the same metric objects,
so FID and KID are computed once over all their pairs:

```python
from eval.pairing import build_pairs
from eval.metrics import PairMetrics, SetMetrics, default_device, evaluate_video, read_video, summarize

videos = [  # (video, trajectory folder, --frame_num)
    ("out/ltwm/scene_a.mp4", "traj/R_L_W_S_R_L90_318", 318),
    ("out/ltwm/scene_b.mp4", "traj/RLWSRL90_245", 245),
]

device = default_device()
pair_metrics = PairMetrics(device)
set_metrics = {"all": SetMetrics(device), "long": SetMetrics(device)}

for video, traj_dir, frame_num in videos:
    rows = evaluate_video(read_video(video), build_pairs(traj_dir, frame_num),
                          pair_metrics, set_metrics)
    s = summarize(rows)  # per-group means of this video
    print(video, s["all"]["lpips"], s["all"]["psnr"], s["all"]["ssim"], s["long"]["lpips"])

for group, sm in set_metrics.items():
    print(group, "FID", sm.compute()["fid"])
```

For a method's PSNR, SSIM and LPIPS, take the mean of the per-video values,
so that every video counts equally.
