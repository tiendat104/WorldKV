# Benchmark: the WorldKV authors' scenes

16 entries, each a **scene picture + description + camera trajectory**, built
from the WorldKV authors' own demo videos so that our runs start from the same
inputs as theirs. The paper's 60-scene benchmark is not public; this is the
closest we can get to it.

Check it before any GPU run:

```bash
python -m eval.check_benchmark eval/benchmark/benchmark.json
```

## Files

| File / folder | What | In git |
|---|---|---|
| `benchmark.json` | The 16 entries (scene + trajectory); read by `eval.runner` | yes |
| `pilot.json` | 4 of the 16 entries for a cheap first run: the 4 whose authors' WorldKV scores are closest to the mean of all 16 (`23_L_R_W_S_L_R90_318`, `21_L_R_W_S_L_R90_318`, `04_RLWSRL90_245`, `31_WSRLRL90_245`) | yes |
| `final_images/` | The 16 scene pictures used by the benchmark | yes |
| `prompts.json` | One description per scene | yes |
| `traj/` | The 5 trajectories | yes |
| `images_cleaned/` | The 12 first frames with the key overlay removed (input to `final_images/`) | yes |
| `images/` | Raw first frames of the 16 videos | no (re-extract from the videos) |
| `videos/` | The authors' 16 source videos (389 MB) | no (download, see below) |

## Source videos

The 16 WorldKV videos from the authors' repo `cvlab-kaist/WorldKV`, branch
`main`, commit `315be40`: every one that is full length and has its trajectory
in its file name. Their paths (under `world_videos/`) are listed in
`eval/authors_videos.py`. Each picture in this benchmark has the same name as
its source video.

## How it was built

1. **Pictures: frame 0 of each video** (`images/`).
   - Exception: for scene 00 we use the authors' **original input picture**,
     `examples/new/00.png` from their code branch (1280 x 704, sharper than the
     video frame).
2. **Key overlay removed.** 12 of the 16 first frames show a W/A/S/D key
   overlay in the bottom-left corner (01, 04 x2, 20, 21, 23 `L_R_W_S_L_R90_318`,
   24, 26, 29, 31, 37, 41). It was painted out with Claude (image editing), which
   changed only the key area: the rest of each picture is pixel-identical to
   the frame (checked), and the size is unchanged. Results in `images_cleaned/`.
   (A ChatGPT edit was rejected because it regenerated the whole picture.)
3. **Cropped to 848 x 464** around the center, so every scene makes 464 x 848
   videos: the 832 x 464 frames lose 4 pixels at the top and bottom; the
   768 x 512 frames (15, uni01) lose 46. Scene 00's original picture is used
   as is (it already makes 464 x 848 videos). Results in `final_images/`.
4. **Descriptions** (`prompts.json`): scene 00 uses the authors' own
   (`examples/new/prompts.json`); the other 15 were written by us from each
   final picture, in the same style, saying what is on the left, center,
   right and in the distance.
5. **Trajectories** (`traj/`): recreated from the video names with the
   authors' `generate_traj.py` (2 degrees per frame when turning, 24 frames
   per step, 16-frame pauses for `_`). `R_L_R_L_R_L90_360` is copied from the
   authors' own `traj/` folder; generating it from its name gives an identical
   file. Every trajectory gives exactly the frame count of the authors' videos
   that use it.

## Entries

| Scene (= source video) | Trajectory | Picture | Video frames | Long-gap pairs |
|---|---|---|---|---|
| `00_R_L_R_L_R_L90_360_compressed` | `R_L_R_L_R_L90_360` | authors' original input | 357 | 206 |
| `01_R_L_W_S_R_L90_318` | `R_L_W_S_R_L90_318` | frame 0, overlay removed | 309 | 113 |
| `04_RLWSRL90_245` | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| `04_WSRLRL90_245` | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| `15_R_L_R_L_R_L90_360_compressed` | `R_L_R_L_R_L90_360` | frame 0 | 357 | 206 |
| `20_R_L_W_S_R_L90_318` | `R_L_W_S_R_L90_318` | frame 0, overlay removed | 309 | 113 |
| `21_L_R_W_S_L_R90_318` | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| `23_L_R_W_S_L_R90_318` | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| `23_RLWSRL90_245` | `RLWSRL90_245` | frame 0 | 237 | 81 |
| `24_WSRLRL90_245` | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| `26_L_R_W_S_L_R90_318` | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| `29_L_R_W_S_L_R90_318` | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| `31_WSRLRL90_245` | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| `37_RLWSRL90_245` | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| `41_RLWSRL90_245` | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| `uni01_L_R_W_S_L_R90_318` | `L_R_W_S_L_R90_318` | frame 0 | 309 | 113 |

In total 4,536 video frames per method (4.7 min of video).

## Notes

- **Scene 04** appears twice with two trajectories; its two pictures are the
  same (they differ only by video compression noise).
- **Scene 23** appears twice, but the authors started the two videos from
  differently framed pictures; `23_L_R_W_S_L_R90_318` is more zoomed in and
  noticeably blurrier. Both are kept as the authors used them.
- **Pictures from video frames are a bit soft** (compressed web video); only
  scene 00 is the authors' sharp original. Except for 00, the descriptions
  are ours, not the authors'.
- **Full KV does not fit on 46 GB GPUs (A40)** for the two 357-frame entries
  (00, 15); run those on more or larger GPUs.
- For every entry we also have the authors' own WorldKV video, so our scores
  can be compared with theirs scene by scene (see `eval/authors_videos.py`).
