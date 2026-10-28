# Benchmark: the WorldKV authors' scenes

16 entries, each a **scene picture + description + camera trajectory**, built
from the WorldKV authors' own demo videos so that our runs start from the same
inputs as theirs. The paper's 60-scene benchmark is not public; this is the
closest we can get to it.

## Files

| File / folder | What | In git |
|---|---|---|
| `benchmark.json` | The 16 entries (scene + trajectory) | yes |
| `final_images/` | The 16 scene pictures used by the benchmark | yes |
| `prompts.json` | One description per scene | yes |
| `traj/` | The 5 trajectories | yes |
| `images_cleaned/` | The 12 first frames with the key overlay removed (input to `final_images/`) | yes |
| `images/` | Raw first frames of the 16 videos | no |
| `videos/` | The authors' 16 source videos (401 MB) | no |

## Source videos

The 16 WorldKV videos from the authors' repo
[cvlab-kaist/WorldKV](https://github.com/cvlab-kaist/WorldKV), branch `main`,
commit [`315be40`](https://github.com/cvlab-kaist/WorldKV/tree/315be40cfe601b724d05bc2d4ba4ff676f10426d):
every one that is full length and has its trajectory
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
3. **Cropped and resized to 848 x 464** around the center, so every scene
   makes 464 x 848 videos: the 832 x 464 frames lose 4 pixels at the top and
   bottom and are then enlarged by about 2%; the 768 x 512 frames (15, uni01)
   lose 46 pixels at the top and bottom and are enlarged by about 10%. Scene
   00's original picture is used as is (it already makes 464 x 848 videos).
   Results in `final_images/`.
4. **Descriptions** (`prompts.json`): scene 00 uses the authors' own
   (`examples/new/prompts.json`); the other 15 were written by us from each
   final picture, in the same style, saying what is on the left, center,
   right and in the distance.
5. **Trajectories** (`traj/`): the authors did not release a trajectory file
   for each video, but every video's file name spells out its camera path. We
   read the path from the name and recreated the trajectory with the authors'
   own `generate_traj.py` (see [Trajectories](#trajectories)).

## Trajectories

A trajectory name such as `R_L_W_S_R_L90_318` reads as follows:

- **Letters:** the camera moves, in order.
  - `R` / `L`: turn right / left by the number written after the letters
    (here 90°), at 2° per frame, so each turn takes 45 frames.
  - `W` / `S`: move forward / backward by 1 unit, over 24 frames.
- **`_` between two moves:** the camera stays still for 16 frames. The name
  does not give this number; 16 is what the authors' own trajectory file
  `R_L_R_L_R_L90_360` uses. Names without `_` have no pauses.
- **Last number:** the total number of frames (here 318). After the last
  move, the camera stays still until that many frames.

Example: `R_L_W_S_R_L90_318` is right, left, forward, back, right, left,
with a pause between moves: 45 + 16 + 45 + 16 + 24 + 16 + 24 + 16 + 45 + 16 +
45 = 308 frames of movement, then 10 still frames, 318 in total. WorldKV's
generation uses only a whole number of chunks of these frames: 309 here,
which is the length of the authors' videos with this trajectory.

Each move is later undone by the opposite one (a right turn by a left turn,
forward by back), so the camera returns to poses it has already shown. These
returns are the revisits the metrics compare.

| Trajectory | Moves | Frames of movement | Frames in total | Frames in the video |
|---|---|---|---|---|
| `R_L_R_L_R_L90_360` | right, left, right, left, right, left; pauses | 350 | 360 | 357 |
| `R_L_W_S_R_L90_318` | right, left, forward, back, right, left; pauses | 308 | 318 | 309 |
| `L_R_W_S_L_R90_318` | left, right, forward, back, left, right; pauses | 308 | 318 | 309 |
| `RLWSRL90_245` | right, left, forward, back, right, left; no pauses | 228 | 245 | 237 |
| `WSRLRL90_245` | forward, back, right, left, right, left; no pauses | 228 | 245 | 237 |

`R_L_R_L_R_L90_360` is also in the authors' own `traj/` folder; generating it
from its name gives an identical file. For every trajectory, the frames in the
video match the frame count of the authors' videos that use it.

## Entries

| Scene (= source video) | Trajectory | Picture | Video frames | Long-gap pairs |
|---|---|---|---|---|
| [`00_R_L_R_L_R_L90_360_compressed`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/worldkvfullkv/00_R_L_R_L_R_L90_360_compressed.mp4) | `R_L_R_L_R_L90_360` | authors' original input | 357 | 206 |
| [`01_R_L_W_S_R_L90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/01_R_L_W_S_R_L90_318.mp4) | `R_L_W_S_R_L90_318` | frame 0, overlay removed | 309 | 113 |
| [`04_RLWSRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/04_RLWSRL90_245.mp4) | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| [`04_WSRLRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/04_WSRLRL90_245.mp4) | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| [`15_R_L_R_L_R_L90_360_compressed`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/worldkvfullkv/15_R_L_R_L_R_L90_360_compressed.mp4) | `R_L_R_L_R_L90_360` | frame 0 | 357 | 206 |
| [`20_R_L_W_S_R_L90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/20_R_L_W_S_R_L90_318.mp4) | `R_L_W_S_R_L90_318` | frame 0, overlay removed | 309 | 113 |
| [`21_L_R_W_S_L_R90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/21_L_R_W_S_L_R90_318.mp4) | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| [`23_L_R_W_S_L_R90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/23_L_R_W_S_L_R90_318.mp4) | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| [`23_RLWSRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/23_RLWSRL90_245.mp4) | `RLWSRL90_245` | frame 0 | 237 | 81 |
| [`24_WSRLRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/24_WSRLRL90_245.mp4) | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| [`26_L_R_W_S_L_R90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/26_L_R_W_S_L_R90_318.mp4) | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| [`29_L_R_W_S_L_R90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/29_L_R_W_S_L_R90_318.mp4) | `L_R_W_S_L_R90_318` | frame 0, overlay removed | 309 | 113 |
| [`31_WSRLRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/31_WSRLRL90_245.mp4) | `WSRLRL90_245` | frame 0, overlay removed | 237 | 108 |
| [`37_RLWSRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/37_RLWSRL90_245.mp4) | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| [`41_RLWSRL90_245`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/41_RLWSRL90_245.mp4) | `RLWSRL90_245` | frame 0, overlay removed | 237 | 81 |
| [`uni01_L_R_W_S_L_R90_318`](https://github.com/cvlab-kaist/WorldKV/blob/315be40cfe601b724d05bc2d4ba4ff676f10426d/world_videos/dynamic/uni01_L_R_W_S_L_R90_318.mp4) | `L_R_W_S_L_R90_318` | frame 0 | 309 | 113 |

In total 4,536 video frames (4.7 min of video). Long-gap pairs are revisits
whose first visit is more than 18 latent frames (72 video frames) earlier.

## Notes

- **Scene 04** appears twice with two trajectories; its two pictures are the
  same (they differ only by video compression noise).
- **Scene 23** appears twice, but the authors started the two videos from
  differently framed pictures; `23_L_R_W_S_L_R90_318` is more zoomed in and
  noticeably blurrier. Both are kept as the authors used them.
- **Pictures from video frames are a bit soft** (compressed web video); only
  scene 00 is the authors' sharp original. Except for 00, the descriptions
  are ours, not the authors'.
