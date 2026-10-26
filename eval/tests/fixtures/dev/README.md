# Dev test set

Five scenes for debugging the harness (Stage 0/1). Not the final benchmark.

| Scene | Content | Type |
|---|---|---|
| 00 | Game character by a stable on a stone road | Game, outdoor |
| 15 | Log cabin with fireplace, snow outside | AI-style, indoor |
| 13 | Neon cartoon city street at night | Stylized, city |
| uni01 | School sports field with running track | Real photo, outdoor |
| uni03 | Rural Japanese street with houses | Real photo, city |

## Where the pictures come from

- `00.png` is the repo's own example input (`examples/new/00.png`), with its
  original prompt.
- The others are frame 0 of the WorldKV authors' demo videos (the `main`
  branch of the upstream repo), e.g. `world_videos/worldkvfullkv/15_*.mp4`.
  Scenes whose videos show a "W A S D" key overlay were left out.
  These pictures come from compressed web videos, so they are a bit soft.

All pictures are cropped (centered) to 848 x 464, the same shape as `00.png`,
so every scene generates 464 x 848 videos with `--size 480*832`.

`prompts.json` follows the format of `examples/new/prompts.json`. Prompts
other than `00` were written by us in the same style.

## Trajectories

Three trajectories in `traj/`, named like the authors' demo videos. The number
at the end is the `--frame_num` to pass to `generate_fast.py`.

| Trajectory | Camera | Video frames | Pairs (long / short / sink) |
|---|---|---|---|
| `R_L_R_L_R_L90_360` | turn right/left three times | 357 | 206 / 60 / 45 |
| `L_R_W_S_L_R90_318` | turn, walk forward and back, turn | 309 | 113 / 99 / 27 |
| `WSRLRL90_245` | walk forward and back, then turn | 237 | 108 / 50 / 9 |

In a trajectory name, `R`/`L` turn 90 deg (45 frames), `W`/`S` walk forward
or back (24 frames) and `_` is a 16-frame pause. `R_L_R_L_R_L90_360` is a copy
of the repo's `traj/` file; the other two were made with:

```bash
python generate_traj.py --frames 318 --trajectory L_R_W_S_L_R90 --noop-frames 16 --output-dir eval/tests/fixtures/dev/traj
python generate_traj.py --frames 245 --trajectory WSRLRL90 --output-dir eval/tests/fixtures/dev/traj
```

With these settings, regenerating `R_L_R_L_R_L90_360` reproduces the repo's
file exactly.
