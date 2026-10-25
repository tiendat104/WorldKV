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
