# Full benchmark (first batch)

24 scene-trajectory pairs for the final runs, following the paper's recipe
(Sec. 5.1): 4 kinds of places (indoor, outdoor, city, nature) x 3 sources
(real photos, game screenshots, AI-generated), 2 scenes each. Each scene is
paired with one camera trajectory that returns to earlier views after a long
time.

Check it before any GPU run:

```bash
python -m eval.check_benchmark eval/benchmark/full/benchmark.json
```

## Files

| File | What |
|---|---|
| `benchmark.json` | The 24 (scene, trajectory) pairs; read by `eval.runner` |
| `images/<scene>.png` | Starting pictures, 848 x 464 (all make 464 x 848 videos) |
| `prompts.json` | One detailed description per scene, in the authors' style |
| `traj/` | The 4 camera trajectories |
| `scenes.json` | Per scene: kind, original file, crop, source |
| `contact_sheet.jpg` | All 24 pictures, to check by eye |

## Trajectories

8 trajectory types, 3 scenes each (each from a different kind of place). In a
name, `R`/`L` turn by the given angle (2 deg per frame), `W`/`S` walk forward
or back and `A`/`D` step left or right (24 frames each), `_` is a 16-frame
pause, and the last number is the length to pass as `--frame_num`.

| Trajectory | Camera | Video | Long-gap pairs | Scenes | Source |
|---|---|---|---|---|---|
| `R_L_R_L_R_L90_360` | turn right/left 90 deg three times, with pauses | 357 frames (22 s) | 206 | 3 | authors |
| `L_R_L_R_L_R90_360` | the same, turning left first | 357 frames (22 s) | 206 | 3 | like the authors' L_R_L_R_L_R_385 |
| `L_R_W_S_L_R90_318` | turn, walk forward and back, turn | 309 frames (19 s) | 113 | 3 | authors |
| `R_L_W_S_R_L90_318` | the same, turning right first | 309 frames (19 s) | 113 | 3 | authors |
| `WSRLRL90_245` | walk forward and back, then turn (no pauses) | 237 frames (15 s) | 108 | 3 | authors |
| `R_L_R_L120_300` | turn right/left 120 deg twice, with pauses | 297 frames (19 s) | 148 | 3 | new |
| `D_A_R_L_D_A90_280` | step sideways right/left, turn, step sideways again | 273 frames (17 s) | 86 | 3 | new |
| `R_L_R180_320` | turn 180 deg right, back left, and right again | 309 frames (19 s) | 140 | 3 | new |

The authors' patterns come from the names of their demo videos. All were made
with `generate_traj.py ... --noop-frames 16 --output-dir eval/benchmark/full/traj`
(no `--noop-frames` for `WSRLRL90_245`), e.g.:

```bash
python generate_traj.py --frames 280 --trajectory D_A_R_L_D_A90 --noop-frames 16 --output-dir eval/benchmark/full/traj
```

Trajectories are assigned in turn over the scenes sorted by name.

## Pictures

Collected by hand into `benchmark_raw/<place>_<kind>/` (outside the repo),
with sources noted in `benchmark_raw/sources.txt`, then cropped with:

```bash
python -m eval.prepare_images ../benchmark_raw --out eval/benchmark/full \
    --zoom indoor_game/69312fe0903508ad41307016177abaf4.jpg=1.1 \
    --zoom indoor_game/luther-s-schmidt-screen-logo-01.jpg=1.1 \
    --zoom nature_game/isaiah-jo-4.jpg=1.02
```

The zooms cut off corner logos (indoor_game) and a thin dark line at the top
edge (nature_game_2). Black bars on nature_real_2 are removed automatically.
Pictures marked "enlarged" were smaller than 848 x 464 and look a bit soft.
Pictures with watermarks or on-screen graphics were left out.

Sources were recorded from file names after download; "unknown web source"
means the site was not noted.

| Scene | Trajectory | Original file | Source |
|---|---|---|---|
| city_ai_1 | R_L_R_L_R_L90_360 | ai_city_image_2.webp | unknown web source |
| city_ai_2 | L_R_L_R_L_R90_360 | city-of-dreams-presenta-metropolis-futurista_462 | likely Freepik (file-name pattern) (enlarged) |
| city_game_1 | L_R_W_S_L_R90_318 | ax9d7rfeh5c91.png | Cyberpunk 2077 screenshot, likely Reddit (file-name pattern) |
| city_game_2 | R_L_W_S_R_L90_318 | i-recreated-an-entire-scene-from-the-game-cyberp | Reddit post "I recreated an entire scene from the game Cyberpunk 2077 in ..." (enlarged) |
| city_real_1 | WSRLRL90_245 | 234595cf4a3f24a584734c76a820f41c9d34438a-1920x12 | unknown web source |
| city_real_2 | R_L_R_L120_300 | new-york-city-street-scene-photo-by-paul-katcher | photo by Paul Katcher (per file name) |
| indoor_ai_1 | D_A_R_L_D_A90_280 | 7650_26d32fba_How_AI_is_Reshaping_Interior_Desig | web article "How AI is Reshaping Interior Design" |
| indoor_ai_2 | R_L_R180_320 | sustainable-architecture-pedram-ki-studio-7.jpg | Pedram Ki Studio (per file name) |
| indoor_game_1 | R_L_R_L_R_L90_360 | 69312fe0903508ad41307016177abaf4.jpg | unknown web source (Pinterest-style file name) (enlarged) |
| indoor_game_2 | L_R_L_R_L_R90_360 | luther-s-schmidt-screen-logo-01.jpg | artwork by Luther S. Schmidt (per file name) |
| indoor_real_1 | L_R_W_S_L_R90_318 | 0c5a3a7e8e59ec622f0cdfc7f9c2998476c047f2.jpeg | unknown web source |
| indoor_real_2 | R_L_W_S_R_L90_318 | infinity-vision-interior-living-room-scene-011-p | "Infinity Vision" 3D interior scene preview (per file name) |
| nature_ai_1 | WSRLRL90_245 | ai-generated-7483596_960_720.jpg | likely Pixabay (file-name pattern) |
| nature_ai_2 | R_L_R_L120_300 | m10.avif | unknown web source |
| nature_game_1 | D_A_R_L_D_A90_280 | adventure-game-background_670382-169567.jpg | likely Freepik (file-name pattern) (enlarged) |
| nature_game_2 | R_L_R180_320 | isaiah-jo-4.jpg | artwork by Isaiah Jo (per file name) |
| nature_real_1 | R_L_R_L_R_L90_360 | beautiful-shot-natural-scenery-autumn_181624-259 | likely Freepik (file-name pattern) (enlarged) |
| nature_real_2 | L_R_L_R_L_R90_360 | sddefault.jpg | YouTube video thumbnail, video unknown (enlarged) |
| outdoor_ai_1 | L_R_W_S_L_R90_318 | TkEEk7mXEt9J.png | unknown web source |
| outdoor_ai_2 | R_L_W_S_R_L90_318 | ai-front-yard-design-curb-appeal-cover-177617462 | unknown web article on AI front-yard design |
| outdoor_game_1 | WSRLRL90_245 | Fullacornacres.jpg | Toontown, Acorn Acres (per file name) |
| outdoor_game_2 | R_L_R_L120_300 | Gardenlife-Screenshot-2024.02.27-21.43.18.80.jpg | Garden Life game screenshot (per file name) |
| outdoor_real_1 | D_A_R_L_D_A90_280 | a-pathway-in-a-park-with-benches-and-trees-photo | likely Vecteezy (file-name pattern) (enlarged) |
| outdoor_real_2 | R_L_R180_320 | a-peaceful-rural-scene-showcasing-a-red-barn-and | likely Vecteezy "free photo" (file-name pattern) |
