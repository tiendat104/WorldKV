# Experiments: offline retrieval schedule

Inputs and scripts for the experiments on the offline retrieval schedule
(runner method `offline`: the retrieval schedule is computed from the known
camera path, each stored chunk is freed after its last use, and each chunk's
retrieved KV is prefetched to the GPU while the previous chunk is generated,
reusing what is already there). Code version: tag `phase1-done`.

| Experiment | Question | Methods | Run list |
| --- | --- | --- | --- |
| E1 | Same video as WorldKV, speed, GPU memory (short videos) | WorldKV (GPU), WorldKV (CPU), ours | `e1.json` (4 benchmark entries) |
| E2 | GPU memory vs video length | WorldKV (GPU), WorldKV (CPU), ours; Full KV | `e2_long.json` (one 40-chunk video), `e2_full_kv.json` (7 lengths) |
| E3 | CPU memory and speed on long videos with different camera movements | WorldKV (CPU), ours | `e3.json` (4 natural 40 s paths) |

## Files

| File | What |
| --- | --- |
| `natural_paths.py` | Builds natural-looking camera paths from straight walks and turns in place with varied angles, distances and pauses (the moves of the WorldKV paper's trajectories, at generate_traj.py's speeds) |
| `prepare.py` | Writes all camera paths (`paths/`) and run lists (`*.json`): `python -m eval.experiments.prepare` |
| `predict.py` | Predicts each run's bank, CPU -> GPU copies and GPU memory before running: `python -m eval.experiments.predict` |
| `run_experiments.sh` | Runs E1, E2, E3 on a 4x A40 pod set up with `eval/setup_pod.sh` (each video in its own process, no quality metrics) |
| `paths/` | Camera paths: `R_L_R_L_R_L_R_L90_<frames>` (E2, made with generate_traj.py; shorter ones are its first frames), `<scenario>_645` (E3) |

## E3 camera paths

`W2` = walk forward 2 units, `S1` = walk backward 1 unit, `R60` / `L40` = turn
right 60 / left 40 degrees in place, `_12` = still for 12 frames. Each path is
645 frames (54 chunks, 40.3 s), padded with still frames at the end.

| Scenario | Moves |
| --- | --- |
| `inspect` | W2 R60 _12 L60 L40 _10 R40 W1 R90 _16 L90 S1 L70 _12 R70 R15 W1.5 L50 _12 R50 L15 R60 _10 L60 S1.5 |
| `tour` | W3 R90 L30 _8 R30 W2 R90 W3 R45 _10 L45 R90 W2 R90 W3 R90 |
| `return` | W3 R20 W2 L35 W2 L40 _10 R40 R180 W2 R35 W2 L20 W3 R180 |
| `street` | W3 L35 _10 R35 W2 L25 W2.5 R50 _12 L50 W3 L20 _8 R20 R30 W2.5 R30 _10 L30 W2 L20 W1.5 |

## Running

On the pod, after `eval/setup_pod.sh`:

    bash /workspace/WorldKV/eval/experiments/run_experiments.sh          # e1 e2 e3
    bash /workspace/WorldKV/eval/experiments/run_experiments.sh e2       # one experiment

Results go to `/workspace/experiments/<e1|e2_long|e2_full_kv|e3>/`. Each
experiment's log ends with `=== <experiment> done`; E1 and E3 also get
`checks.txt` (identical videos and bank sizes, from `eval/check_offline.py`).
`DRY_RUN=1` prints the commands without running anything.
