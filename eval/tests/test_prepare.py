import os
import subprocess
import sys

import numpy as np
import pytest

from eval import pairing, runner
from eval.experiments import prepare

RUN_LISTS = {  # file -> the methods it is run with
    "e1.json": ["gpu_only", "cpu_only", "offline"],
    "e2_long.json": ["gpu_only", "cpu_only", "offline"],
    "e2_full_kv.json": ["full_kv"],
    "e3.json": ["cpu_only", "offline"],
}


@pytest.mark.parametrize("name", list(RUN_LISTS))
def test_run_lists_load_and_fit(name):
    entries = runner.load_benchmark(os.path.join(prepare.HERE, name))
    runner.check_video_lengths(entries, RUN_LISTS[name])  # at least WorldKV's window
    for entry in entries:
        assert os.path.exists(entry["image"])
        if name != "e1.json":
            # Our paths are a whole number of chunks, so WorldKV uses every pose as given.
            num_poses = len(pairing.load_poses(entry["traj_dir"]))
            assert num_poses == entry["frame_num"]
            assert pairing.num_video_frames(num_poses, entry["frame_num"]) == num_poses
            assert (num_poses + 3) % 12 == 0


def test_run_list_contents():
    load = lambda name: runner.load_benchmark(os.path.join(prepare.HERE, name))
    assert [e["scene"] for e in load("e1.json")] == [s for s, _ in prepare.E1_PAIRS]
    assert [e["frame_num"] for e in load("e2_long.json")] == [477]
    assert [e["frame_num"] for e in load("e2_full_kv.json")] == [105, 213, 309, 321, 333, 345, 357]
    assert [e["frame_num"] for e in load("e3.json")] == [645] * 4


def test_e2_path_is_what_generate_traj_writes(tmp_path):
    subprocess.run([sys.executable, "generate_traj.py", "--frames", "477",
                    "--trajectory", prepare.E2_TRAJECTORY, "--noop-frames", "16",
                    "--output-dir", str(tmp_path)], cwd=runner.REPO_ROOT, check=True,
                   capture_output=True)
    name = f"{prepare.E2_TRAJECTORY}_477"
    authors = pairing.load_poses(str(tmp_path / name))
    ours = pairing.load_poses(os.path.join(prepare.PATHS_DIR, name))
    np.testing.assert_array_equal(ours, authors)


def test_run_script_dry_run(tmp_path):
    env = dict(os.environ, WORKSPACE=str(tmp_path), REPO=runner.REPO_ROOT,
               OUT=str(tmp_path / "out"), PYTHON=sys.executable, DRY_RUN="1")
    result = subprocess.run(["bash", os.path.join(prepare.HERE, "run_experiments.sh")],
                            env=env, capture_output=True, text=True, timeout=300)
    log = result.stdout
    assert result.returncode == 0, result.stderr
    for line in ("4 scene-trajectory pairs x 3 methods = 12 runs",       # e1
                 "1 scene-trajectory pairs x 3 methods = 3 runs",        # e2, long video
                 "7 scene-trajectory pairs x 1 methods = 7 runs",        # e2, Full KV lengths
                 "4 scene-trajectory pairs x 2 methods = 8 runs",        # e3
                 "=== e1 done", "=== e2 done", "=== e3 done", "=== all done"):
        assert line in log, line
    assert log.count("runner exit 0") == 4
    assert log.count("one worker per video") == 4


def test_full_kv_lengths_are_the_start_of_the_long_path():
    full = pairing.load_poses(os.path.join(prepare.PATHS_DIR, f"{prepare.E2_TRAJECTORY}_477"))
    for chunks in prepare.FULL_KV_CHUNKS:
        frames = prepare.frames_for(chunks)
        part = pairing.load_poses(os.path.join(prepare.PATHS_DIR, f"{prepare.E2_TRAJECTORY}_{frames}"))
        np.testing.assert_array_equal(part, full[:frames])
