import json
import os
import shutil

from PIL import Image

from eval import check_benchmark, runner

BENCH = os.path.join(runner.REPO_ROOT, "eval", "benchmark")


def test_video_size_matches_worldkv():
    # The repo's example (1280x704) and our crops (848x464) both make
    # 464x848 videos, the size of the authors' demo videos.
    assert check_benchmark.video_size(1280, 704) == (464, 848)
    assert check_benchmark.video_size(848, 464) == (464, 848)
    assert check_benchmark.video_size(768, 512) != (464, 848)  # a 3:2 picture


def test_dev_benchmark_passes():
    for name in ("dev",):
        problems, rows = check_benchmark.check(os.path.join(BENCH, name, "benchmark.json"))
        assert problems == [], (name, problems)
        assert rows


def test_problems_are_reported(tmp_path):
    # A copy of the smoke benchmark (130 frames: no long-gap pairs) with a
    # picture of the wrong shape (4:3).
    root = tmp_path / "bench"
    shutil.copytree(os.path.join(BENCH, "smoke", "traj"), root / "traj")
    os.makedirs(root / "images")
    Image.new("RGB", (800, 600), "gray").save(root / "images" / "00.png")
    (root / "prompts.json").write_text(json.dumps({"00": "a scene"}))
    (root / "benchmark.json").write_text(json.dumps({
        "name": "t", "images_dir": "images", "prompts": "prompts.json", "traj_dir": "traj",
        "pairs": [{"scene": "00", "trajectory": "RL90_130"}]}))
    problems, _ = check_benchmark.check(str(root / "benchmark.json"))
    assert any("expected 464x848" in p for p in problems)
    assert any("long-gap pairs" in p for p in problems)
