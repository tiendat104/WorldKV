import numpy as np

from eval import authors_videos


def test_shift_check_peaks_at_zero_when_aligned():
    # A video whose frame j shows the same "view" as frame i for each pair,
    # and different views elsewhere: shift 0 must score best.
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, size=(40, 16, 16, 3), dtype=np.uint8)
    pairs = [{"first": i, "revisit": i + 20} for i in range(5, 15)]
    for p in pairs:
        video[p["revisit"]] = video[p["first"]]
    shifts = authors_videos.shift_check(video, pairs, shifts=range(-2, 3), scale=1)
    assert max(shifts, key=shifts.get) == 0


def test_shift_check_detects_an_offset():
    # If the video is one frame "late" compared with the trajectory, shift +1 wins.
    rng = np.random.default_rng(1)
    video = rng.integers(0, 256, size=(40, 16, 16, 3), dtype=np.uint8)
    pairs = [{"first": i, "revisit": i + 20} for i in range(5, 15)]
    for p in pairs:
        video[p["revisit"] + 1] = video[p["first"]]
    shifts = authors_videos.shift_check(video, pairs, shifts=range(-2, 3), scale=1)
    assert max(shifts, key=shifts.get) == 1


def test_video_list():
    names = [v[0] for v in authors_videos.VIDEOS]
    assert len(names) == len(set(names)) == 21
    for path, scene, traj, method, _ in authors_videos.VIDEOS:
        assert method in ("sliding", "full_kv", "worldkv")
        assert traj in path and path.split("/")[-1].startswith(scene + "_")
