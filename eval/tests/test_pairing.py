import os

import numpy as np
import pytest

from eval import pairing
from generate_traj import build_yaw_series, make_poses

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RLRLRL_DIR = os.path.join(REPO_ROOT, "traj", "R_L_R_L_R_L90_360")


def test_num_video_frames_matches_worldkv_trimming():
    # 360 poses -> 357 frames -> 90 latent frames (a multiple of 3).
    assert pairing.num_video_frames(360, 360) == 357
    # 100 frames -> 25 latent -> trimmed to 24 latent -> 93 frames.
    assert pairing.num_video_frames(100, 100) == 93


def test_latent_index():
    assert [pairing.latent_index(f) for f in range(10)] == [0, 1, 1, 1, 1, 2, 2, 2, 2, 3]


@pytest.fixture(scope="module")
def rlrlrl():
    return pairing.build_pairs(RLRLRL_DIR, frame_num=360)


# R_L_R_L_R_L90 is "turn right, pause, turn left, pause, ...": each 90 deg turn
# takes 45 frames and each "_" pause takes 16 frames. So frames 0-44 turn right
# (2 -> 90 deg), 45-60 pause, 61-105 turn left (88 -> 0 deg), 106-121 pause,
# 122-166 turn right again.


def test_rlrlrl_known_pairs(rlrlrl):
    # Yaw 40 deg is first seen at frame 19 (first right turn), then again at
    # frame 85 (turning back left) and frame 141 (second right turn).
    by_revisit = {p["revisit"]: p for p in rlrlrl["pairs"]}
    assert by_revisit[85]["first"] == 19
    assert by_revisit[85]["gap_frames"] == 66
    assert by_revisit[85]["gap_latent"] == 17
    assert by_revisit[141]["first"] == 19
    assert by_revisit[141]["gap_frames"] == 122


def test_rlrlrl_sink_flag(rlrlrl):
    # Frame 0 (yaw 2 deg) is revisited at frame 104 and lies in the sink.
    by_revisit = {p["revisit"]: p for p in rlrlrl["pairs"]}
    assert by_revisit[104]["first"] == 0
    assert by_revisit[104]["in_sink"]
    assert not by_revisit[85]["in_sink"]


def test_pairs_point_to_earliest_visit(rlrlrl):
    for p in rlrlrl["pairs"]:
        assert p["first"] < p["revisit"] < rlrlrl["num_video_frames"]
    # The first right turn (frames 0..44) contains no revisits.
    assert min(p["revisit"] for p in rlrlrl["pairs"]) >= 45


def test_forward_backward_translation():
    # W then S: walking back passes the positions of the walk forward.
    yaw, xyz, *_ = build_yaw_series(["W", "S"], 0.0, [], translation_distance=1.0)
    poses = make_poses(yaw, xyz).astype(np.float64)
    pairs = {p["revisit"]: p for p in pairing.find_pairs(poses)}
    # Frame 24 (first step back) is at the position of frame 22.
    assert pairs[24]["first"] == 22
    assert 23 not in pairs  # frames during the walk forward are all new
