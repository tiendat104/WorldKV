import importlib.util
import os

import numpy as np
import pytest

from eval import pairing
from generate_traj import build_yaw_series, make_poses

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RLRLRL_DIR = os.path.join(REPO_ROOT, "traj", "R_L_R_L_R_L90_360")
BENCH_TRAJ = os.path.join(REPO_ROOT, "eval", "benchmark", "traj")


def worldkv_latent_poses(poses, frame_num):
    """Latent-frame poses computed with the same steps as wan/image2video_fast.py."""
    spec = importlib.util.spec_from_file_location(
        "cam_utils", os.path.join(REPO_ROOT, "wan", "utils", "cam_utils.py"))
    cam_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam_utils)
    len_c2ws = ((len(poses) - 1) // 4) * 4 + 1
    c2ws = poses[:min(((frame_num - 1) // 4) * 4 + 1, len_c2ws)]
    lat_f = (len(c2ws) - 1) // 4 + 1
    lat_f -= lat_f % 3
    n = len(c2ws)
    return cam_utils.interpolate_camera_poses(
        src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, n - 1, lat_f)).numpy()


@pytest.mark.parametrize("name", ["R_L_W_S_R_L90_318", "RLWSRL90_245", "R_L_R_L_R_L90_360"])
def test_generation_poses_match_worldkv(name):
    # Latent frame k is video frame 4k, so every 4th pose must be WorldKV's.
    poses = pairing.load_poses(os.path.join(BENCH_TRAJ, name))
    frame_num = int(name.rsplit("_", 1)[1])
    ours = pairing.generation_poses(poses, frame_num)
    assert len(ours) == pairing.num_video_frames(len(poses), frame_num)
    np.testing.assert_allclose(ours[::4], worldkv_latent_poses(poses, frame_num), atol=1e-5)


def test_no_stretch_when_frame_count_fits():
    # Asking for 309 frames keeps 309 poses for a 309-frame video: pose j is frame j.
    traj_dir = os.path.join(BENCH_TRAJ, "R_L_W_S_R_L90_318")
    poses = pairing.load_poses(traj_dir)
    np.testing.assert_array_equal(pairing.generation_poses(poses, 309), poses[:309])
    assert pairing.build_pairs(traj_dir, 309)["stretch"] == 1.0


def test_stretched_path_pairs_same_viewpoints():
    # Asking for 318 frames keeps 317 poses but makes a 309-frame video, so
    # generation stretches the path. Frames 29 and 75 both face 60 deg in the
    # file, but more than 5 deg apart in the video.
    traj_dir = os.path.join(BENCH_TRAJ, "R_L_W_S_R_L90_318")
    poses = pairing.load_poses(traj_dir)
    gen = pairing.generation_poses(poses, 318)
    assert pairing.rotation_distance_deg(poses[None, 29, :3, :3], poses[75, :3, :3])[0] < 0.1
    assert pairing.rotation_distance_deg(gen[None, 29, :3, :3], gen[75, :3, :3])[0] > 5

    result = pairing.build_pairs(traj_dir, 318)
    assert result["stretch"] == pytest.approx(316 / 308)
    for p in result["pairs"]:
        i, j = p["first"], p["revisit"]
        assert pairing.rotation_distance_deg(gen[None, i, :3, :3], gen[j, :3, :3])[0] <= pairing.ROT_TOL_DEG
        assert np.linalg.norm(gen[i, :3, 3] - gen[j, :3, 3]) <= pairing.TRANS_TOL


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
