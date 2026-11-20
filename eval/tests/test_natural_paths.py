import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import generate_traj as gt
from eval import pairing
from eval.experiments import natural_paths as npth


def test_same_poses_as_the_authors_generator():
    # Walk 1, turn right 90, pause 16, turn back: "WR_L90" in generate_traj.py.
    yaw, xyz, *_ = gt.build_yaw_series(list("WR_L"), 90.0, [16], translation_distance=1.0)
    frames = len(yaw) + 5
    authors = gt.make_poses(np.concatenate([yaw, np.full(5, yaw[-1], np.float32)]),
                            np.concatenate([xyz, np.repeat(xyz[-1:], 5, axis=0)]))
    ours = npth.Path().walk(1).turn(90).pause(16).turn(-90).poses(frames)
    np.testing.assert_allclose(ours, authors, atol=1e-6)


@pytest.mark.parametrize("name", list(npth.SCENARIOS))
def test_only_the_papers_moves_at_the_authors_speeds(name):
    # Each frame either walks straight or turns in place (never both), no faster
    # than generate_traj.py: 1/24 unit or 2 deg per frame.
    poses = npth.SCENARIOS[name]().poses(npth.FRAMES)
    assert poses.shape == (npth.FRAMES, 4, 4)
    step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    turn = np.degrees((Rotation.from_matrix(poses[1:, :3, :3]).inv()
                       * Rotation.from_matrix(poses[:-1, :3, :3])).magnitude())
    assert step.max() <= 1 / npth.FRAMES_PER_UNIT + 1e-6
    assert turn.max() <= npth.DEG_PER_FRAME + 1e-4
    assert not np.any((step > 1e-6) & (turn > 1e-3))


def test_loop_and_return_come_back_to_the_start():
    for build in (npth.room_tour, npth.go_and_return):
        path = build()
        start = np.array(path.positions[0])
        closest = min(np.linalg.norm(np.array(p) - start) for p in path.positions[100:])
        assert closest < 1e-6, build.__name__
    assert np.allclose(npth.go_and_return().xyz, 0, atol=1e-6)


def test_written_paths_match_the_scenarios():
    for name, build in npth.SCENARIOS.items():
        folder = os.path.join(npth.PATHS_DIR, f"{name}_{npth.FRAMES}")
        np.testing.assert_allclose(pairing.load_poses(folder), build().poses(npth.FRAMES),
                                   atol=1e-6)
