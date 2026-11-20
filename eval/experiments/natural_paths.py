"""Natural-looking camera paths for the long-video experiments.

The authors' generate_traj.py builds paths from identical blocks (one turn
angle per path, fixed 1-unit moves, fixed pauses). This module uses the same
kinds of moves as the WorldKV paper's trajectories, walking straight forward or
backward and turning left or right in place, one at a time, but with varied
parameters: turns of any angle, walks of any length and pauses of any length.
It keeps generate_traj.py's conventions, so the camera moves at the same speeds
and the files look the same to WorldKV:

  - turns at 2 deg per frame (positive = right), walking 1 unit per 24 frames
    (frame counts are rounded up, so never faster);
  - camera-to-world poses rotated about the y axis, forward = (sin yaw, 0, cos yaw);
  - the first pose is the one after the first step; still frames are appended
    at the end to reach the requested length;
  - the same default intrinsics.

Run from the repo root to write the paths:
  python -m eval.experiments.natural_paths
"""

import math
import os

import numpy as np

import generate_traj as gt

DEG_PER_FRAME = gt.ROTATION_DEG_PER_FRAME
FRAMES_PER_UNIT = gt.TRANSLATION_FRAMES_PER_OP
HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_DIR = os.path.join(HERE, "paths")


class Path:
    """A camera path built move by move."""

    def __init__(self):
        self.yaw, self.xyz = 0.0, np.zeros(3)
        self.yaws, self.positions = [], []
        self.moves = []  # (symbol, frames), e.g. ("W2", 48), ("R60", 30), ("_12", 12)

    def _step(self, yaw_change, distance):
        yaw = np.deg2rad(self.yaw)
        self.xyz = self.xyz + distance * np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        self.yaw += yaw_change
        self.yaws.append(self.yaw)
        self.positions.append(self.xyz.copy())

    def turn(self, degrees):
        """Turn in place (positive = right) at up to 2 deg per frame."""
        frames = max(1, math.ceil(abs(degrees) / DEG_PER_FRAME - 1e-9))
        for _ in range(frames):
            self._step(degrees / frames, 0.0)
        self.moves.append((("R" if degrees > 0 else "L") + _number(abs(degrees)), frames))
        return self

    def walk(self, units):
        """Walk straight forward (negative = backward) at up to 1 unit per 24 frames."""
        frames = max(1, math.ceil(abs(units) * FRAMES_PER_UNIT - 1e-9))
        for _ in range(frames):
            self._step(0.0, units / frames)
        self.moves.append((("W" if units > 0 else "S") + _number(abs(units)), frames))
        return self

    def pause(self, frames):
        for _ in range(frames):
            self._step(0.0, 0.0)
        self.moves.append((f"_{frames}", frames))
        return self

    def look(self, degrees, hold=12):
        """Look to one side (positive = right), hold, and look back."""
        return self.turn(degrees).pause(hold).turn(-degrees)

    def notation(self):
        """The moves in order, e.g. "W2 R60 _12 L60": W/S = walk forward/backward
        (units), R/L = turn right/left in place (degrees), _ = still (frames)."""
        return " ".join(symbol for symbol, _ in self.moves)

    def poses(self, frames):
        """Camera-to-world poses, padded with still frames to `frames`."""
        if len(self.yaws) > frames:
            raise ValueError(f"path has {len(self.yaws)} frames, more than {frames}")
        yaws = np.array(self.yaws + [self.yaw] * (frames - len(self.yaws)), dtype=np.float32)
        xyz = np.array(self.positions + [self.xyz] * (frames - len(self.positions)),
                       dtype=np.float32)
        return gt.make_poses(yaws, xyz)


def _number(x):
    return f"{x:g}"


def inspect_area():
    """Walk up to something and look around it from a few nearby spots
    (revisits the same views often, after short gaps)."""
    return (Path().walk(2).look(60).look(-40, hold=10)
            .walk(1).look(90, hold=16).walk(-1).look(-70)
            .turn(15).walk(1.5).look(-50).turn(-15)
            .look(60, hold=10).walk(-1.5))


def room_tour():
    """Walk along the walls of a 3 x 2 room, turning at the corners, back to the
    start, then along the first wall again (revisits after a long gap: a loop)."""
    return (Path().walk(3).turn(90).look(-30, hold=8)
            .walk(2).turn(90).walk(3).look(45, hold=10)
            .turn(90).walk(2).turn(90)
            .walk(3).turn(90))


def go_and_return():
    """Walk out along a path with two bends, look around at the far end, turn
    around, walk back the same way and turn to face the start direction again
    (each place revisited once, late)."""
    out = [(3.0, 20.0), (2.0, -35.0), (2.0, 0.0)]  # (walk, then turn) per stretch
    path = Path()
    for units, turn in out:
        path.walk(units)
        if turn:
            path.turn(turn)
    path.look(-40, hold=10).turn(180)
    turns_back = [-turn for _, turn in out[:-1]][::-1]  # the bends, undone in reverse
    for (units, _), turn in zip(out[::-1], turns_back + [0.0]):
        path.walk(units)
        if turn:
            path.turn(turn)
    return path.turn(180)


def street_walk():
    """Walk along a street with a few corners, glancing around now and then
    (new places most of the time, few revisits)."""
    return (Path().walk(3).look(-35, hold=10).walk(2).turn(-25)
            .walk(2.5).look(50).walk(3).look(-20, hold=8).turn(30)
            .walk(2.5).look(30, hold=10).walk(2).turn(-20).walk(1.5))


SCENARIOS = {
    "inspect": inspect_area,
    "tour": room_tour,
    "return": go_and_return,
    "street": street_walk,
}
FRAMES = 645  # 54 chunks, 40.3 s at 16 fps


def write(name, poses):
    folder = os.path.join(PATHS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"{name}_poses.npy"), poses)
    np.save(os.path.join(folder, f"{name}_intrinsics.npy"),
            gt.make_intrinsics(len(poses), gt.DEFAULT_INTRINSIC))


def main():
    for scenario, build in SCENARIOS.items():
        path = build()
        name = f"{scenario}_{FRAMES}"
        write(name, path.poses(FRAMES))
        print(f"{name}: {len(path.yaws)} frames of movement, "
              f"{FRAMES - len(path.yaws)} still frames at the end")
        print(f"  {path.notation()}")


if __name__ == "__main__":
    main()
