"""Our own leader -> follower teleop, so touch sensing runs in the SAME loop.

`lerobot-teleoperate` owns the follower's serial port while it runs, and a port can only
be opened by one process. So touch sensing cannot run next to it: this loop replaces it.

Mapping (same as lerobot with use_degrees=false): each joint is mapped linearly from the
leader's calibrated range to the follower's calibrated range, in raw ticks:

    follower = f_min + (leader - l_min) / (l_max - l_min) * (f_max - f_min)

Safety: the leader is only ever READ (torque never enabled). Every follower goal is
clamped to its range, limited to `max_step` ticks per tick (like lerobot's
max_relative_target), and blended in over `ramp_s` at startup so the follower never snaps
to the leader pose.
"""
from __future__ import annotations

import math
import os
import time

from . import config


def map_leader_to_follower(leader: dict[str, int], lr: dict, fr: dict) -> dict[str, float]:
    out = {}
    for j, v in leader.items():
        (l0, l1), (f0, f1) = lr[j], fr[j]
        s = min(max((v - l0) / max(l1 - l0, 1), 0.0), 1.0)
        out[j] = f0 + s * (f1 - f0)
    return out


class LeaderArm:
    """The black leader. Read-only: we never write a register on it."""

    def __init__(self, port: str = config.LEADER_PORT):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        if not os.path.exists(port):
            raise SystemExit(f"Leader not found at {port}. Plug it in, or use --fake.")
        self.ranges = config.load_ranges(config.LEADER_CALIBRATION_JSON)
        motors = {j: Motor(config.MOTOR_IDS[j], "sts3215", MotorNormMode.RANGE_M100_100) for j in config.JOINTS}
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
        self.bus.connect()

    def read(self) -> dict[str, int]:
        return self.bus.sync_read("Present_Position", config.JOINTS, normalize=False, num_retry=1)

    def close(self) -> None:
        self.bus.disconnect(disable_torque=False)   # never touch the leader's torque


class FakeLeader:
    """Smooth pseudo-random leader motion with PAUSES (still periods), like a person
    teleoperating: move ~4 s, hold ~3 s. The pauses are when the static baseline re-learns."""

    def __init__(self, move_s: float = 4.0, pause_s: float = 3.0, seed: int = 1):
        try:
            self.ranges = config.load_ranges(config.LEADER_CALIBRATION_JSON)
        except FileNotFoundError:
            self.ranges = {j: (900, 3100) for j in config.JOINTS}
        self.move_s, self.pause_s = move_s, pause_s
        self.t0 = time.perf_counter()
        self.phase = {j: 1.7 * i + seed for i, j in enumerate(config.JOINTS)}
        self.freq = {j: 0.15 + 0.07 * i for i, j in enumerate(config.JOINTS)}
        self.amp = {"shoulder_pan": 0.25, "shoulder_lift": 0.18, "elbow_flex": 0.18, "wrist_flex": 0.2,
                    "wrist_roll": 0.15, "gripper": 0.2}
        self._clock = 0.0
        self._last = self.t0

    def read(self) -> dict[str, int]:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        cyc = (now - self.t0) % (self.move_s + self.pause_s)
        if cyc < self.move_s:
            self._clock += dt                           # the motion clock stops during pauses
        out = {}
        for j in config.JOINTS:
            lo, hi = self.ranges[j]
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            out[j] = int(mid + self.amp[j] * half * math.sin(2 * math.pi * self.freq[j] * self._clock + self.phase[j]))
        return out

    def close(self) -> None:
        pass


class Teleop:
    def __init__(self, leader, follower_ranges: dict, max_step: float = 40.0, ramp_s: float = 2.0):
        self.leader, self.fr = leader, follower_ranges
        self.max_step, self.ramp_s = max_step, ramp_s
        self.t_start = None
        self.start_pose = None
        self.last_goal = None

    def goals(self, t: float, follower_pos: dict[str, int]) -> dict[str, float]:
        target = map_leader_to_follower(self.leader.read(), self.leader.ranges, self.fr)
        if self.t_start is None:
            self.t_start, self.start_pose = t, dict(follower_pos)
            self.last_goal = {j: float(v) for j, v in follower_pos.items()}
        a = min((t - self.t_start) / self.ramp_s, 1.0) if self.ramp_s > 0 else 1.0
        out = {}
        for j in config.JOINTS:
            want = (1 - a) * self.start_pose[j] + a * target[j]
            prev = self.last_goal[j]
            step = max(-self.max_step, min(self.max_step, want - prev))
            out[j] = prev + step
        self.last_goal = out
        return out
