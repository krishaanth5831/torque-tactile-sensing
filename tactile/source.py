"""Where the numbers come from: the real follower arm, or a fake one.

Both sources share one interface, so every script runs with `--fake` before you
plug the arm in:

    src = open_source(fake=False)
    s = src.read()            # Sample: one reading of every register, all joints
    src.hold_here(["elbow_flex"], torque_limit=300)
    src.set_goal({"elbow_flex": 2100})
    src.close()

All values are RAW servo units. What they mean on the STS3215:

    pos      Present_Position, 0..4095 ticks per turn (already offset by the
             homing_offset stored in EEPROM)
    goal     the Goal_Position we last sent (tracked here, not read back)
    err      goal - pos. The servo's position loop is mostly P, so in steady
             state drive ~= P * err. That makes err a free force sensor, and
             often a cleaner one than load.
    load     Present_Load, -1000..1000 = -100..100 % drive duty (PWM). lerobot
             decodes the sign (bit 10). This is what the motor is TRYING to push.
             It is not measured torque.
    cur      Present_Current, raw units (about 6.5 mA each per Feetech docs,
             UNVERIFIED on this arm). Motor torque ~= Kt * current, so this is
             the closest thing to real torque. lerobot does not decode a sign
             for it. If bit 15 is set we treat it as sign-magnitude. Check
             `cur_raw` in the CSV the first time.
    vel      Present_Velocity, ticks/s-ish (sign decoded by lerobot)
    status   Status register bitfield. Non-zero means an error or protection
             trip (overload, overheat, voltage). A tripped servo goes soft,
             which looks EXACTLY like a slip or a crushed fruit. Always check.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from . import config

SIGNALS = ["pos", "goal", "err", "load", "cur", "cur_raw", "vel", "status"]


@dataclass
class Sample:
    t: float                                   # seconds since the source opened
    pos: dict[str, int] = field(default_factory=dict)
    goal: dict[str, int | None] = field(default_factory=dict)
    err: dict[str, int | None] = field(default_factory=dict)
    load: dict[str, int] = field(default_factory=dict)
    cur: dict[str, int] = field(default_factory=dict)
    cur_raw: dict[str, int] = field(default_factory=dict)
    vel: dict[str, int] = field(default_factory=dict)
    status: dict[str, int] = field(default_factory=dict)


def decode_current(raw: int) -> int:
    """Sign-magnitude on bit 15, if the servo ever sets it. Positive values pass through."""
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


class _Base:
    joints = config.JOINTS

    def __init__(self, ranges: dict[str, tuple[int, int]]):
        self.ranges = ranges
        self.goals: dict[str, int | None] = {j: None for j in self.joints}
        self.t0 = time.perf_counter()

    def clamp(self, joint: str, raw: float) -> int:
        lo, hi = self.ranges[joint]
        return int(min(max(raw, lo), hi))

    def _fill_err(self, s: Sample) -> Sample:
        for j in self.joints:
            g = self.goals[j]
            s.goal[j] = g
            s.err[j] = None if g is None else g - s.pos[j]
        return s


# --------------------------------------------------------------------------- real arm
class ArmSource(_Base):
    """The white follower over the Feetech bus. Raw units, our own clamping."""

    def __init__(self, port: str = config.FOLLOWER_PORT, full: bool = True):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        super().__init__(config.load_ranges())
        motors = {j: Motor(config.MOTOR_IDS[j], "sts3215", MotorNormMode.RANGE_M100_100) for j in self.joints}
        # calibration=None: we never normalise, so lerobot needs no calibration object.
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
        self.bus.connect()
        self.full = full  # False = only pos/load/current, for a faster loop
        self._torque_on: set[str] = set()
        self._orig_torque_limit: dict[str, int] = {}

    def _r(self, reg: str) -> dict[str, int]:
        return self.bus.sync_read(reg, self.joints, normalize=False, num_retry=1)

    def read(self) -> Sample:
        s = Sample(t=time.perf_counter() - self.t0)
        s.pos = self._r("Present_Position")
        s.load = self._r("Present_Load")
        s.cur_raw = self._r("Present_Current")
        s.cur = {j: decode_current(v) for j, v in s.cur_raw.items()}
        if self.full:
            s.vel = self._r("Present_Velocity")
            s.status = self._r("Status")
        return self._fill_err(s)

    def safety_report(self) -> dict[str, dict[str, int]]:
        """Read back the protection registers. A bare FeetechMotorsBus does NOT run
        so_follower.configure(), so lerobot's 50 % gripper caps may not be live."""
        return {reg: self._r(reg) for reg in config.SAFETY_REGISTERS}

    def set_torque_limit(self, joints: list[str], value: int) -> None:
        value = int(min(max(value, 0), 1000))
        for j in joints:
            if j not in self._orig_torque_limit:
                self._orig_torque_limit[j] = self.bus.read("Torque_Limit", j, normalize=False)
            self.bus.write("Torque_Limit", j, value, normalize=False)

    def hold_here(self, joints: list[str], torque_limit: int) -> None:
        """Goal = where it is now, cap the torque, THEN switch torque on. No jump."""
        pos = self._r("Present_Position")
        goal = {j: self.clamp(j, pos[j]) for j in joints}
        self.bus.sync_write("Goal_Position", goal, normalize=False)
        self.goals.update(goal)
        self.set_torque_limit(joints, torque_limit)
        self.bus.enable_torque(joints)
        self._torque_on.update(joints)

    def set_goal(self, goals: dict[str, float]) -> None:
        goal = {j: self.clamp(j, v) for j, v in goals.items()}
        self.bus.sync_write("Goal_Position", goal, normalize=False)
        self.goals.update(goal)

    def release(self, joints: list[str]) -> None:
        self.bus.disable_torque(joints)
        self._torque_on.difference_update(joints)
        for j in joints:
            self.goals[j] = None

    def close(self, release: bool = False) -> None:
        """Default: leave torque ON so the arm does not drop. Torque_Limit is restored."""
        try:
            for j, v in self._orig_torque_limit.items():
                self.bus.write("Torque_Limit", j, v, normalize=False)
        finally:
            self.bus.disconnect(disable_torque=release)
        if self._torque_on and not release:
            print(f"[tactile] torque still ON for {sorted(self._torque_on)} (arm holds its pose). "
                  "Support the arm and re-run with --release-on-exit to let it go limp.")


# --------------------------------------------------------------------------- fake arm
class FakeSource(_Base):
    """A crude P-servo simulation, good enough to exercise plots and detectors.

    Each joint: drive = P*(goal-pos), clipped to the torque limit. The external
    torque is gravity + random 'hand pushes' + (gripper only) a squishy object
    that crushes past a force. Overdamped motion: the joint moves until drive
    balances the external torque. Steady state: err = external / P, load = drive.
    """

    P = 2.0           # load units per tick of error
    FOLLOW = 5.0      # ticks/s per unit of unbalanced load -> time constant 1/(P*FOLLOW) = 0.1 s

    def __init__(self, pushes: bool = True, rate_hz: float = 60.0, seed: int = 0):
        try:
            ranges = config.load_ranges()
        except FileNotFoundError:
            ranges = {j: (900, 3100) for j in self.joints}
        super().__init__(ranges)
        self.rng = random.Random(seed)
        self.dt = 1.0 / rate_hz
        self.pos = {j: (lo + hi) / 2 for j, (lo, hi) in ranges.items()}
        self.torque_limit = {j: 1000 for j in self.joints}
        self.torque_on: set[str] = set()
        self.pushes = pushes
        self._push = None          # (joint, magnitude, t_end)
        self._next_push = 5.0      # after the scripts' default 3 s "don't touch" learning
        # Gripper object: contact `obj_gap` ticks before the closed end, stiffness
        # k load/tick, and it "crushes" (stiffness collapses) above `obj_crush` load.
        lo, hi = ranges["gripper"]
        self._closed_end = config.GRIPPER_CLOSED_END or "min"
        self.closed_raw = lo if self._closed_end == "min" else hi
        self.obj_contact = self.closed_raw + (550 if self._closed_end == "min" else -550)
        self.obj_k = 1.6
        self.obj_crush = 380.0
        self.obj_crushed = False
        self._last = time.perf_counter()

    def _gravity(self, j: str) -> float:
        g = {"shoulder_lift": 180, "elbow_flex": 120, "wrist_flex": 40}.get(j, 0)
        return g * math.cos((self.pos[j] - 2048) / 4096 * 2 * math.pi)

    def _object(self) -> float:
        """Force the object pushes back on the gripper (positive = opposes closing)."""
        closing_sign = -1 if self._closed_end == "min" else 1
        pen = (self.pos["gripper"] - self.obj_contact) * closing_sign  # >0 once squeezing
        if pen <= 0:
            return 0.0
        k = self.obj_k * (0.15 if self.obj_crushed else 1.0)
        f = k * pen
        if not self.obj_crushed and f > self.obj_crush:
            self.obj_crushed = True
        return closing_sign * f   # ext > 0 pushes pos down, so this pushes the jaws open

    def read(self) -> Sample:
        # pace like a real bus
        now = time.perf_counter()
        sleep = self.dt - (now - self._last)
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.perf_counter()
        t = self._last - self.t0

        if self.pushes and self._push is None and t > self._next_push:
            j = self.rng.choice(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"])
            self._push = (j, self.rng.choice([-1, 1]) * self.rng.uniform(120, 260), t + self.rng.uniform(0.4, 1.0))
        if self._push and t > self._push[2]:
            self._push = None
            self._next_push = t + self.rng.uniform(2.5, 5.0)

        s = Sample(t=t)
        for j in self.joints:
            ext = self._gravity(j)
            if self._push and self._push[0] == j:
                ext += self._push[1]
            if j == "gripper":
                ext += self._object()
            if j in self.torque_on and self.goals[j] is not None:
                lim = self.torque_limit[j]
                drive = max(-lim, min(lim, self.P * (self.goals[j] - self.pos[j])))
                v = (drive - ext) * self.FOLLOW
                lo, hi = self.ranges[j]
                self.pos[j] = min(max(self.pos[j] + v * self.dt, lo), hi)   # physical stops
                load = drive
            else:
                v, load = 0.0, 0.0
            noise = self.rng.gauss(0, 6)
            s.pos[j] = int(round(self.pos[j]))
            s.load[j] = int(round(load + noise))
            s.cur[j] = int(round(0.55 * abs(load) + 4 + abs(self.rng.gauss(0, 3))))
            s.cur_raw[j] = s.cur[j]
            s.vel[j] = int(round(v))
            s.status[j] = 0
        return self._fill_err(s)

    def safety_report(self):
        return {"Torque_Limit": dict(self.torque_limit), "note": "fake source"}

    def set_torque_limit(self, joints, value):
        for j in joints:
            self.torque_limit[j] = int(min(max(value, 0), 1000))

    def hold_here(self, joints, torque_limit):
        for j in joints:
            self.goals[j] = self.clamp(j, self.pos[j])
            self.torque_limit[j] = torque_limit
            self.torque_on.add(j)

    def set_goal(self, goals):
        for j, v in goals.items():
            self.goals[j] = self.clamp(j, v)

    def release(self, joints):
        for j in joints:
            self.torque_on.discard(j)
            self.goals[j] = None

    def close(self, release=False):
        pass


def open_source(fake: bool, port: str = config.FOLLOWER_PORT, full: bool = True, pushes: bool = True):
    if fake:
        return FakeSource(pushes=pushes)
    import os
    if not os.path.exists(port):
        raise SystemExit(f"Follower not found at {port}\nPlug it in (12 V PSU on), or run with --fake.")
    return ArmSource(port=port, full=full)
