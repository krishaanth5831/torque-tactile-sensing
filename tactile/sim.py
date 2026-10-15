"""A simulated follower that fails the way the real one does, for the touch GUI's --fake
and for scripts/bench_touch.py.

Per joint (servo torque in N.m, +tick direction):

    drive    = K * (goal - pos), K = NM_PER_TICK_P16 * P / 16, 0 inside the dead zone,
               clipped at Torque_Limit
    external = gravity (MuJoCo) + the finger: J(p)^T F at a point p on the real surface
    friction = Coulomb: while |drive + external| <= friction the joint does NOT move.
               This is what the real arm does with a light push, and why a released
               joint can settle a little off where it started.
    motion   = overdamped: speed proportional to the force left after friction
    reading  = round(pos + encoder jitter)    (integer ticks, like Present_Position)
    load     = drive in load units, 0 inside +-LOAD_DEADBAND, in steps of 4 (as measured)

Pushes: a random point on a FINER surface map than the localiser's (different seed),
force 1-5 N within a friction cone around the inward normal, ramped in and out like a
fingertip. `true_push` = (link, world point, world force) while it lasts.
"""
from __future__ import annotations

import time

import numpy as np

from . import config
from .model import SO101Model
from .source import Sample, _Base
from .surface import build_surface

ENCODER_JITTER = 0.45        # ticks rms: measured err std at rest was 0.4-0.6
DEAD_ZONE = 1                # ticks (CW/CCW_Dead_Zone)
LOAD_DEADBAND = 18           # load units: real Present_Load reads 0, then jumps to +-20
TIME_CONSTANT = 0.03         # s, how fast a joint settles after the net force changes


class ServoArmSim:
    def __init__(self, ranges, friction_nm: float = 0.02, p_gain: int = config.P_DEFAULT, seed: int = 0,
                 pose: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.ranges = ranges
        self.model = SO101Model(ranges=ranges)
        self.g = self.model.torque_tick_to_model
        self.J = config.JOINTS
        self.friction = np.full(len(self.J), friction_nm)
        self.p_gain = np.full(len(self.J), float(p_gain))
        self.pos = np.array([float((pose or {}).get(j, (lo + hi) / 2)) for j, (lo, hi) in
                             ((j, ranges[j]) for j in self.J)])
        self.goal = np.full(len(self.J), np.nan)
        self.limit_nm = np.full(len(self.J), 2.94)
        self.push = None          # (body, local point, world force at full strength, t0, t_end)
        self._true = None
        self.truth_surface = None

    # ------------------------------------------------------------------ physics
    @property
    def K(self) -> np.ndarray:
        return config.NM_PER_TICK_P16 * self.p_gain / config.P_DEFAULT

    def drive(self) -> np.ndarray:
        e = np.nan_to_num(self.goal - self.pos)
        e = np.where(np.abs(e) <= DEAD_ZONE, 0.0, e)
        return np.clip(self.K * e, -self.limit_nm, self.limit_nm) * ~np.isnan(self.goal)

    def external(self, t: float) -> np.ndarray:
        m = self.model
        m.set_pose(dict(zip(self.J, self.pos)))
        ext = -m.model_to_tick(m.d.qfrc_bias)               # gravity pulls; holding drive opposes it
        self._true = None
        if self.push is not None:
            body, local, F, t0, t1 = self.push
            ramp = min(1.0, (t - t0) / 0.15, (t1 - t) / 0.15)
            if ramp > 0:
                R = m.d.xmat[body].reshape(3, 3)
                p = m.d.xpos[body] + R @ local
                f = F * ramp
                ext += m.model_to_tick(m.torque_from_force(body, p, f))
                self._true = (self.truth_surface.link_of_body[int(body)], p.copy(), f.copy())
        return ext

    def step(self, t: float, dt: float) -> None:
        on = ~np.isnan(self.goal)
        net = self.drive() + self.external(t)
        excess = np.sign(net) * np.maximum(np.abs(net) - self.friction, 0.0)
        v = excess / (self.K * TIME_CONSTANT)                 # ticks/s
        lo = np.array([self.ranges[j][0] for j in self.J]) - 60
        hi = np.array([self.ranges[j][1] for j in self.J]) + 60
        self.pos = np.where(on, np.clip(self.pos + v * dt, lo, hi), self.pos)
        self._v = v * on

    def measure(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.round(self.pos + self.rng.normal(0, ENCODER_JITTER, len(self.J))).astype(int)
        load = self.drive() / config.NM_PER_LOAD
        load = np.where(np.abs(load) < LOAD_DEADBAND, 0, np.round(load / 4) * 4).astype(int)
        return pos, load

    # ------------------------------------------------------------------ pushes
    def start_push(self, t: float, duration: float, force_n: tuple = (1.0, 5.0), mu: float = 0.5,
                   links: list[str] | None = None) -> None:
        if self.truth_surface is None:
            self.truth_surface = build_surface(self.model.m, spacing_m=0.003, seed=12345)
        s = self.truth_surface
        ok = np.arange(s.n) if links is None else np.flatnonzero(
            [s.link_of_body[int(b)] in links for b in s.body])
        i = int(self.rng.choice(ok))
        self.model.set_pose(dict(zip(self.J, self.pos)))
        R = self.model.d.xmat[s.body[i]].reshape(3, 3)
        n = R @ s.normal[i]
        # random direction within the cone around -n
        a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        phi, tilt = self.rng.uniform(0, 2 * np.pi), np.arctan(mu) * np.sqrt(self.rng.uniform())
        d = -np.cos(tilt) * n + np.sin(tilt) * (np.cos(phi) * a + np.sin(phi) * b)
        self.push = (int(s.body[i]), s.local[i].copy(), d * self.rng.uniform(*force_n), t, t + duration)
        self.push_link = s.link_of_body[int(s.body[i])]

    @property
    def true_push(self):
        return self._true


class SimArmSource(_Base):
    """The touch GUI's --fake: ServoArmSim behind the ArmSource interface, real time."""

    POSE = {"shoulder_pan": 2048, "shoulder_lift": 1500, "elbow_flex": 2500, "wrist_flex": 2300,
            "wrist_roll": 2048, "gripper": 2200}

    def __init__(self, pushes: bool = True, rate_hz: float = 320.0, seed: int = 0, friction_nm: float = 0.02):
        try:
            ranges = config.load_ranges()
        except FileNotFoundError:
            ranges = {j: (900, 3100) for j in self.joints}
        super().__init__(ranges)
        self.sim = ServoArmSim(ranges, friction_nm=friction_nm, seed=seed,
                               pose={j: self.clamp(j, v) for j, v in self.POSE.items()})
        self.rng = np.random.default_rng(seed + 1)
        self.dt = 1.0 / rate_hz
        self.pushes = pushes
        self._next_push = 6.0          # after the GUI's settle + learn
        self._t_sim = 0.0
        self._last = time.perf_counter()

    @property
    def true_push(self):
        return self.sim.true_push

    def read(self) -> Sample:
        now = time.perf_counter()
        sleep = self.dt - (now - self._last)
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.perf_counter()
        t = self._last - self.t0
        while self._t_sim < t:                      # fixed physics step, whatever the read rate
            self.sim.step(self._t_sim, 1 / 1000)
            self._t_sim += 1 / 1000
        if self.pushes and self.sim.push is None and t > self._next_push:
            self.sim.start_push(t, self.rng.uniform(0.8, 2.0))
        if self.sim.push is not None and t > self.sim.push[4]:
            self.sim.push = None
            self._next_push = t + self.rng.uniform(2.0, 4.0)
        pos, load = self.sim.measure()
        s = Sample(t=t)
        for k, j in enumerate(self.joints):
            s.pos[j], s.load[j] = int(pos[k]), int(load[k])
            s.cur[j] = s.cur_raw[j] = int(abs(load[k]) // 150)
            s.vel[j], s.status[j] = 0, 0
        return self._fill_err(s)

    def _sync_goals(self):
        self.sim.goal = np.array([np.nan if self.goals[j] is None else float(self.goals[j]) for j in self.joints])

    def read_status(self):
        return {j: 0 for j in self.joints}

    def safety_report(self):
        return {"P_Coefficient": dict(zip(self.joints, self.sim.p_gain.astype(int).tolist())),
                "note": f"simulated arm, friction {self.sim.friction[0]:.3f} N.m per joint"}

    def set_torque_limit(self, joints, value):
        for j in joints:
            self.sim.limit_nm[self.joints.index(j)] = int(min(max(value, 0), 1000)) * config.NM_PER_LOAD

    def hold_here(self, joints, torque_limit):
        for j in joints:
            self.goals[j] = self.clamp(j, round(self.sim.pos[self.joints.index(j)]))
        self.set_torque_limit(joints, torque_limit)
        self._sync_goals()

    def set_goal(self, goals):
        for j, v in goals.items():
            self.goals[j] = self.clamp(j, v)
        self._sync_goals()

    def set_p_gain(self, joints, p):
        """Softer servo, same pose: re-aim the goal so the holding torque is unchanged."""
        for j in joints:
            k = self.joints.index(j)
            if self.goals[j] is not None:
                e = self.goals[j] - self.sim.pos[k]
                self.goals[j] = self.clamp(j, round(self.sim.pos[k] + e * self.sim.p_gain[k] / p))
            self.sim.p_gain[k] = p
        self._sync_goals()

    def release(self, joints):
        for j in joints:
            self.goals[j] = None
        self._sync_goals()

    def close(self, release=False):
        pass
