"""One object that turns each Sample into "am I being touched, how hard, and where".

THE SIGNAL is each servo's position error err = goal - pos (see config.NM_PER_TICK_P16):
external torque = -K * (err - expected err). Single readings are integer ticks, so they
are averaged over a short window (WINDOW_S): the encoder's +-0.5 tick jitter then
resolves fractions of a tick.

"Expected err" depends on what the COMMAND is doing:

    commanded still >= still_s, no baseline yet  -> LEARNING  (learns mean/noise, don't touch)
    commanded still, baseline ready              -> WATCHING  (static baseline: most sensitive)
    commanded moving, LoadModel given            -> MOVING    (dynamic model residual)
    commanded moving, no LoadModel               -> MOVING (no model): no detection

DETECTION is one number over all joints, d = sqrt(sum_j (tau_j / sigma_j)^2), with
hysteresis (d_on / d_off). One joint with a big push or three with small ones both count.

LOCALISATION (where) is ContactLocalizer, run by the caller at ~25 Hz on the latest
window (`localize()`), because it costs ~15 ms and the sensing loop runs at ~300 Hz.

REST BAND: after a push a servo settles anywhere within its dead zone + friction band
(config.DEAD_ZONE_TICKS / FRICTION_NM), so only the part of the error beyond that band
counts. While nothing touches, the baseline also follows wherever friction has parked each
joint (ADAPT_S), so a touch is measured as the change from just before it: a joint that an
earlier poke left 2 ticks off no longer looks pushed during the next poke (real 2026-10-15
poke data: wrist_roll offsets made shoulder pokes look like gripper pokes). If a "touch"
still stays perfectly steady for stuck_s (a bigger friction offset than configured), it is
re-learned as the new baseline: a real finger is never that steady.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config
from .contact import Contact, ContactLocalizer
from .detect import Hysteresis
from .dynmodel import GoalDerivatives, LoadModel, features
from .model import SO101Model

J = config.JOINTS
WINDOW_S = 0.10            # averaging window for err
ADAPT_S = 2.0              # while nothing touches, the baseline follows parked offsets this fast
SETTLED_S = 0.6            # a touch is over once every joint has been dead still this long ...
SETTLED_TICKS = 0.35       # ... (window means moved less than this) within 2 rest bands of the baseline
JITTER_CORR = 3.0          # samples per independent jitter value (the encoder noise is not white)


@dataclass
class TouchState:
    state: str = "starting"
    d: float = 0.0                                                            # detection statistic
    z: np.ndarray = field(default_factory=lambda: np.zeros(len(J)))          # per joint, beyond rest band / noise
    tau_nm: np.ndarray = field(default_factory=lambda: np.zeros(len(J)))     # ext torque, +tick dir, ~N.m
    sigma_nm: np.ndarray = field(default_factory=lambda: np.ones(len(J)))    # tau_nm uncertainty (for localising)
    touching: bool = False
    touch_id: int = 0                                                         # +1 on every new touch
    note: str = ""
    speed: float = 0.0


class _ErrBaseline:
    def __init__(self):
        self._buf: list[np.ndarray] = []
        self.mean = self.std = None

    def add(self, x):
        self._buf.append(np.asarray(x, float))

    def freeze(self):
        a = np.stack(self._buf)
        self.mean, self.std = a.mean(0), np.maximum(a.std(0), 0.3)
        self._buf = []

    @property
    def ready(self):
        return self.mean is not None


class TouchEstimator:
    def __init__(self, model: SO101Model, load_model: LoadModel | None = None, p_gain: float = config.P_DEFAULT,
                 d_on: float = 6.0, d_off: float = 3.5, learn_s: float = 1.5, still_s: float = 0.8,
                 still_speed: float = 15.0, stuck_s: float | None = 5.0, localizer: ContactLocalizer | None = None,
                 window_s: float = WINDOW_S, settle_release: bool = True):
        self.model, self.lm = model, load_model
        self.K = config.NM_PER_TICK_P16 * p_gain / config.P_DEFAULT
        self.d_on, self.d_off = d_on, d_off
        self.learn_s, self.still_s, self.still_speed, self.stuck_s = learn_s, still_s, still_speed, stuck_s
        self.window_s = window_s
        self.settle_release = settle_release      # off with --dither: no friction parking to undo
        self._rezeroed = None                     # (t, residual) of the last settled touch
        self.der = GoalDerivatives()
        self.hyst = Hysteresis(1, d_on, d_off)
        self.base: _ErrBaseline | None = None
        self._learn_until = None
        self._still_since = None
        self.win: deque = deque()
        self.touch_id = 0
        self._touch_t0 = None
        self._steady: deque = deque()
        self.loc = localizer
        self._loc_id = -1
        self.pos = None
        self._t_last = None
        self.out = TouchState()

    def relearn(self) -> None:
        self.base = None
        self._learn_until = None
        self._still_since = None   # wait for stillness again
        self.win.clear()

    def set_p_gain(self, p_gain: float) -> None:
        self.K = config.NM_PER_TICK_P16 * p_gain / config.P_DEFAULT
        self.relearn()

    def update(self, t: float, goal: dict, pos: dict, load: dict | None = None) -> TouchState:
        o = TouchState(touch_id=self.touch_id)
        if any(goal.get(j) is None for j in J):
            o.state = "not holding (torque off?)"
            self.out = o
            return o
        self.pos = dict(pos)
        err = np.array([float(goal[j]) - float(pos[j]) for j in J])
        t_last, self._t_last = self._t_last, t
        self.der.update(t, goal)
        o.speed = self.der.speed()
        still = o.speed < self.still_speed
        if still:
            self._still_since = t if self._still_since is None else self._still_since
        else:
            self._still_since = None
            self._learn_until = None
            self.base = None      # new pose after moving: old (or half-learned) baseline invalid

        expected = sig_err = None
        model_exp = model_sig = None
        if self.lm is not None:
            p = self.lm.predict(features(self.model, goal, self.der))
            model_exp = np.array([p[j] for j in J])
            model_sig = np.array([self.lm.sigma[j] for j in J])
        n_win = max(len(self.win), 1)
        if still and self._still_since is not None and t - self._still_since >= self.still_s:
            if self.base is None:
                self.base = _ErrBaseline()
                self._learn_until = t + self.learn_s
            if not self.base.ready:
                o.state = "LEARNING"
                # With a model we are never blind: keep detecting with it while learning,
                # and restart learning if it sees a touch (don't learn a push as "normal").
                if model_exp is not None:
                    expected, sig_err = model_exp, model_sig
                    if (np.abs(err - model_exp) / model_sig > self.d_on).any():
                        self.base = _ErrBaseline()
                        self._learn_until = t + self.learn_s
                    else:
                        self.base.add(err)
                else:
                    self.base.add(err)
                if self.base._buf and t >= self._learn_until:
                    self.base.freeze()
                    self.win.clear()
            else:
                expected = self.base.mean.copy()
                sig_err = self.base.std * np.sqrt(min(JITTER_CORR / n_win, 1.0))   # window-mean noise
                o.state = "WATCHING"
        elif model_exp is not None:
            expected, sig_err = model_exp, model_sig       # model error is systematic: no averaging gain
            o.state = "MOVING" if not still else "settling (model)"
        else:
            o.state = "MOVING (no model: run fit_model.py)" if not still else "settling"

        if expected is None:
            self.win.clear()
            self.hyst.update(np.zeros(1))
            self._end_touch()
            self.out = o
            return o

        self.win.append((t, err - expected))
        while self.win and t - self.win[0][0] > self.window_s:
            self.win.popleft()
        r = np.mean([w[1] for w in self.win], axis=0)
        band = max(config.DEAD_ZONE_TICKS, config.FRICTION_NM / self.K)
        noise = np.sqrt((self.K * sig_err) ** 2 + config.TAU_FLOOR_NM ** 2)
        # detection: only what exceeds the rest band (see config) counts
        o.z = -self.K * np.sign(r) * np.maximum(np.abs(r) - band, 0.0) / noise
        # localisation: the full torque, with the band as extra (uniform) uncertainty, so
        # the ratios between joints are not distorted
        o.tau_nm = -self.K * r
        o.sigma_nm = np.sqrt(noise ** 2 + (self.K * band) ** 2 / 3)
        o.d = float(np.sqrt(np.sum(o.z ** 2)))
        if (o.state == "WATCHING" and self._touch_t0 is None and t_last is not None
                and o.d < self.d_off):
            # Quiet: follow wherever friction has parked each joint (ADAPT_S), so the next
            # touch is measured from just before it. Only while quiet and within 2 rest bands:
            # adapting during the ramp of a push (before it is detected) leaked 15 % of big
            # pushes into the baseline and left them "touching" after release.
            a = min((t - t_last) / ADAPT_S, 1.0)
            ok = np.abs(r) <= 2 * band
            self.base.mean[ok] += a * (err[ok] - self.base.mean[ok])
        on = bool(self.hyst.update(np.array([o.d]))[0])
        if on and self._touch_t0 is None and self._rezeroed is not None and t - self._rezeroed[0] < 5.0:
            # The last touch was taken into the baseline while the finger was still on. When the
            # finger lifts, the joints spring back by about minus that touch: not a new touch.
            prev = self._rezeroed[1]
            cos = float(-r @ prev / (np.linalg.norm(r) * np.linalg.norm(prev) + 1e-9))
            if cos > 0.7 and np.linalg.norm(r) < 1.5 * np.linalg.norm(prev) + 1.0:
                if self.base is not None and self.base.ready:
                    self.base.mean = self.base.mean + r
                self.win.clear()
                self.hyst = Hysteresis(1, self.d_on, self.d_off)
                self._rezeroed = None
                o.state, o.note = "WATCHING", "finger lifted (springing back is not a touch)"
                self.out = o
                return o
        if on:
            if self._touch_t0 is None:
                self._touch_t0 = t
                self.touch_id += 1
                self._steady.clear()
            o.touching, o.state, o.touch_id = True, "TOUCH", self.touch_id
            self._steady.append((t, r))
            while self._steady and t - self._steady[0][0] > 2.0:
                self._steady.popleft()
            recent = np.array([s[1] for s in self._steady if t - s[0] <= SETTLED_S])
            settled = (t - self._touch_t0 > SETTLED_S and len(recent) > 2 and
                       np.ptp(recent, axis=0).max() < SETTLED_TICKS)
            if self.settle_release and settled and (np.abs(r) <= 2 * band + 0.5).all():
                # Every joint has stopped dead, close to where it was: the finger has gone and
                # friction parks each joint up to a band either side of a baseline that was
                # itself parked (so 2 bands). Take the parked pose as the new baseline.
                if self.base is not None and self.base.ready:
                    self.base.mean = self.base.mean + r
                self._rezeroed = (t, r.copy())
                self.win.clear()
                self.hyst = Hysteresis(1, self.d_on, self.d_off)
                self._end_touch()
                o.touching, o.state = False, "WATCHING"
                o.note = "touch over: joints settled (friction parked them)"
            elif (self.stuck_s and t - self._touch_t0 > self.stuck_s and
                    np.ptp(np.array([s[1] for s in self._steady]), axis=0).max() < 0.5):
                # a finger is never this steady: friction holds the joint off its baseline
                self.relearn()
                self._end_touch()
                o.touching = False
                o.note = "re-learned: steady offset after a push (friction)"
        else:
            self._end_touch()
        self.out = o
        return o

    def _end_touch(self):
        self._touch_t0 = None

    def localize(self, stop_aware: bool = True) -> Contact | None:
        """Run the contact localiser on the latest window (call at ~25 Hz, any thread that
        owns `self.loc`). Accumulates evidence while the same touch lasts."""
        o = self.out
        if self.loc is None or not o.touching or self.pos is None:
            return None
        if o.touch_id != self._loc_id:
            self.loc.reset()
            self._loc_id = o.touch_id
        return self.loc.update(self.pos, o.tau_nm, o.sigma_nm,
                               self.stop_dir(self.pos) if stop_aware else None)

    def stop_dir(self, pos: dict) -> np.ndarray:
        """Per joint: +1 resting on the stop at the top of its range, -1 at the bottom, else 0."""
        r = self.model.ranges
        if r is None:
            return np.zeros(len(J))
        m = config.STOP_MARGIN_TICKS
        return np.array([1.0 if pos[j] > r[j][1] - m else -1.0 if pos[j] < r[j][0] + m else 0.0 for j in J])

    def on_stops(self, pos: dict) -> np.ndarray:
        """Per joint: resting at (or near) a calibrated range end, i.e. probably on its stop."""
        r = self.model.ranges
        if r is None:
            return np.zeros(len(J), bool)
        m = config.STOP_MARGIN_TICKS
        return np.array([pos[j] < r[j][0] + m or pos[j] > r[j][1] - m for j in J])
