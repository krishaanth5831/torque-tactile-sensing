"""One object that turns each Sample into "where am I being touched, how hard".

Which "expected load" it compares against depends on what the COMMAND is doing:

    commanded still >= still_s, no baseline yet  -> LEARNING  (learns mean/noise, don't touch)
    commanded still, baseline ready              -> WATCHING  (static baseline: most sensitive)
    commanded moving, LoadModel given            -> MOVING    (dynamic model residual)
    commanded moving, no LoadModel               -> MOVING (no model): no detection

So in hold mode it learns once and watches. In teleop it watches while moving (with a
model) and re-learns a sharper static baseline every time you pause the leader.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .detect import Baseline, Hysteresis
from .dynmodel import GoalDerivatives, LoadModel, features
from .model import Contact, SO101Model

J = config.JOINTS


@dataclass
class TouchState:
    state: str = "starting"
    residual: np.ndarray = field(default_factory=lambda: np.zeros(len(J)))   # load units
    z: np.ndarray = field(default_factory=lambda: np.zeros(len(J)))
    active: np.ndarray = field(default_factory=lambda: np.zeros(len(J), bool))
    tau_nm: np.ndarray = field(default_factory=lambda: np.zeros(len(J)))     # ext torque, +tick dir, ~N.m
    contact: Contact | None = None
    speed: float = 0.0


class TouchEstimator:
    def __init__(self, model: SO101Model, load_model: LoadModel | None = None, z_on: float = 6.0,
                 z_off: float = 3.0, learn_s: float = 1.5, still_s: float = 0.8, still_speed: float = 15.0,
                 z_loc: float = 2.0):   # fake-arm sweep: 2.0 best; every miss is one link too proximal
        self.model, self.lm = model, load_model
        self.z_on, self.z_off, self.z_loc = z_on, z_off, z_loc
        self.learn_s, self.still_s, self.still_speed = learn_s, still_s, still_speed
        self.der = GoalDerivatives()
        self.hyst = Hysteresis(len(J), z_on, z_off)
        self.base: Baseline | None = None
        self._learn_until = None
        self._still_since = None
        self.out = TouchState()

    def relearn(self) -> None:
        self.base = None
        self._learn_until = None
        self._still_since = None   # wait for stillness again

    def update(self, t: float, goal: dict[str, float | None], pos: dict[str, int], load: dict[str, int]) -> TouchState:
        o = TouchState()
        x = np.array([float(load[j]) for j in J])
        if any(goal.get(j) is None for j in J):
            o.state = "not holding (torque off?)"
            self.out = o
            return o
        self.der.update(t, goal)
        o.speed = self.der.speed()
        still = o.speed < self.still_speed
        if still:
            self._still_since = t if self._still_since is None else self._still_since
        else:
            self._still_since = None
            self._learn_until = None
            self.base = None      # new pose after moving: old (or half-learned) baseline invalid

        expected, sigma = None, None
        model_exp = model_sig = None
        if self.lm is not None:
            p = self.lm.predict(features(self.model, goal, self.der))
            model_exp = np.array([p[j] for j in J])
            model_sig = np.array([self.lm.sigma[j] for j in J])
        if still and self._still_since is not None and t - self._still_since >= self.still_s:
            if self.base is None:
                self.base = Baseline()
                self._learn_until = t + self.learn_s
            if not self.base.ready:
                o.state = "LEARNING"
                # With a model we are never blind: keep detecting with it while learning,
                # and restart learning if it sees a touch (don't learn a push as "normal").
                if model_exp is not None:
                    expected, sigma = model_exp, model_sig
                    if (np.abs((x - model_exp) / model_sig) > self.z_on).any():
                        self.base = Baseline()
                        self._learn_until = t + self.learn_s
                    else:
                        self.base.add(x)
                else:
                    self.base.add(x)
                if self.base._buf and t >= self._learn_until:
                    self.base.freeze()
            else:
                expected, sigma = self.base.mean, self.base.std
                o.state = "WATCHING"
        elif model_exp is not None:
            expected, sigma = model_exp, model_sig
            o.state = "MOVING" if not still else "settling (model)"
        else:
            o.state = "MOVING (no model: run fit_model.py)" if not still else "settling"

        if expected is not None:
            o.residual = x - expected
            o.z = o.residual / sigma
            o.active = self.hyst.update(o.z)
            o.tau_nm = config.FORCE_SIGN * o.residual * config.NM_PER_LOAD
            if o.active.any():
                o.state = "TOUCH"
                self.model.set_pose(pos)
                tau_model = self.model.tick_to_model({j: o.tau_nm[k] for k, j in enumerate(J)})
                # Detection needs |z| > z_on somewhere, but WHICH link is decided by the most
                # distal joint that clearly left its baseline (|z| > z_loc): a push far out
                # often loads the distal joint only weakly.
                o.contact = self.model.localize(tau_model, o.active | (np.abs(o.z) > self.z_loc))
        else:
            self.hyst.update(np.zeros(len(J)))
        self.out = o
        return o
