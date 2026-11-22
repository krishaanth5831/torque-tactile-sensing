"""Play a recorded CSV back as if it were the arm: the touch GUI on old data, no hardware.

    src = ReplaySource("data/poke_<stamp>.csv", speed=1.0)
    s = src.read()        # the next recorded Sample, paced like the original run

Goals come from the logged nominal goal (`nom_*`, the unshaken goal of a --dither run) when
present, else from `goal_*`. Commands (hold, set_goal, P gain) are accepted and ignored: a
replay can't move anything. `target` is the poke target logged in that row (tgt_* columns,
--poke runs from 2026-10-15 on), so the GUI can draw it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .source import Sample, _Base


class ReplayEnded(Exception):
    pass


class ReplaySource(_Base):
    def __init__(self, path: str, speed: float = 1.0):
        super().__init__(config.load_ranges())
        d = pd.read_csv(path, low_memory=False)
        J = self.joints
        gcols = [f"nom_{j}" for j in J]
        if not (all(c in d.columns for c in gcols) and d[gcols].notna().all(axis=None)):
            gcols = [f"goal_{j}" for j in J]
        self.path, self.speed = path, speed
        self.t = d.t.to_numpy(float)
        self.pos = d[[f"pos_{j}" for j in J]].to_numpy()
        self.goal = d[gcols].to_numpy(float)
        self.load = d[[f"load_{j}" for j in J]].to_numpy()
        self.p_gain = int(d.p_gain.dropna().iloc[-1]) if "p_gain" in d and d.p_gain.notna().any() else None
        has_tgt = "tgt_body" in d.columns
        self.tgt = (d.tgt_body.to_numpy(object), d[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(float)) if has_tgt else None
        self.i = 0
        self.target = None
        self._wall0 = None

    def read(self) -> Sample:
        if self.i >= len(self.t):
            raise ReplayEnded(f"end of {self.path}")
        if self._wall0 is None:
            self._wall0 = time.perf_counter() - self.t[0] / self.speed
        wait = self._wall0 + self.t[self.i] / self.speed - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        i = self.i
        self.i += 1
        s = Sample(t=float(self.t[i]))
        for k, j in enumerate(self.joints):
            s.pos[j] = int(self.pos[i, k])
            s.load[j] = int(self.load[i, k])
            s.vel[j], s.status[j] = 0, 0
            g = self.goal[i, k]
            self.goals[j] = None if np.isnan(g) else float(g)
        if self.tgt is not None:
            b = self.tgt[0][i]
            self.target = None if not isinstance(b, str) else (b, self.tgt[1][i])
        return self._fill_err(s)

    # ---- a replay accepts every command and changes nothing
    def read_status(self):
        return {}

    def safety_report(self):
        return {"note": f"replay of {self.path} at {self.speed:g}x" +
                (f", recorded at P={self.p_gain}" if self.p_gain else "")}

    def hold_here(self, joints, torque_limit):
        pass

    def set_goal(self, goals):
        pass

    def set_torque_limit(self, joints, value):
        pass

    def set_p_gain(self, joints, p, steps=4, reaim=True):
        pass

    def release(self, joints):
        pass

    def close(self, release=False):
        pass
