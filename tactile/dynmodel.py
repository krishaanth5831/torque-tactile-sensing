"""Expected position error while the arm MOVES (teleop), so touch can be detected during motion.

The touch signal is err = goal - pos (see config.NM_PER_TICK_P16). Per joint j, fitted by
least squares on a no-touch recording:

    err_j ~= a_j * gravity_j(q) + b_j * qd_j + c_j * tanh(qd_j / v0) + d_j * qdd_j + e_j
              (gravity from MuJoCo) (viscous)   (Coulomb friction)     (inertia)   (offset)

IMPORTANT: every input comes from the COMMANDED trajectory (the goals we send:
leader-mapped positions and their derivatives), never from the follower's measured
position or velocity. Those contain your hand's push; a model fed with them would
"predict" the push and the residual would stay flat. The free a_j also absorbs sign/scale errors in the gravity model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import config
from .model import SO101Model

FEATURES = ["gravity", "qd", "coulomb", "qdd", "offset"]
TARGET = "err"     # what the model predicts (older models predicted Present_Load)
V0 = 30.0          # ticks/s: Coulomb friction smoothing


class GoalDerivatives:
    """Streaming, filtered velocity/acceleration of the commanded goals (ticks/s, /s^2).
    The SAME filter runs in fit and live, so features match."""

    def __init__(self, tau: float = 0.06):
        self.tau = tau
        self.prev_t = self.prev_g = None
        self.qd = {j: 0.0 for j in config.JOINTS}
        self.qdd = {j: 0.0 for j in config.JOINTS}

    def update(self, t: float, goal: dict[str, float | None]) -> None:
        if self.prev_t is not None:
            dt = max(t - self.prev_t, 1e-3)
            a = dt / (self.tau + dt)
            for j in config.JOINTS:
                g, pg = goal.get(j), self.prev_g.get(j)
                if g is None or pg is None:
                    continue
                v = (g - pg) / dt
                new_qd = self.qd[j] + a * (v - self.qd[j])
                self.qdd[j] += a * ((new_qd - self.qd[j]) / dt - self.qdd[j])
                self.qd[j] = new_qd
        self.prev_t, self.prev_g = t, dict(goal)

    def speed(self, joints=config.JOINTS[:5]) -> float:
        return max(abs(self.qd[j]) for j in joints)


def features(model: SO101Model, goal: dict[str, float], der: GoalDerivatives) -> dict[str, np.ndarray]:
    g = model.holding_load(goal)
    return {j: np.array([g[j], der.qd[j], np.tanh(der.qd[j] / V0), der.qdd[j], 1.0]) for j in config.JOINTS}


class LoadModel:
    def __init__(self, coef: dict[str, list[float]], sigma: dict[str, float], r2: dict[str, float]):
        self.coef = {j: np.asarray(c, float) for j, c in coef.items()}
        self.sigma = sigma
        self.r2 = r2

    def predict(self, feats: dict[str, np.ndarray]) -> dict[str, float]:
        return {j: float(self.coef[j] @ feats[j]) for j in self.coef}

    # ------------------------------------------------------------------ fit / io
    @classmethod
    def fit(cls, rows: list[dict], model: SO101Model, min_std: float = 0.5) -> "LoadModel":
        """rows: dicts with t, goal{j}, y{j} = err in ticks (one per sample, time order, one recording)."""
        X = {j: [] for j in config.JOINTS}
        Y = {j: [] for j in config.JOINTS}
        der = GoalDerivatives()
        for r in rows:
            if any(r["goal"].get(j) is None for j in config.JOINTS):
                der = GoalDerivatives()          # gap: restart the filter
                continue
            der.update(r["t"], r["goal"])
            f = features(model, r["goal"], der)
            for j in config.JOINTS:
                X[j].append(f[j])
                Y[j].append(r["y"][j])
        coef, sigma, r2 = {}, {}, {}
        for j in config.JOINTS:
            A, y = np.array(X[j]), np.array(Y[j], float)
            if len(y) < 50:
                raise ValueError(f"only {len(y)} usable rows for {j}; record longer")
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            res = y - A @ c
            coef[j] = c.tolist()
            sigma[j] = max(float(res.std()), min_std)
            r2[j] = float(1 - res.var() / max(y.var(), 1e-9))
        return cls(coef, sigma, r2)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"features": FEATURES, "target": TARGET, "coef": {j: list(c) for j, c in self.coef.items()},
                                          "sigma": self.sigma, "r2": self.r2}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LoadModel":
        d = json.loads(Path(path).read_text())
        if d.get("features") != FEATURES or d.get("target") != TARGET:
            raise ValueError(f"{path}: made by an older version (predicts load, not err): re-run fit_model.py")
        return cls(d["coef"], d["sigma"], d["r2"])


def read_log(path: Path) -> list[dict]:
    """A CsvLogger file -> rows with t, goal{j}, y{j} = err (goal - pos, ticks)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            g = {j: (float(r[f"goal_{j}"]) if r[f"goal_{j}"] != "" else None) for j in config.JOINTS}
            y = {j: (g[j] - float(r[f"pos_{j}"]) if g[j] is not None else 0.0) for j in config.JOINTS}
            out.append({"t": float(r["t"]), "goal": g, "y": y})
    return out
