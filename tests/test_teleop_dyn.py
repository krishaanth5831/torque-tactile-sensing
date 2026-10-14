"""Leader->follower mapping, step limiter, and the dynamic load model on fake-arm data."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.dynmodel import LoadModel  # noqa: E402
from tactile.model import SO101Model  # noqa: E402
from tactile.teleop import Teleop, map_leader_to_follower  # noqa: E402


def test_mapping_is_linear_by_range_and_clamped():
    lr = {"a": (1000, 3000)}
    fr = {"a": (500, 1500)}
    assert map_leader_to_follower({"a": 2000}, lr, fr)["a"] == 1000
    assert map_leader_to_follower({"a": 9999}, lr, fr)["a"] == 1500
    assert map_leader_to_follower({"a": 0}, lr, fr)["a"] == 500


class _StubLeader:
    ranges = {j: (0, 4000) for j in config.JOINTS}

    def read(self):
        return {j: 4000 for j in config.JOINTS}


def test_teleop_ramps_and_limits_step():
    fr = {j: (0, 4000) for j in config.JOINTS}
    tp = Teleop(_StubLeader(), fr, max_step=40, ramp_s=2.0)
    start = {j: 1000 for j in config.JOINTS}
    g0 = tp.goals(0.0, start)
    assert all(v == 1000 for v in g0.values())          # first tick: no jump
    g1 = tp.goals(0.02, start)
    assert all(abs(g1[j] - g0[j]) <= 40 for j in config.JOINTS)


def test_load_model_fits_fake_gravity_and_motion():
    """Synthetic: load = holding drive (MuJoCo gravity) + viscous + noise; the model
    built only from commanded goals must explain it well."""
    m = SO101Model(ranges={j: (900, 3100) for j in config.JOINTS})
    rng = np.random.default_rng(0)
    rows, t = [], 0.0
    for i in range(1500):
        t += 1 / 60
        goal = {j: 2048 + 400 * np.sin(0.4 * t * (k + 1) + k) for k, j in enumerate(config.JOINTS)}
        hold = m.holding_load(goal)
        load = {j: hold[j] + rng.normal(0, 5) for j in config.JOINTS}
        rows.append({"t": t, "goal": goal, "load": load})
    lm = LoadModel.fit(rows, m)
    for j in ["shoulder_lift", "elbow_flex"]:
        assert lm.r2[j] > 0.95, (j, lm.r2[j])
        assert lm.sigma[j] < 8
