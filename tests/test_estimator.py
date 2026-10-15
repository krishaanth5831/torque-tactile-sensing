"""Estimator state machine: learn -> watch -> touch, and motion interrupting learning."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.estimator import TouchEstimator  # noqa: E402
from tactile.model import SO101Model  # noqa: E402

J = config.JOINTS
rng = np.random.default_rng(0)
POSE = {j: 2048 for j in J}


def _est():
    return TouchEstimator(SO101Model(ranges={j: (900, 3100) for j in J}))


def _load(extra=None):
    out = {j: 100 + rng.normal(0, 4) for j in J}
    for j, v in (extra or {}).items():
        out[j] += v
    return out


def test_learn_watch_touch():
    e, t = _est(), 0.0
    states = []
    for _ in range(300):
        t += 1 / 60
        states.append(e.update(t, POSE, POSE, _load()).state)
    assert "LEARNING" in states and states[-1] == "WATCHING"
    o = e.update(t + 0.02, POSE, POSE, _load({"wrist_flex": 150}))
    assert o.state == "TOUCH" and o.active[J.index("wrist_flex")]


def test_motion_during_learning_then_pause_does_not_crash():
    # regression: moving mid-LEARNING left a half baseline with no end time
    e, t = _est(), 0.0
    goal = dict(POSE)
    for i in range(400):
        t += 1 / 60
        if 70 < i < 90:                       # move briefly while it is learning
            goal = {j: v + 5 for j, v in goal.items()}
        e.update(t, goal, goal, _load())
    assert e.out.state == "WATCHING"
