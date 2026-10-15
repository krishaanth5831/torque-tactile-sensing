"""Estimator state machine on the err signal: learn -> watch -> touch, motion interrupting
learning, and friction leaving a steady offset that must not stay a touch forever."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.estimator import TouchEstimator  # noqa: E402
from tactile.model import SO101Model  # noqa: E402

J = config.JOINTS
rng = np.random.default_rng(0)
GOAL = {j: 2048 for j in J}


def _est(**kw):
    return TouchEstimator(SO101Model(ranges={j: (900, 3100) for j in J}), **kw)


def _pos(goal, shift=None):
    """Integer ticks with jitter around a 3-tick gravity sag; `shift` = extra deflection."""
    out = {}
    for j in J:
        out[j] = int(round(goal[j] - 3 + rng.normal(0, 0.45) + (shift or {}).get(j, 0.0)))
    return out


def test_learn_watch_touch():
    e, t = _est(), 0.0
    states = []
    for _ in range(900):
        t += 1 / 300
        states.append(e.update(t, GOAL, _pos(GOAL)).state)
    assert "LEARNING" in states and states[-1] == "WATCHING"
    assert not any(e.update(t + i / 300, GOAL, _pos(GOAL)).touching for i in range(1, 300))  # no false touch
    t += 1
    for i in range(60):          # 4 ticks of push on wrist_flex (> the 1.4-tick rest band at P=16)
        o = e.update(t + i / 300, GOAL, _pos(GOAL, {"wrist_flex": 4.0}))
    assert o.touching and o.z[J.index("wrist_flex")] > 0     # pushed toward +ticks -> +torque


def test_motion_during_learning_then_pause_does_not_crash():
    # regression: moving mid-LEARNING left a half baseline with no end time
    e, t = _est(), 0.0
    goal = dict(GOAL)
    for i in range(1500):
        t += 1 / 300
        if 300 < i < 380:                       # move briefly while it is learning
            goal = {j: v + 1 for j, v in goal.items()}
        e.update(t, goal, _pos(goal))
    assert e.out.state == "WATCHING"


def test_steady_offset_after_push_is_relearned():
    e, t = _est(stuck_s=2.0), 0.0
    for _ in range(900):
        t += 1 / 300
        e.update(t, GOAL, _pos(GOAL))
    shift = {"elbow_flex": 3.0}                 # friction holds elbow 3 ticks off, forever
    seen_touch = False
    for _ in range(300 * 8):
        t += 1 / 300
        o = e.update(t, GOAL, {j: int(round(GOAL[j] - 3 + shift.get(j, 0))) for j in J})
        seen_touch |= o.touching
    assert seen_touch and not o.touching
