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
    shift = {"elbow_flex": 5.0}                 # held 5 ticks off (beyond the ~1.9-tick rest band), forever
    seen_touch = False
    for _ in range(300 * 8):
        t += 1 / 300
        o = e.update(t, GOAL, {j: int(round(GOAL[j] - 3 + shift.get(j, 0))) for j in J})
        seen_touch |= o.touching
    assert seen_touch and not o.touching


def test_parked_offset_inside_the_band_is_not_a_touch_and_is_absorbed():
    """Friction parks a joint 1.5 ticks off after a push: not a touch, and the baseline
    follows it, so the next push is measured from there (real-data fix, 2026-10-15)."""
    e, t = _est(), 0.0
    for _ in range(900):
        t += 1 / 300
        e.update(t, GOAL, _pos(GOAL))
    shift = {"wrist_roll": -1.5}
    for _ in range(300 * 4):
        t += 1 / 300
        o = e.update(t, GOAL, _pos(GOAL, shift))
        assert not o.touching
    k = J.index("wrist_roll")
    for i in range(60):                          # now a push on wrist_flex only
        o = e.update(t + i / 300, GOAL, _pos(GOAL, {**shift, "wrist_flex": 6.0}))
    assert o.touching
    assert abs(o.tau_nm[k]) < abs(o.tau_nm[J.index("wrist_flex")]) / 3    # roll's parked offset is gone



def test_joints_on_their_end_stops_are_flagged():
    e = _est()
    pos = {j: 2048 for j in J}
    pos["elbow_flex"] = 3100 - 5          # range 900..3100 in _est(): resting on the top stop
    pos["gripper"] = 900 + 3              # closed on its stop
    flags = dict(zip(J, e.on_stops(pos)))
    assert flags["elbow_flex"] and flags["gripper"] and not flags["wrist_flex"]


def test_big_push_does_not_leak_into_the_baseline():
    """A large push ramping in (before it is detected) must not be absorbed by the adaptive
    baseline, or the touch outlives the finger (seen with --dither in the simulation)."""
    e, t = _est(settle_release=False), 0.0
    for _ in range(900):
        t += 1 / 300
        e.update(t, GOAL, _pos(GOAL))
    for i in range(45):                                   # 0.15 s ramp to 25 ticks
        t += 1 / 300
        e.update(t, GOAL, _pos(GOAL, {"shoulder_lift": 25 * (i + 1) / 45, "elbow_flex": -30 * (i + 1) / 45}))
    for _ in range(300):                                  # hold 1 s
        t += 1 / 300
        o = e.update(t, GOAL, _pos(GOAL, {"shoulder_lift": 25, "elbow_flex": -30}))
    assert o.touching
    for _ in range(150):                                  # release; 0.5 s later it must be over
        t += 1 / 300
        o = e.update(t, GOAL, _pos(GOAL))
    assert not o.touching
