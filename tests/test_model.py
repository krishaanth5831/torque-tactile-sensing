"""Contact localisation on the MuJoCo SO-101 with synthetic pushes: tau = J(p)^T F."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.model import ARM, SO101Model  # noqa: E402

rng = np.random.default_rng(0)
POSE = {"shoulder_pan": 2300, "shoulder_lift": 1700, "elbow_flex": 2500, "wrist_flex": 2200,
        "wrist_roll": 2048, "gripper": 2300}


def _model():
    m = SO101Model(ranges={j: (900, 3100) for j in config.JOINTS})
    m.set_pose(POSE)
    return m


def _push(m, k, s, F):
    a, b = m.link_segment(k)
    p = a + s * (b - a)
    return p, m.torque_from_force(m.body_of_joint[k], p, F)


def _active(m, tau, thr=1e-3):
    return np.array([abs(tau[m.dof[i]]) > thr for i in range(len(config.JOINTS))])


def test_identifies_the_pushed_link():
    m = _model()
    hits = total = 0
    for k in range(len(ARM)):
        for _ in range(20):
            F = rng.normal(size=3)
            F *= 2.0 / np.linalg.norm(F)
            _, tau = _push(m, k, rng.uniform(0.2, 0.8), F)
            c = m.localize(tau, _active(m, tau))
            total += 1
            hits += c is not None and c.joint_index == k
    assert hits / total > 0.9          # misses = forces whose line passes near a joint axis


def test_wrist_force_direction_recovered():
    m = _model()
    cos = []
    for k in (3, 4):                   # wrist_flex / wrist_roll links: point is constrained
        for _ in range(30):
            F = rng.normal(size=3)
            F *= 2.0 / np.linalg.norm(F)
            _, tau = _push(m, k, rng.uniform(0.2, 0.8), F)
            c = m.localize(tau, _active(m, tau))
            if c is not None and c.joint_index == k:
                cos.append(F @ c.force / (np.linalg.norm(F) * np.linalg.norm(c.force) + 1e-12))
    assert np.median(cos) > 0.9


def test_proximal_links_flagged_ambiguous_and_explain_torque():
    m = _model()
    _, tau = _push(m, 1, 0.5, np.array([0.0, 0.0, -2.0]))
    c = m.localize(tau, _active(m, tau))
    assert c.ambiguous and c.joint_index == 1
    assert c.fit_error < 1e-6          # the min-norm force still explains the torques exactly


def test_holding_load_matches_gravity_sign():
    m = _model()
    hold = m.holding_load(POSE)
    assert abs(hold["shoulder_lift"]) > abs(hold["wrist_roll"])   # big lever vs ~none
