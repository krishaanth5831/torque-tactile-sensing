"""MuJoCo SO-101: raw ticks <-> model, gravity, torque conversion."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.model import SO101Model  # noqa: E402

POSE = {"shoulder_pan": 2300, "shoulder_lift": 1700, "elbow_flex": 2500, "wrist_flex": 2200,
        "wrist_roll": 2048, "gripper": 2300}


def _model():
    m = SO101Model(ranges={j: (900, 3100) for j in config.JOINTS})
    m.set_pose(POSE)
    return m


def test_holding_load_matches_gravity_sign():
    m = _model()
    hold = m.holding_load(POSE)
    assert abs(hold["shoulder_lift"]) > abs(hold["wrist_roll"])   # big lever vs ~none


def test_torque_conversion_roundtrip_and_work():
    m = _model()
    g = m.torque_tick_to_model
    assert np.allclose(np.abs(g[:5]), 1.0)                         # arm joints are 1:1
    tau = np.arange(1, m.m.nv + 1, dtype=float)
    back = m.model_to_tick(tau)
    assert np.allclose(back * g, [tau[m.dof[k]] for k in range(len(config.JOINTS))])
