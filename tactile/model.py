"""The SO-101 as a MuJoCo model: raw ticks -> joint angles, gravity, Jacobians.

Contact localisation lives in contact.py (a probability over the whole arm surface).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import config

SCENE_XML = Path(__file__).resolve().parent.parent / "assets/so101/scene.xml"
ARM = config.JOINTS[:5]                     # gripper handled separately
TICKS_TO_RAD = 2 * np.pi / 4096


class SO101Model:
    def __init__(self, xml: Path = SCENE_XML, ranges: dict[str, tuple[int, int]] | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.ranges = ranges
        self.jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in config.JOINTS]
        self.dof = [int(self.m.jnt_dofadr[i]) for i in self.jid]
        self.qadr = [int(self.m.jnt_qposadr[i]) for i in self.jid]
        self.body_of_joint = [int(self.m.jnt_bodyid[i]) for i in self.jid]
        self.tip_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._jacp = np.zeros((3, self.m.nv))

    # ------------------------------------------------------------------ mapping
    def raw_to_qpos(self, raw: dict[str, float]) -> np.ndarray:
        q = self.d.qpos.copy()
        for k, j in enumerate(config.JOINTS):
            if j not in raw or raw[j] is None:
                continue
            if j == "gripper":
                q[self.qadr[k]] = self._gripper_angle(raw[j], k)
            else:
                q[self.qadr[k]] = (config.JOINT_SIGN[j] * (raw[j] - 2048 - config.JOINT_OFFSET_TICKS[j])
                                   * TICKS_TO_RAD)
        return q

    def _gripper_angle(self, raw: float, k: int) -> float:
        lo_m, hi_m = self.m.jnt_range[self.jid[k]]
        if self.ranges is None:
            return float(lo_m)
        lo, hi = self.ranges["gripper"]
        s = np.clip((raw - lo) / max(hi - lo, 1), 0, 1)
        if config.GRIPPER_MODEL_FLIP:
            s = 1 - s
        return float(lo_m + s * (hi_m - lo_m))

    def set_pose(self, raw: dict[str, float]) -> None:
        self.d.qpos[:] = self.raw_to_qpos(raw)
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ torques: model axes <-> servo ticks
    def holding_load(self, raw: dict[str, float]) -> dict[str, float]:
        """Drive (load units, tick direction) needed to hold this pose against gravity.
        qfrc_bias is the generalised force the actuators must apply at rest."""
        self.set_pose(raw)
        tau = self.model_to_tick(self.d.qfrc_bias)
        return {j: tau[k] / config.NM_PER_LOAD for k, j in enumerate(config.JOINTS)}

    def model_to_tick(self, tau_model: np.ndarray) -> np.ndarray:
        """Generalised torque (model axes, length nv) -> per config.JOINTS servo torque, N.m
        in the +tick direction."""
        return np.array([tau_model[self.dof[k]] for k in range(len(config.JOINTS))]) / self.torque_tick_to_model

    @property
    def torque_tick_to_model(self) -> np.ndarray:
        """Per config.JOINTS: model-axis torque per unit of servo (+tick direction) torque.
        Work is the same on both sides: tau_model * dq_model = tau_servo * dq_servo, so the
        factor is (servo rad per tick) / (model rad per tick). +-1 for the arm joints; the
        gripper is mapped linearly from its calibrated range, so it differs from 1."""
        out = np.array([float(config.JOINT_SIGN.get(j, 1)) for j in config.JOINTS])
        if self.ranges is not None:
            k = config.JOINTS.index("gripper")
            lo, hi = self.ranges["gripper"]
            lo_m, hi_m = self.m.jnt_range[self.jid[k]]
            model_per_tick = (hi_m - lo_m) / max(hi - lo, 1)
            out[k] = TICKS_TO_RAD / model_per_tick * (-1 if config.GRIPPER_MODEL_FLIP else 1)
        return out

    # ------------------------------------------------------------------ geometry
    def jac(self, point: np.ndarray, body: int) -> np.ndarray:
        mujoco.mj_jac(self.m, self.d, self._jacp, None, point, body)
        return self._jacp.copy()

    def torque_from_force(self, body: int, point: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Generalised torques (model axes) that a force at a point on `body` produces."""
        return self.jac(point, body).T @ force
