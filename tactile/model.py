"""The SO-101 as a MuJoCo model: raw ticks -> joint angles, gravity, and contact localisation.

Where is the push coming from? A force F pressing on the arm at point p produces joint
torques   tau = J(p)^T F   (J = the translational Jacobian of point p). Two facts invert it:

1. ISOLATION. A push on link k only loads joints 0..k (the joints between it and the base).
   Joints further out feel nothing. So the MOST DISTAL joint that feels a torque tells
   you which link was touched.
2. FIT. On that link, try candidate points along it. For each one, solve
   J(p)^T F = tau by least squares. The point whose fit leaves the smallest error is the
   best guess, and F is the force.

Be honest about the limits:
- On the shoulder / upper arm / lower arm only 1-3 joints respond. With 3 unknowns in F
  the fit is underdetermined: every point fits. We then report the link, its midpoint
  and the smallest force that explains the torques (a lower bound, `ambiguous=True`).
  From the wrist on, 4-5 joints respond and the point is actually constrained.
- A force whose line passes through a joint axis gives that joint zero torque. That can
  fool the isolation step into picking a more proximal link.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import config

SCENE_XML = Path(__file__).resolve().parent.parent / "assets/so101/scene.xml"
ARM = config.JOINTS[:5]                     # gripper handled separately
TICKS_TO_RAD = 2 * np.pi / 4096


@dataclass
class Contact:
    link: str                 # body name in the model
    joint_index: int          # most distal joint that felt it
    point: np.ndarray         # world xyz (m) of the best-guess contact point
    force: np.ndarray         # world force vector (N) applied BY the hand ON the arm
    fit_error: float          # |J^T F - tau| / |tau|, 0 = perfect
    ambiguous: bool           # True: point is a guess (link midpoint), force a lower bound


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

    # ------------------------------------------------------------------ torques <-> load units
    def holding_load(self, raw: dict[str, float]) -> dict[str, float]:
        """Drive (load units, tick direction) needed to hold this pose against gravity.
        qfrc_bias is the generalised force the actuators must apply at rest."""
        self.set_pose(raw)
        return {j: self.model_to_tick(j, self.d.qfrc_bias[self.dof[k]]) / config.NM_PER_LOAD
                for k, j in enumerate(config.JOINTS)}

    @staticmethod
    def model_to_tick(joint: str, tau: float) -> float:
        """Torque about the model joint axis -> torque in the servo's +tick direction."""
        return tau * config.JOINT_SIGN.get(joint, 1)

    def tick_to_model(self, tau_tick: dict[str, float]) -> np.ndarray:
        tau = np.zeros(self.m.nv)
        for k, j in enumerate(config.JOINTS):
            tau[self.dof[k]] = tau_tick.get(j, 0.0) * config.JOINT_SIGN.get(j, 1)
        return tau

    # ------------------------------------------------------------------ geometry
    def jac(self, point: np.ndarray, body: int) -> np.ndarray:
        mujoco.mj_jac(self.m, self.d, self._jacp, None, point, body)
        return self._jacp.copy()

    def link_segment(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Start/end (world) of the link moved by joint k: from its joint anchor to the
        next joint's anchor, or to the gripper tip for the wrist_roll link."""
        a = self.d.xanchor[self.jid[k]].copy()
        b = (self.d.site_xpos[self.tip_site].copy() if k == len(ARM) - 1
             else self.d.xanchor[self.jid[k + 1]].copy())
        return a, b

    def torque_from_force(self, body: int, point: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Generalised torques (model axes) that a force at a point on `body` produces."""
        return self.jac(point, body).T @ force

    # ------------------------------------------------------------------ localisation
    def localize(self, tau: np.ndarray, active: np.ndarray, n_candidates: int = 25) -> Contact | None:
        """tau: external generalised torque (model axes, N.m), length nv.
        active: bool per config.JOINTS (detector says this joint is being pushed)."""
        arm_active = [k for k in range(len(ARM)) if active[k]]
        if not arm_active:
            if active[len(ARM)]:
                b = self.body_of_joint[len(ARM)]
                return Contact(self.m.body(b).name, len(ARM), self.d.xpos[b].copy(), np.zeros(3), 0.0, True)
            return None
        k = max(arm_active)
        if k + 1 < 4:
            # Underdetermined: report the link midpoint and the min-norm force.
            body = self.body_of_joint[k]
            cols = [self.dof[i] for i in range(k + 1)]
            a, b = self.link_segment(k)
            p = a + 0.5 * (b - a)
            F, err = self._fit(p, body, cols, tau)
            return Contact(self.m.body(body).name, k, p, F, err, True)
        # Constrained (wrist on): search points on link k AND on the next link out. A push
        # on the next link whose line passes near that joint's axis leaves it almost
        # torque-free, so it was not flagged; the fit with its ~0 torque still finds it.
        best = None
        for kk in (k, k + 1):
            if kk >= len(ARM):
                continue
            body = self.body_of_joint[kk]
            cols = [self.dof[i] for i in range(kk + 1)]
            for p, s in self.candidates(kk, n_candidates):
                F, err = self._fit(p, body, cols, tau)
                score = err + 0.02 * abs(s - 0.5) + (0.02 if kk != k else 0.0)
                if best is None or score < best[0]:
                    best = (score, err, p, F, kk, body)
        _, err, p, F, kk, body = best
        return Contact(self.m.body(body).name, kk, p, F, err, False)

    def _fit(self, p, body, cols, tau):
        t = tau[cols]
        Jt = self.jac(p, body)[:, cols]               # 3 x n_joints
        F, *_ = np.linalg.lstsq(Jt.T, t, rcond=None)  # min-norm F if underdetermined
        return F, float(np.linalg.norm(Jt.T @ F - t) / (np.linalg.norm(t) + 1e-12))

    def candidates(self, k: int, n: int = 25):
        """Candidate contact points on link k: along its centre line, plus the centres of
        its collision geoms (the real surfaces, e.g. the jaws, off the roll axis).
        Yields (point, s) with s = 0..1 position along the link (geoms: 0.5)."""
        a, b = self.link_segment(k)
        for s in np.linspace(0.05, 0.95, n):
            yield a + s * (b - a), s
        body = self.body_of_joint[k]
        for g in range(self.m.body_geomadr[body], self.m.body_geomadr[body] + self.m.body_geomnum[body]):
            if self.m.geom_contype[g] or self.m.geom_conaffinity[g]:      # collision geoms only
                yield self.d.geom_xpos[g].copy(), 0.5
