"""WHERE is the arm being touched: a probability over every point of its surface.

A fingertip pressing at point p with force F produces joint torques  tau = J(p)^T F.
We measure tau (6 joints, with noise sigma) and ask, for EVERY candidate surface point
(surface.py, ~4600 points every 5 mm): what is the best force a finger could apply
there, and how well does it explain the measurement?

    min_F  sum_j ((J(p)^T F - tau)_j / sigma_j)^2        (+ a tiny |F|^2 regulariser)
    s.t.   F pushes INTO the surface, within a friction cone around the inward normal

and then weights each point by how plausible that force is for a finger (log-normal around
config.FINGER_FORCE_N): a point that needs 0.2 N or 40 N is less likely than one that needs 3 N.

The constraint is what makes this sharp. Without it, every point on the force's line of
action explains tau equally well (where it enters the arm AND where it leaves). A
finger can only push, so only the entry point survives. All six joints count, including
the ones that read ~0: a push on the forearm MUST leave the wrist joints at zero, so
zero-reading joints rule out every point further out.

The fit is a tiny convex problem per point (3 unknowns), solved for all points at once
with accelerated projected gradient (FISTA) in numpy: ~10-20 ms per update.

Score -> probability: p(point) ~ exp(-chi^2 / 2). While one touch lasts, the scores of
successive updates are summed (tempered, because successive windows share noise), so
the estimate sharpens as long as the finger stays, and sharpens fastest when the push
direction or the pose changes a little.

What physics allows (no algorithm can beat it):
  shoulder    1 joint (pan) responds: the link is known, the point is not
  upper arm   2 joints: a curve of points fits; the heat map shows it
  forearm+    3+ joints and the zero-reading distal joints: a point
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config
from .model import SO101Model
from .surface import SurfaceMap, build_surface


@dataclass
class Contact:
    link: str                 # link name (body, or the link a welded body belongs to)
    point: np.ndarray         # world xyz (m), most probable contact point
    normal: np.ndarray        # world outward normal there
    force: np.ndarray         # world force (N) the finger applies ON the arm
    spread_cm: float          # posterior spread around the point (1-sigma-ish), cm
    link_prob: dict           # link -> probability
    chi2: float               # fit quality of the best point (~n_joints when it fits)
    heat_idx: np.ndarray      # indices of the most probable points (for drawing)
    heat_w: np.ndarray        # their probabilities, normalised to max 1
    heat_pts: np.ndarray      # their world points


def _cone_project(F: np.ndarray, u: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    """Project each F (N,3) onto the cone {angle(F, u) <= theta} (u unit, N,3)."""
    s = np.einsum("ni,ni->n", F, u)
    t = F - s[:, None] * u
    r = np.linalg.norm(t, axis=1)
    tan_t = sin_t / cos_t
    inside = r <= tan_t * s
    polar = tan_t * r <= -s                      # pointing away: nearest cone point is 0
    that = t / np.maximum(r, 1e-12)[:, None]
    a = s * cos_t + r * sin_t                     # component along the cone's edge ray
    edge = a[:, None] * (cos_t * u + sin_t * that)
    out = np.where(inside[:, None], F, edge)
    out[polar & ~inside] = 0.0
    return out


class ContactLocalizer:
    def __init__(self, model: SO101Model, surface: SurfaceMap | None = None, mu: float = config.FINGER_MU,
                 finger_force_n: float | None = config.FINGER_FORCE_N, temper: float = 0.3, iters: int = 60,
                 n_heat: int = 120, force_reg_n: float = config.FORCE_REG_N):
        self.model = model
        self.s = surface if surface is not None else build_surface(model.m)
        theta = np.arctan(mu)
        self.cos_t, self.sin_t = float(np.cos(theta)), float(np.sin(theta))
        self.lam = 1.0 / force_reg_n ** 2
        self.finger_force_n = finger_force_n          # None: no force-size prior
        self.temper, self.iters, self.n_heat = temper, iters, n_heat
        self.links = sorted(set(self.s.link_of_body.values()), key=list(self.s.link_of_body.values()).index)
        self._link_idx = np.array([self.links.index(self.s.link_of_body[int(b)]) for b in self.s.body])
        nv = model.m.nv
        self._jacp = np.zeros((3, nv))
        self._jacr = np.zeros((3, nv))
        self.reset()

    def reset(self) -> None:
        """Call when a touch ends: the next touch is a new point."""
        self.logp = np.zeros(self.s.n)
        self.updates = 0

    # ------------------------------------------------------------------ core
    def point_jacobians(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """At the model's current pose: (N,3,nv) translational Jacobian of every surface
        point, plus world points and normals. J_p = J_v(body) - [p - x_body]x J_w(body)."""
        m, d = self.model.m, self.model.d
        Jv = np.zeros((m.nbody, 3, m.nv))
        Jw = np.zeros((m.nbody, 3, m.nv))
        for b in self.s.bodies:
            mujoco.mj_jacBody(m, d, self._jacp, self._jacr, int(b))
            Jv[b], Jw[b] = self._jacp, self._jacr
        P, Nrm = self.s.world(d)
        r = P - d.xpos[self.s.body]
        Jw_p = Jw[self.s.body]                                   # (N,3,nv)
        cross = np.cross(r[:, :, None], Jw_p, axis=1)            # r x (each column)
        return Jv[self.s.body] - cross, P, Nrm

    def update(self, raw_pos: dict, tau_tick: np.ndarray, sigma_tick: np.ndarray,
               stop_dir: np.ndarray | None = None) -> Contact:
        """raw_pos: measured raw ticks. tau_tick / sigma_tick: external torque and its noise
        per config.JOINTS, N.m in each servo's +tick direction.
        stop_dir: per joint, +1 / -1 if it rests on the stop at the top / bottom of its range,
        else 0. Such a joint can't feel a push INTO its stop (the stop takes it), but a push
        that would lift it OFF the stop (beyond the preload holding it there) would have
        moved it: one-sided evidence instead of a reading."""
        self.model.set_pose(raw_pos)
        Jp, P, Nrm = self.point_jacobians()
        g_tick = self.model.torque_tick_to_model                  # (6,) model torque per tick torque
        stop_dir = np.zeros(len(config.JOINTS)) if stop_dir is None else np.asarray(stop_dir, float)
        tau = np.zeros(self.model.m.nv)
        sig = np.ones(self.model.m.nv)
        for k in range(len(config.JOINTS)):
            tau[self.model.dof[k]] = 0.0 if stop_dir[k] else tau_tick[k] * g_tick[k]
            sig[self.model.dof[k]] = (10.0 if stop_dir[k] else sigma_tick[k]) * abs(g_tick[k])
        A = np.transpose(Jp, (0, 2, 1)) / sig[None, :, None]       # (N,nv,3)
        b = tau / sig
        H = np.einsum("nki,nkj->nij", A, A) + self.lam * np.eye(3)
        g = np.einsum("nki,k->ni", A, b)
        L = np.trace(H, axis1=1, axis2=2)[:, None]                 # >= largest eigenvalue
        u = -Nrm                                                   # push direction: into the surface
        F = np.zeros((self.s.n, 3))
        Y, tk = F.copy(), 1.0
        for _ in range(self.iters):
            Fn = _cone_project(Y - (np.einsum("nij,nj->ni", H, Y) - g) / L, u, self.cos_t, self.sin_t)
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            Y = Fn + ((tk - 1) / tn) * (Fn - F)
            F, tk = Fn, tn
        chi2 = np.einsum("ni,nij,nj->n", F, H, F) - 2 * np.einsum("ni,ni->n", g, F) + b @ b
        chi2 = np.maximum(chi2, 0.0)
        for k in np.flatnonzero(stop_dir):
            # torque this candidate's force would put on joint k, in its +tick direction
            pred = np.einsum("ni,ni->n", Jp[:, :, self.model.dof[k]], F) / g_tick[k]
            away = -stop_dir[k] * pred                       # > 0: pulls the joint off its stop
            seen = max(-stop_dir[k] * tau_tick[k], 0.0)      # did it actually come off?
            excess = np.maximum(away - config.STOP_PRELOAD_NM, 0.0)
            chi2 = chi2 + ((excess - seen) / sigma_tick[k]) ** 2
        loglik = -0.5 * chi2
        if self.finger_force_n:
            fmag = np.maximum(np.linalg.norm(F, axis=1), 0.05)
            loglik -= 0.5 * (np.log(fmag / self.finger_force_n) / np.log(config.FINGER_FORCE_SPREAD)) ** 2

        self.logp += (1.0 if self.updates == 0 else self.temper) * loglik
        self.updates += 1
        lp = self.logp - self.logp.max()
        w = np.exp(lp)
        w /= w.sum()
        # Most probable LINK first (total probability over its surface), then the best point
        # on it. The single best point can sit on a neighbouring link when one link's
        # probability is spread over many near-equal points (upper arm: 2 joints only).
        link_prob = {name: float(w[self._link_idx == k].sum()) for k, name in enumerate(self.links)}
        best_link = self.links.index(max(link_prob, key=link_prob.get))
        i = int(np.argmax(np.where(self._link_idx == best_link, w, -1.0)))
        spread = float(np.sqrt(w @ np.sum((P - P[i]) ** 2, axis=1))) * 100
        top = np.argsort(w)[::-1][: self.n_heat]
        return Contact(self.links[best_link], P[i].copy(), Nrm[i].copy(), F[i].copy(),
                       spread, link_prob, float(chi2[i]), top, w[top] / w[top[0]], P[top].copy())
