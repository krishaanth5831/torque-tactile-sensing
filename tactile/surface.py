"""Where on the arm CAN a finger touch? Points sampled over the real link surfaces.

The localiser (contact.py) does not guess along a link's centre line: it scores every
point of this map. Each point is stored in its BODY frame with its outward normal, so
the map is built once and moves with the arm.

Built from the visual meshes (group 2, the STLs you see in the viewer):
  1. area-weighted random points on every triangle (so big flat faces get as many
     points per cm^2 as small curved ones), with the triangle's normal
  2. drop points a finger cannot reach: shoot a ray 0.5 mm off the surface along the
     normal; if it hits arm geometry within REACH_M, that face is inside the arm, faces
     a neighbouring part across a gap, or points inward (a mesh winding error)

The fixed base is left out: a push on it moves no joint, so it can never be felt.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config

VISUAL_GROUP = 2
REACH_M = 0.012          # free space a fingertip needs in front of a surface point


@dataclass
class SurfaceMap:
    body: np.ndarray        # (N,) body id of each point
    local: np.ndarray       # (N,3) point in its body frame (m)
    normal: np.ndarray      # (N,3) outward unit normal in its body frame
    bodies: np.ndarray      # unique body ids, in chain order
    link_of_body: dict      # body id -> link name shown to the user

    @property
    def n(self) -> int:
        return len(self.body)

    def world(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) world points and (N,3) world normals at the current pose of `d`."""
        R = d.xmat[self.body].reshape(-1, 3, 3)
        p = d.xpos[self.body] + np.einsum("nij,nj->ni", R, self.local)
        nrm = np.einsum("nij,nj->ni", R, self.normal)
        return p, nrm


def _link_name(m: mujoco.MjModel, b: int) -> str:
    """Bodies welded to a moving link (camera_mount on the gripper) report that link."""
    while m.body_jntnum[b] == 0 and m.body_parentid[b] != 0:
        b = m.body_parentid[b]
    return m.body(b).name


def build_surface(m: mujoco.MjModel, spacing_m: float = config.SURFACE_SPACING_M, seed: int = 0,
                  bodies: list[str] = config.TOUCH_BODIES) -> SurfaceMap:
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)               # qpos = 0: any pose works, points are stored body-local
    want = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
    if -1 in want:
        raise ValueError(f"unknown body in TOUCH_BODIES {bodies}")

    B, L, N = [], [], []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in want or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != VISUAL_GROUP:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        a, bb, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cr = np.cross(bb - a, c - a)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        keep = area > 1e-12
        a, bb, c, cr, area = a[keep], bb[keep], c[keep], cr[keep], area[keep]
        n_pts = int(np.ceil(area.sum() / spacing_m ** 2))
        tri = rng.choice(len(area), size=n_pts, p=area / area.sum())
        r1, r2 = rng.random(n_pts), rng.random(n_pts)
        s = np.sqrt(r1)
        pts = (1 - s)[:, None] * a[tri] + (s * (1 - r2))[:, None] * bb[tri] + (s * r2)[:, None] * c[tri]
        nrm = cr[tri] / np.linalg.norm(cr[tri], axis=1, keepdims=True)
        # mesh frame -> world (qpos=0) -> body frame
        Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        pw = pts @ Rg.T + pg
        nw = nrm @ Rg.T
        B.append(np.full(n_pts, b))
        L.append((pw - pb) @ Rb)
        N.append(nw @ Rb)
    body, local, normal = np.concatenate(B), np.concatenate(L), np.concatenate(N)

    # reachability: rays against everything visible (all visual meshes + the floor's group 0)
    groups = np.zeros(6, np.uint8)
    groups[VISUAL_GROUP] = 1
    groups[0] = 1
    geomid = np.zeros(1, np.int32)
    R = d.xmat[body].reshape(-1, 3, 3)
    pw = d.xpos[body] + np.einsum("nij,nj->ni", R, local)
    nw = np.einsum("nij,nj->ni", R, normal)
    ok = np.ones(len(body), bool)
    for i in range(len(body)):
        dist = mujoco.mj_ray(m, d, pw[i] + 5e-4 * nw[i], nw[i], groups, 1, -1, geomid)
        ok[i] = dist < 0 or dist > REACH_M
    order = np.argsort(body[ok], kind="stable")
    body, local, normal = body[ok][order], local[ok][order], normal[ok][order]
    uniq = np.array(sorted(set(body.tolist())))
    return SurfaceMap(body, local, normal, uniq, {int(b): _link_name(m, int(b)) for b in uniq})
