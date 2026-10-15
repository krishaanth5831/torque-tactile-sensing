"""How accurately can the arm feel WHERE it is touched? Measured on the realistic sim.

    python scripts/bench_touch.py                         # P 16 vs 4, three friction levels
    python scripts/bench_touch.py --p 4 --friction 0.02 --trials 60

Each trial: random pose, the follower holds it, learns its baseline, then a fingertip
pushes a random point of the real surface (1-5 N, ramped like a finger) for 1.2 s.
Reported per touched link:
    det    fraction of pushes detected
    link   fraction where the most probable link is the touched one
    med/p80 cm   distance from the most probable point to the true point (detected only)
    spread cm    how unsure the estimate says it is (it should be ~ the error)
    dir    median cosine between estimated and true force direction
and over all trials: false touches before the push, and touches that never end after it.

No window: this is a numbers run. To SEE the same thing live: touch_gui.py --fake.
Run it under the memory cap like any heavy job:
    systemd-run --user --scope -p MemoryMax=3G python scripts/bench_touch.py
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.contact import ContactLocalizer  # noqa: E402
from tactile.estimator import TouchEstimator  # noqa: E402
from tactile.model import SO101Model  # noqa: E402
from tactile.sim import ServoArmSim  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--p", type=int, nargs="+", default=[16, 4])
p.add_argument("--friction", type=float, nargs="+", default=[0.01, 0.02, 0.04], help="N.m per joint")
p.add_argument("--trials", type=int, default=40)
p.add_argument("--force", type=float, nargs=2, default=[1.0, 5.0], help="push force range, N")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

J = config.JOINTS
try:
    RANGES = config.load_ranges()
except FileNotFoundError:
    RANGES = {j: (900, 3100) for j in J}
loc_model = SO101Model(ranges=RANGES)
loc = ContactLocalizer(loc_model)
est_model = SO101Model(ranges=RANGES)
LINKS = loc.links
READ_HZ, PHYS_HZ, LOC_HZ = 320, 1000, 25
T_PUSH, T_HOLD = 3.0, 1.2


def random_pose(rng):
    pose = {}
    for j in J:
        lo, hi = RANGES[j]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        pose[j] = mid + rng.uniform(-0.6, 0.6) * half if j != "gripper" else lo + 0.3 * (hi - lo)
    return pose


def trial(p_gain, friction, seed):
    rng = np.random.default_rng(seed)
    sim = ServoArmSim(RANGES, friction_nm=friction, p_gain=p_gain, seed=seed, pose=random_pose(rng))
    sim.goal = np.round(sim.pos)
    est = TouchEstimator(est_model, p_gain=p_gain, localizer=loc)
    t, next_read, next_loc = 0.0, 0.0, 0.0
    fp = stuck = False
    det = False
    c = None
    end = T_PUSH + T_HOLD + 1.6
    while t < end:
        if sim.push is None and abs(t - T_PUSH) < 0.5 / PHYS_HZ:
            sim.start_push(t, T_HOLD, force_n=tuple(args.force))
        if sim.push is not None and t > sim.push[4]:
            sim.push = None
        sim.step(t, 1 / PHYS_HZ)
        t += 1 / PHYS_HZ
        if t >= next_read:
            next_read += 1 / READ_HZ
            pos, _ = sim.measure()
            goal = dict(zip(J, sim.goal))
            o = est.update(t, goal, dict(zip(J, pos)))
            if o.touching and t < T_PUSH:
                fp = True
            if T_PUSH < t < T_PUSH + T_HOLD:
                truth = sim.true_push
                if o.touching:
                    det = True
                if t >= next_loc and o.touching:
                    next_loc = t + 1 / LOC_HZ
                    c2 = est.localize()
                    if c2 is not None and t < T_PUSH + T_HOLD - 0.15:
                        c, last_truth = c2, truth
            if t > T_PUSH + T_HOLD + 1.4 and o.touching:
                stuck = True
    out = {"fp": fp, "stuck": stuck, "det": det, "link": sim.push_link}
    if c is not None:
        link, tp, tf = last_truth
        out.update(link=link, ok=c.link == link, err_cm=float(np.linalg.norm(c.point - tp) * 100),
                   spread=c.spread_cm, cos=float(tf @ c.force / (np.linalg.norm(tf) * np.linalg.norm(c.force) + 1e-12)),
                   p_true=c.link_prob.get(link, 0.0))
    return out


rows = []
for P in args.p:
    for fr in args.friction:
        t0 = time.time()
        res = [trial(P, fr, args.seed + i) for i in range(args.trials)]
        by, pushed = defaultdict(list), defaultdict(list)
        for r in res:
            pushed[r["link"]].append(r["det"])
            if "err_cm" in r:
                by[r["link"]].append(r)
        n_det = sum(r["det"] for r in res)
        print(f"\n=== P={P}  friction={fr:.3f} N.m  ({args.trials} pushes, {time.time() - t0:.0f} s)  "
              f"detected {n_det}/{args.trials}   false touch before push: {sum(r['fp'] for r in res)}   "
              f"stuck after push: {sum(r['stuck'] for r in res)}")
        print(f"  {'link':22s} {'n':>3s} {'det':>5s} {'link ok':>8s} {'med cm':>7s} {'p80 cm':>7s} {'spread':>7s} {'dir cos':>8s}")
        for link in LINKS:
            rs = by.get(link, [])
            if not rs:
                continue
            e = np.array([r["err_cm"] for r in rs])
            print(f"  {link:22s} {len(rs):3d} {np.mean(pushed[link]):5.0%} {np.mean([r['ok'] for r in rs]):8.0%} {np.median(e):7.1f} "
                  f"{np.percentile(e, 80):7.1f} {np.median([r['spread'] for r in rs]):7.1f} "
                  f"{np.median([r['cos'] for r in rs]):8.2f}")
        allr = [r for rs in by.values() for r in rs]
        if allr:
            e = np.array([r["err_cm"] for r in allr])
            print(f"  {'ALL (localised)':22s} {len(allr):3d} {n_det / len(res):5.0%} {np.mean([r['ok'] for r in allr]):8.0%} "
                  f"{np.median(e):7.1f} {np.percentile(e, 80):7.1f}")
