"""Replay a REAL recording through the current touch code: improve it without the arm.

    python scripts/replay_eval.py data/touch_gui_<stamp>.csv               # detection stats
    python scripts/replay_eval.py data/poke_<stamp>.csv                    # + poke accuracy
    python scripts/replay_eval.py data/poke_<stamp>.csv --variants         # compare settings

Detection: how much of the run the current estimator calls TOUCH, how many separate
touches, how long they last (a touch that lasts many seconds with nothing happening is
usually friction holding a joint off, not a finger).

Poke accuracy: every poke (touch) of a `--poke` run has a known target spot. Newer poke
CSVs log the target in every row (tgt_* columns); for older ones the targets come from the
matching data/poke_results_<stamp>.json, paired with the GUI's logged touches in order.
Each poke's torques are replayed through the localiser and the most likely point is
compared to the target, in cm. The touch segments are the ones the GUI logged at the time,
so detection changes don't shift which samples are scored.

No window: numbers only. To SEE a recording: python scripts/touch_gui.py --replay <csv>.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.contact import ContactLocalizer  # noqa: E402
from tactile.estimator import TouchEstimator  # noqa: E402
from tactile.model import SO101Model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("csv")
p.add_argument("--results", default=None, help="poke_results json (default: the one written by that run)")
p.add_argument("--variants", action="store_true", help="compare: old behaviour, each fix, wrist_roll geometry")
p.add_argument("--p-gain", type=int, default=config.P_DEFAULT, help="P gain the run used (16, or --soft P)")
args = p.parse_args()

J = config.JOINTS
RANGES = config.load_ranges()
d = pd.read_csv(args.csv, low_memory=False)
d = d[d[[f"goal_{j}" for j in J]].notna().all(axis=1)].reset_index(drop=True)   # holding only
t = d.t.to_numpy()
print(f"{args.csv}: {len(d)} holding rows, {t[-1] - t[0]:.0f} s, {len(d) / (t[-1] - t[0]):.0f} Hz")
if "p_gain" in d.columns and d.p_gain.notna().any():
    args.p_gain = int(d.p_gain.dropna().iloc[-1])


def segments(flag: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) row indices where flag is true."""
    f = np.r_[False, flag.astype(bool), False].astype(int)
    on, off = np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1)
    return list(zip(on, off))


# ---------------------------------------------------------------- detection with the current code
est = TouchEstimator(SO101Model(ranges=RANGES), p_gain=args.p_gain)
states, touching = [], np.zeros(len(d), bool)
pos_cols = [f"pos_{j}" for j in J]
goal_cols = [f"goal_{j}" for j in J]
if all(f"nom_{j}" in d.columns for j in J) and d[[f"nom_{j}" for j in J]].notna().all(axis=None):
    goal_cols = [f"nom_{j}" for j in J]          # the unshaken goal (runs with --dither)
P, G = d[pos_cols].to_numpy(), d[goal_cols].to_numpy()
taus, sigs = np.zeros((len(d), len(J))), np.ones((len(d), len(J)))
for i in range(len(d)):
    o = est.update(t[i], dict(zip(J, G[i])), dict(zip(J, P[i])))
    states.append(o.state)
    touching[i] = o.touching
    taus[i], sigs[i] = o.tau_nm, o.sigma_nm
segs_now = segments(touching)
dur = np.array([t[b - 1] - t[a] for a, b in segs_now]) if segs_now else np.zeros(0)
old = segments(d.touch.to_numpy() == 1) if "touch" in d else []
print(f"  current code: TOUCH {touching.mean():.0%} of the time, {len(segs_now)} touches"
      + (f", median {np.median(dur):.1f} s, longest {dur.max():.1f} s" if len(dur) else ""))
if old:
    od = np.array([t[b - 1] - t[a] for a, b in old])
    print(f"  as recorded:  TOUCH {(d.touch == 1).mean():.0%} of the time, {len(old)} touches, "
          f"median {np.median(od):.1f} s, longest {od.max():.1f} s")

# ---------------------------------------------------------------- poke targets
targets = []                              # (row start, row end, body name, local xyz)
if "tgt_body" in d.columns:
    for a, b in old:
        tb = d.tgt_body.iloc[a:b].dropna()
        if len(tb):
            k = tb.index[-1]
            targets.append((a, b, tb.iloc[-1], d.loc[k, ["tgt_x", "tgt_y", "tgt_z"]].to_numpy(float)))
else:
    rp = Path(args.results) if args.results else None
    if rp is None:
        stamp = Path(args.csv).stem.split("_", 1)[1] if "_" in Path(args.csv).stem else ""
        cands = sorted(Path(args.csv).parent.glob("poke_results_*.json"))
        # the results file is written when the run ends: the first one after the CSV started
        rp = next((c for c in cands if c.stem.split("_", 2)[2] >= stamp), None) if stamp else None
    if rp is not None and rp.exists() and old:
        res = json.loads(rp.read_text())
        print(f"  poke targets: {rp} ({len(res)} pokes, paired with the last {len(res)} recorded touches)")
        for (a, b), r in zip(old[-len(res):], res):
            targets.append((a, b, r["target_body"], np.array(r["target_local"], float)))
if not targets:
    sys.exit(0)

# ---------------------------------------------------------------- localisation per variant
VARIANTS = {"now (stop-aware + finger prior)": dict(stop=True, finger=config.FINGER_FORCE_N, reg=config.FORCE_REG_N)}
if args.variants:
    VARIANTS = {
        "as it ran (no stop handling, min-force prior)": dict(stop=False, finger=None, reg=10.0),
        "stop-aware only": dict(stop=True, finger=None, reg=10.0),
        "finger prior only": dict(stop=False, finger=config.FINGER_FORCE_N, reg=config.FORCE_REG_N),
        **VARIANTS,
        "now + wrist_roll +1024": dict(stop=True, finger=config.FINGER_FORCE_N, reg=config.FORCE_REG_N, roll=1024),
        "now + wrist_roll -1024": dict(stop=True, finger=config.FINGER_FORCE_N, reg=config.FORCE_REG_N, roll=-1024),
    }
model = SO101Model(ranges=RANGES)
base_roll = config.JOINT_OFFSET_TICKS["wrist_roll"]
summary = {}
for name, v in VARIANTS.items():
    config.JOINT_OFFSET_TICKS["wrist_roll"] = base_roll + v.get("roll", 0)
    loc = ContactLocalizer(model, finger_force_n=v["finger"], force_reg_n=v["reg"])
    rows = []
    for a, b, body, local in targets:
        loc.reset()
        c, next_t = None, -1.0
        for i in range(a, b):
            if t[i] < next_t:
                continue
            next_t = t[i] + 0.04
            pos = dict(zip(J, P[i]))
            c = loc.update(pos, taus[i], sigs[i], est.stop_dir(pos) if v["stop"] else None)
        bid = model.m.body(body).id
        tw = model.d.xpos[bid] + model.d.xmat[bid].reshape(3, 3) @ local
        link_true = loc.s.link_of_body.get(bid, body)
        rows.append((link_true, c.link, float(np.linalg.norm(c.point - tw) * 100), c.spread_cm,
                     float(np.linalg.norm(c.force))))
    summary[name] = rows
config.JOINT_OFFSET_TICKS["wrist_roll"] = base_roll

for name, rows in summary.items():
    e = np.array([r[2] for r in rows])
    ok = np.mean([r[0] == r[1] for r in rows])
    print(f"\n  {name}: median error {np.median(e):.1f} cm, 80 % within {np.percentile(e, 80):.1f} cm, "
          f"right link {ok:.0%}")
    if name.startswith("now (") or not args.variants:
        for k, (lt, le, err, spr, fn) in enumerate(rows, 1):
            print(f"    poke {k:2d}: target {lt:20s} -> {le:20s} {err:5.1f} cm (claims +-{spr:.1f}), ~{fn:.1f} N")
