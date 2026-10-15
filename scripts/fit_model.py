"""Fit the "expected load while moving" model from a NO-TOUCH recording.

    python scripts/touch_gui.py --teleop --record        # 2-3 min: move the leader everywhere,
                                                         # slow and fast, DON'T touch the follower
    python scripts/fit_model.py data/free_motion_<stamp>.csv     # opens a check window
    python scripts/touch_gui.py --teleop --model models/load_model.json

Several CSVs can be given (each is replayed separately so the derivative filter never
spans two recordings). The window shows measured vs predicted load per joint. The gap
between them is the noise floor of touch detection while moving: the printed sigma.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.dynmodel import GoalDerivatives, LoadModel, features, read_log  # noqa: E402
from tactile.model import SO101Model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("csv", nargs="+")
p.add_argument("--out", default="models/load_model.json")
p.add_argument("--no-window", action="store_true")
args = p.parse_args()

try:
    ranges = config.load_ranges()
except FileNotFoundError:
    ranges = {j: (900, 3100) for j in config.JOINTS}
model = SO101Model(ranges=ranges)

rows = []
for f in args.csv:
    r = read_log(Path(f))
    print(f"{f}: {len(r)} rows")
    if rows:
        rows.append({"t": rows[-1]["t"], "goal": {j: None for j in config.JOINTS}, "load": rows[-1]["load"]})
    rows += r
lm = LoadModel.fit(rows, model)
out = Path(args.out)
if out.exists():
    stamp = out.with_name(out.stem + "_prev" + out.suffix)
    out.replace(stamp)
    print(f"previous model kept as {stamp}")
lm.save(out)
print(f"\nsaved {out}")
print(f"{'joint':14s} {'R^2':>6s} {'sigma (load units)':>20s}")
for j in config.JOINTS:
    print(f"{j:14s} {lm.r2[j]:6.2f} {lm.sigma[j]:20.1f}")
print("\nLow R^2 on a joint that barely moved is normal. Large sigma = less sensitive while moving.")

if not args.no_window:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    der = GoalDerivatives()
    t, meas, pred = [], {j: [] for j in config.JOINTS}, {j: [] for j in config.JOINTS}
    for r in rows:
        if any(r["goal"][j] is None for j in config.JOINTS):
            der = GoalDerivatives()
            continue
        der.update(r["t"], r["goal"])
        pr = lm.predict(features(model, r["goal"], der))
        t.append(r["t"])
        for j in config.JOINTS:
            meas[j].append(r["load"][j])
            pred[j].append(pr[j])
    fig, axes = plt.subplots(len(config.JOINTS), 1, sharex=True, figsize=(11, 12))
    fig.canvas.manager.set_window_title("fit_model: measured vs predicted load")
    for ax, j in zip(axes, config.JOINTS):
        ax.plot(t, meas[j], lw=0.8, label="measured")
        ax.plot(t, pred[j], lw=0.8, label="predicted")
        ax.set_title(f"{j}   R^2 {lm.r2[j]:.2f}   sigma {lm.sigma[j]:.1f}", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()
