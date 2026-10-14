"""Which end of the gripper's raw range is CLOSED? Run once, before squeeze.py.

Turns torque OFF on the gripper only (the rest of the arm is untouched). Then:
  1. open the jaws fully by hand
  2. close them fully by hand and keep them closed
  3. close the window (or wait for --duration)
It prints the line to put in tactile/config.py.

    python scripts/probe_gripper.py
"""
from _common import RateMeter, base_parser, open_and_report

from tactile.liveplot import Live, Panel, run_headless

p = base_parser(__doc__)
p.set_defaults(duration=20.0)
args = p.parse_args()

src = open_and_report(args, pushes=False)
lo, hi = src.ranges["gripper"]
live = Live("probe_gripper: open fully, then CLOSE and hold", [
    Panel(f"gripper Present_Position (calibrated range {lo}..{hi})", ["pos/gripper"], ylabel="ticks",
          ylim=(lo - 100, hi + 100), hlines=(lo, hi)),
])
seen = []


def worker(lv: Live):
    src.release(["gripper"])
    rate = RateMeter()
    while not lv.stop.is_set():
        s = src.read()
        seen.append(s.pos["gripper"])
        lv.push(s.t, {"pos/gripper": s.pos["gripper"]})
        lv.hz = rate.tick()
        lv.set_banner(f"pos {s.pos['gripper']}: OPEN fully, then CLOSE and hold", "black")


try:
    if args.no_window:
        run_headless(worker, args.duration)
    else:
        live.run(worker, args.duration)
finally:
    src.close(release=args.release_on_exit)

if len(seen) < 10:
    raise SystemExit("not enough samples")
final = sum(seen[-10:]) / 10
span = max(seen) - min(seen)
print(f"\nmoved over {span} ticks (min {min(seen)}, max {max(seen)}), ended at {final:.0f}")
if span < 0.5 * (hi - lo):
    print("Hardly moved. Open AND close the jaws fully, then run again.")
else:
    end = "min" if abs(final - lo) < abs(final - hi) else "max"
    print(f'Closed end = "{end}". Put this in tactile/config.py:\n\n    GRIPPER_CLOSED_END = "{end}"\n')
