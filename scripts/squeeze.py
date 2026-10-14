"""Step 2: squeeze something with the gripper and FEEL it.

The gripper closes slowly from where it is now. Every tick it watches the grip
signal (how hard the jaws are being resisted). It reports:

  contact    the travel where the object first pushed back
  stiffness  grip signal per tick of further closing, right after contact.
             A firm green tomato should read higher than a soft ripe one.
  yield      the stiffness suddenly collapsed while still closing, i.e. something
             gave way (tomato skin splitting, a sponge bottoming out...). It backs
             off immediately.

It stops closing at whichever comes first:
  --target    grip signal reached: hold there for --hold-s, then open again
  yield       backs off by --backoff ticks
  stall       the jaw stops advancing. Either the servo hit Torque_Limit (the
              HARDWARE force cap: grip force is now set by Torque_Limit, whatever
              the object does), or the goal hit --max-lead. On a P servo,
              drive ~= P * (goal - pos), so capping the lead ALSO caps the force.
  closed end  nothing in the gripper

SAFETY: --torque-limit is the real force ceiling, enforced by the servo itself.
Start LOW (default 250 = 25 %%) and with something you don't mind crushing
(a sponge, a foam ball, a cheap supermarket tomato).

Run probe_gripper.py first and set GRIPPER_CLOSED_END in tactile/config.py.

    python scripts/squeeze.py --fake
    python scripts/squeeze.py --fake --torque-limit 600 --target 500 --max-lead 400   # crush + back off
    python scripts/squeeze.py --torque-limit 250 --target 150 --speed 80
"""
import numpy as np
from _common import RateMeter, base_parser, open_and_report

from tactile import config
from tactile.detect import analyse_squeeze
from tactile.liveplot import Live, Panel, run_headless
from tactile.logger import CsvLogger

p = base_parser(__doc__)
p.add_argument("--closed-end", choices=["min", "max"], default=config.GRIPPER_CLOSED_END,
               help="which raw end is closed (from probe_gripper.py)")
p.add_argument("--signal", choices=["load", "cur", "err"], default="load")
p.add_argument("--torque-limit", type=int, default=250)
p.add_argument("--target", type=float, default=150, help="stop closing at this grip signal")
p.add_argument("--speed", type=float, default=100, help="closing speed, ticks/s")
p.add_argument("--max-lead", type=float, default=120,
               help="max ticks the goal may run ahead of the jaw (stall guard)")
p.add_argument("--hold-s", type=float, default=3.0, help="hold the grip this long, then open")
p.add_argument("--backoff", type=float, default=150, help="ticks to open on yield / error")
args = p.parse_args()

if args.fake and args.closed_end is None:
    args.closed_end = "min"
if args.closed_end is None:
    raise SystemExit("Unknown closing direction. Run scripts/probe_gripper.py, then set "
                     "GRIPPER_CLOSED_END in tactile/config.py (or pass --closed-end).")
if args.fake:
    config.GRIPPER_CLOSED_END = args.closed_end
src = open_and_report(args, pushes=False)
G = "gripper"
lo, hi = src.ranges[G]
closed_raw = lo if args.closed_end == "min" else hi
sign = -1 if args.closed_end == "min" else 1          # +1 tick of travel = sign * 1 raw tick
log = CsvLogger("squeeze", config.JOINTS, extra=["travel", "grip", "phase"], fake=args.fake)

live = Live("squeeze: gripper force sensing", [
    Panel("grip signal vs time", ["grip"], hlines=(args.target,)),
    Panel("closing travel (ticks): jaw vs goal", ["travel", "goal_travel"], ylabel="ticks"),
])
result = {}


def grip_of(s) -> float:
    if args.signal == "err":
        return 0.0 if s.err[G] is None else sign * s.err[G]   # goal ahead of the jaw = being resisted
    v = s.load[G] if args.signal == "load" else s.cur[G]
    return abs(v)


def worker(lv: Live):
    src.hold_here([G], args.torque_limit)
    s = src.read()
    start = s.pos[G]
    goal = float(start)
    rate = RateMeter()
    travels, grips = [], []
    phase, t_phase = "closing", s.t
    last_t = s.t
    last_progress_t, best_travel = s.t, 0.0
    noise = None

    while not lv.stop.is_set():
        s = src.read()
        dt, last_t = s.t - last_t, s.t
        travel = sign * (s.pos[G] - start)
        grip = grip_of(s)
        travels.append(travel)
        grips.append(grip)

        if any(s.status.values()):
            phase = f"STATUS ERROR {s.status}"
        if phase == "closing":
            lead = sign * (goal - s.pos[G])
            if lead < args.max_lead:
                goal += sign * args.speed * dt
            goal = float(src.clamp(G, goal))
            src.set_goal({G: goal})
            if travel > best_travel + 2:
                best_travel, last_progress_t = travel, s.t
            if noise is None and len(grips) > 15:
                noise = max(float(np.std(grips[3:15])), 3.0)
            r = analyse_squeeze(travels, grips, free_load_std=noise or 6.0)
            result["r"] = r
            if r.yield_idx is not None:
                phase = "YIELD"
            elif grip >= args.target:
                phase, t_phase = "holding", s.t
            elif abs(goal - closed_raw) < 2 and lead < 15:
                phase = "EMPTY (reached closed end, no contact)"
            elif s.t - last_progress_t > 1.5 and r.contact_idx is not None:
                phase, t_phase = "STALL (force cap: Torque_Limit or --max-lead), holding", s.t
        elif phase.startswith(("holding", "STALL")):
            if s.t - t_phase > args.hold_s:
                phase = "opening"
                src.set_goal({G: start})
        elif phase.startswith(("YIELD", "STATUS", "EMPTY")):
            src.set_goal({G: s.pos[G] - sign * args.backoff})
            result["end"] = phase
            lv.set_banner(phase + ", backed off", "red")
            for _ in range(30):
                s = src.read()
                lv.push(s.t, {"grip": grip_of(s), "travel": sign * (s.pos[G] - start)})
            break
        elif phase == "opening":
            if abs(s.pos[G] - start) < 20:
                result["end"] = result.get("end", "done")
                break

        r = result.get("r")
        txt = phase
        if r and r.contact_idx is not None:
            txt += f" | contact @ {r.contact_pos:.0f} ticks"
            if r.stiffness is not None:
                txt += f", stiffness {r.stiffness:.2f}/tick"
        color = {"closing": "black", "holding": "green", "opening": "grey"}.get(phase.split()[0], "orange")
        lv.set_banner(txt, color)
        lv.push(s.t, {"grip": grip, "travel": travel, "goal_travel": sign * (goal - start)})
        lv.hz = rate.tick()
        log.write(s, {"travel": travel, "grip": grip, "phase": phase})
        if phase in ("holding",) or phase.startswith("STALL"):
            result.setdefault("end", phase)


try:
    if args.no_window:
        run_headless(worker, args.duration)
    else:
        live.run(worker, args.duration)
finally:
    log.close()
    src.close(release=args.release_on_exit)

r = result.get("r")
print("\n=== squeeze result ===")
print(f"  ended:      {result.get('end', 'stopped early')}")
if r and r.contact_idx is not None:
    print(f"  contact at: {r.contact_pos:.0f} ticks of travel")
    print(f"  stiffness:  {r.stiffness:.3f} {args.signal}-units per tick" if r.stiffness else "  stiffness:  n/a")
    print(f"  peak grip:  {r.peak_load:.0f} {args.signal}-units")
    print(f"  yield:      {'YES at sample %d' % r.yield_idx if r.yield_idx is not None else 'no'}")
else:
    print("  no contact detected")
