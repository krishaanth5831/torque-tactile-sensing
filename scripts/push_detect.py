"""Step 1: "is someone pushing me?" from the servo signals alone.

Two modes:

  hold    The arm holds its pose. For the first --learn seconds DO NOT TOUCH IT: it
          learns the normal signal (mean + noise) per joint. After that, any joint
          whose signal leaves the band lights up red. Push it with your hand.

  wiggle  One joint moves back and forth (sine, --amp ticks, --period s). Gravity,
          friction and inertia change all through the motion, so a fixed threshold
          cannot work. Instead the first --learn-cycles cycles teach it the expected
          signal at each point in the cycle. Then push the arm while it moves.

Reaction to a touch (--react):
  none    just show it
  stop    wiggle: pause the motion for --stop-s, then carry on (collision stop)
  comply  hold: give way. The goal follows where you push it, so you can lead the arm
          around by hand. Once it has been still for 0.3 s it re-holds and re-learns.

    python scripts/push_detect.py --fake
    python scripts/push_detect.py --fake --mode wiggle --joint elbow_flex --react stop
    python scripts/push_detect.py --mode hold --torque-limit 300
    python scripts/push_detect.py --mode wiggle --joint shoulder_pan --amp 150 --period 4 --react stop

--signal picks what to watch: load (default), cur, or err (goal - present).
Try all three: on a P-controlled servo err is often the cleanest.
"""
import math
from collections import deque

import numpy as np
from _common import RateMeter, base_parser, open_and_report, parse_joints

from tactile import config
from tactile.detect import Baseline, Hysteresis, PhaseBaseline
from tactile.liveplot import Live, Panel, run_headless
from tactile.logger import CsvLogger

p = base_parser(__doc__)
p.add_argument("--mode", choices=["hold", "wiggle"], default="hold")
p.add_argument("--joints", default="arm", help="joints to hold + watch (default: arm = all but gripper)")
p.add_argument("--signal", choices=["load", "cur", "err"], default="load")
p.add_argument("--torque-limit", type=int, default=300)
p.add_argument("--learn", type=float, default=3.0, help="hold: seconds of no-touch baseline")
p.add_argument("--z-on", type=float, default=6.0, help="touch when |z| above this")
p.add_argument("--z-off", type=float, default=3.0, help="touch ends when |z| below this")
p.add_argument("--joint", default="elbow_flex", help="wiggle: which joint moves")
p.add_argument("--amp", type=float, default=120, help="wiggle amplitude, ticks (4096 = one turn)")
p.add_argument("--period", type=float, default=3.0, help="wiggle period, s")
p.add_argument("--learn-cycles", type=int, default=3)
p.add_argument("--react", choices=["none", "stop", "comply"], default="none")
p.add_argument("--stop-s", type=float, default=1.0, help="wiggle + stop: pause this long, then carry on")
args = p.parse_args()

joints = parse_joints(args.joints, allow_gripper=False)
if args.mode == "wiggle" and args.joint not in joints:
    joints.append(args.joint)
src = open_and_report(args, pushes=True)
log = CsvLogger(f"push_{args.mode}", config.JOINTS, extra=["phase", "touch"] + [f"z_{j}" for j in joints],
                fake=args.fake)

panels = [
    Panel(f"{args.signal} (raw)", [f"sig/{j}" for j in joints]),
    Panel(f"z-score vs learned baseline (red = {args.z_on})", [f"z/{j}" for j in joints], ylabel="z",
          hlines=(args.z_on, -args.z_on)),
]
live = Live(f"push_detect ({args.mode})", panels)


def signal(s):
    d = getattr(s, args.signal)
    return np.array([0.0 if d[j] is None else d[j] for j in joints])


def worker(lv: Live):
    src.hold_here(joints, args.torque_limit)
    hyst = Hysteresis(len(joints), args.z_on, args.z_off)
    rate = RateMeter()
    t_start = None
    center = None
    phase_t = 0.0             # wiggle clock, pauses while stopped
    paused_until = grace_until = 0.0
    last_t = None
    base = Baseline() if args.mode == "hold" else PhaseBaseline(len(joints))
    learn_until = None
    follow = np.zeros(len(joints), bool)   # comply: joints currently following the hand
    hist = deque()
    errs, err0 = [], np.zeros(len(joints))

    while not lv.stop.is_set():
        s = src.read()
        if t_start is None:
            t_start = s.t
            last_t = s.t
            learn_until = s.t + args.learn
            center = s.pos[args.joint] if args.mode == "wiggle" else None
        dt = s.t - last_t
        last_t = s.t
        x = signal(s)
        touch = np.zeros(len(joints), bool)
        z = np.full(len(joints), np.nan)

        if args.mode == "hold":
            if not base.ready:
                base.add(x)
                errs.append([s.err[j] or 0 for j in joints])
                lv.set_banner(f"LEARNING, don't touch ({learn_until - s.t:.1f}s)", "orange")
                if s.t >= learn_until:
                    base.freeze()
                    err0 = np.mean(errs, axis=0)   # the error that holds the arm up against gravity
                    errs.clear()
            else:
                z = base.z(x)
                touch = hyst.update(z)
                if args.react == "comply":
                    # Give way: every touched joint's goal follows where the hand puts it.
                    # When the arm has been still for 0.3 s the hand has let go (or is just
                    # holding). Re-hold there and re-learn, because gravity is different
                    # at the new pose, so the old baseline no longer applies.
                    # goal = pos + err0, not pos: keep the gravity-holding offset or it sags.
                    follow |= touch
                    if follow.any():
                        src.set_goal({j: s.pos[j] + e for j, e, on in zip(joints, err0, follow) if on})
                        hist.append((s.t, np.array([s.pos[j] for j in joints])))
                        while s.t - hist[0][0] > 0.3:
                            hist.popleft()
                        span = np.ptp(np.stack([p for _, p in hist]), axis=0)
                        if s.t - hist[0][0] > 0.25 and (span[follow] < 3).all():
                            follow[:] = False
                            hist.clear()
                            base = Baseline()
                            hyst = Hysteresis(len(joints), args.z_on, args.z_off)
                            learn_until = s.t + 1.5
            phase = float("nan")
        else:
            paused = s.t < paused_until
            if not paused:
                phase_t += dt                    # the trajectory clock stops during a "stop"
            phase = (phase_t / args.period) % 1.0
            cycles = phase_t / args.period
            src.set_goal({args.joint: center + args.amp * math.sin(2 * math.pi * phase)})
            if cycles < args.learn_cycles:
                base.add(phase, x)
                lv.set_banner(f"LEARNING the motion, don't touch (cycle {int(cycles) + 1}/{args.learn_cycles})",
                              "orange")
            else:
                if not base.frozen:
                    base.freeze()
                z = base.z(phase, x)
                # while paused (and just after) the learned motion profile does not apply
                touch = hyst.update(z if s.t >= grace_until else np.zeros_like(z))
                if args.react == "stop" and touch.any() and not paused and s.t >= grace_until:
                    paused_until = s.t + args.stop_s
                    grace_until = paused_until + 0.5

        if base_ready(base) and not np.isnan(z).all():
            if touch.any():
                names = [f"{j}({'+' if zz > 0 else '-'})" for j, zz, on in zip(joints, z, touch) if on]
                lv.set_banner("TOUCH: " + ", ".join(names), "red")
            else:
                lv.set_banner("watching, push the arm", "green")
        bad = {j: v for j, v in s.status.items() if v}
        if bad:
            lv.set_banner(f"STATUS ERROR {bad}: servo may have gone soft", "purple")

        lv.push(s.t, {**{f"sig/{j}": v for j, v in zip(joints, x)}, **{f"z/{j}": v for j, v in zip(joints, z)}})
        lv.hz = rate.tick()
        log.write(s, {"phase": phase, "touch": int(touch.any()), **{f"z_{j}": v for j, v in zip(joints, z)}})


def base_ready(b):
    return b.ready if isinstance(b, Baseline) else b.frozen


try:
    if args.no_window:
        run_headless(worker, args.duration)
    else:
        live.run(worker, args.duration)
finally:
    log.close()
    src.close(release=args.release_on_exit)
