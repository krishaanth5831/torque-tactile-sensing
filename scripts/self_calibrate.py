"""Self-calibration: measure what NOT-a-touch looks like on this arm, with nobody touching it.

The touch model has to know two things per joint that can't be guessed:
  jitter     how much err (goal - pos) wobbles at rest  -> the noise floor, i.e. how weak
             a push can be and still be felt
  rest band  how far a joint can settle off its goal because of gear friction + the
             servo's dead zone -> offsets inside it are not a touch (too small a band gave
             touches that never ended, too big a band misses light pushes)

How, for one joint at a time (the others hold still):
  1. go a little above a test point, come back down to it, park  -> friction holds it ABOVE
  2. go a little below, come back up to it, park                  -> friction holds it BELOW
  Gravity is identical both times, so  (err from below - err from above) / 2  = rest band.

SAFETY (hard limits, tactile/envelope.py):
  * every goal is clamped to start +- 1.5 % of the joint's calibrated range
  * if any joint is MEASURED more than 1.8 % from where it started, the run aborts and all
    goals go back to the start pose
  * speed ~15 ticks/s (1.3 deg/s), Torque_Limit 500, P gain untouched (16)
  * joints that start next to an end stop are only moved away from it

    python scripts/self_calibrate.py --fake                 # simulated arm, from the real base pose
    python scripts/self_calibrate.py --fake --demo-abort    # shows the abort: a simulated shove
    python scripts/self_calibrate.py                        # the real follower (torque on, holds at the end)
"""
import json
import time

import numpy as np
from _common import base_parser, open_and_report

from tactile import config
from tactile.envelope import SafetyEnvelope
from tactile.liveplot import Live, Panel, run_headless
from tactile.logger import DATA_DIR, CsvLogger
from tactile.sim import SimArmSource

p = base_parser(__doc__)
p.add_argument("--torque-limit", type=int, default=500)
p.add_argument("--speed", type=float, default=15.0, help="ticks per second while moving (max 20)")
p.add_argument("--hold-s", type=float, default=1.5, help="seconds parked at each test point")
p.add_argument("--sim-friction", type=float, default=0.05, help="--fake: gear friction to plant, N.m (0.05 ~ the real arm)")
p.add_argument("--demo-abort", action="store_true", help="--fake: shove the simulated gripper mid-run")
args = p.parse_args()
if args.speed > 20:
    p.error("--speed is capped at 20 ticks/s")
if args.demo_abort and not args.fake:
    p.error("--demo-abort is for --fake only")

J = config.JOINTS
# The real follower's base pose, read 2026-10-15 after teleop (torque off, resting on its stops)
BASE_POSE = {"shoulder_pan": 2100, "shoulder_lift": 789, "elbow_flex": 3133, "wrist_flex": 2761,
             "wrist_roll": 3118, "gripper": 2040}

if args.fake:
    src = SimArmSource(pushes=False, pose=BASE_POSE, friction_nm=args.sim_friction)
    print(f"[tactile] source: FAKE, starting from the real base pose, planted friction {args.sim_friction} N.m")
else:
    src = open_and_report(args, full=False)
log = CsvLogger("self_calibrate", J, extra=["stage", "joint"], fake=args.fake)


class Aborted(Exception):
    pass


def plan(env, j):
    """Waypoints for joint j: (goal offset from start, hold seconds, label)."""
    lo, hi = src.ranges[j]
    s, a = env.start[j], 0.95 * env.cmd[j]
    up = min(s + a, hi - 10) - s                 # room above, staying 10 ticks off the range end
    down = s - max(s - a, lo + 10)
    if up > 0.6 * a and down > 0.6 * a:
        top, bot, mid, how = up, -down, 0.0, "both ways"
    elif up >= down:
        top, bot, mid, how = up, 0.0, up / 2, "upward only (end stop below)"
    else:
        top, bot, mid, how = 0.0, -down, -down / 2, "downward only (end stop above)"
    return [(top, 0.3, "top"), (mid, args.hold_s, "from_above"), (bot, 0.3, "bottom"),
            (mid, args.hold_s, "from_below"), (0.0, 0.5, "back")], how


def worker(live: Live):
    res = {"joints": {}, "fake": args.fake, "aborted": None}
    reads = [src.read() for _ in range(10)]
    start = {j: float(np.mean([r.pos[j] for r in reads])) for j in J}
    src.hold_here(J, args.torque_limit)                       # goal = where it is: no motion
    env = SafetyEnvelope(start, src.ranges)
    goals = {j: env.clamp_goal(j, start[j]) for j in J}
    maxdev = {j: 0.0 for j in J}
    maxcmd = {j: 0.0 for j in J}
    t_push = [None]

    def tick(stage, joint=""):
        cmd = {j: env.clamp_goal(j, g) for j, g in goals.items()}
        for j in J:
            maxcmd[j] = max(maxcmd[j], abs(env.frac(j, cmd[j])))
        src.set_goal(cmd)
        s = src.read()
        log.write(s, {"stage": stage, "joint": joint})
        for j in J:
            maxdev[j] = max(maxdev[j], abs(env.frac(j, s.pos[j])))
        live.push(s.t, {**{f"goal/{j}": 100 * env.frac(j, src.goals[j]) for j in J},
                        **{f"pos/{j}": 100 * env.frac(j, s.pos[j]) for j in J}})
        if args.demo_abort and t_push[0] is None and stage == "sweep":
            t_push[0] = s.t
            src.sim.start_push(s.t, 1.5, force_n=(10.0, 10.0), links=["gripper"])
            print("[demo] a hand pushes the simulated gripper with 10 N for 1.5 s (more than the 500 torque limit holds)")
        bad = env.violations(s.pos)
        if bad:
            raise Aborted("; ".join(bad))
        if live.stop.is_set():
            raise Aborted("window closed")
        return s

    def hold(seconds, stage, joint=""):
        out, t0 = [], time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            out.append(tick(stage, joint))
        return out

    def ramp(j, target):
        while abs(goals[j] - target) > 0.5:
            step = np.clip(target - goals[j], -args.speed / 300, args.speed / 300)
            goals[j] += step
            tick("sweep", j)
        goals[j] = target

    try:
        live.set_banner("A: holding still, measuring jitter", "blue")
        rest = hold(3.0, "rest")
        err = np.array([[s.err[j] for j in J] for s in rest[len(rest) // 3:]], float)
        for k, j in enumerate(J):
            res["joints"][j] = {"jitter_ticks": float(err[:, k].std())}
        for j in J:
            wps, how = plan(env, j)
            res["joints"][j]["sweep"] = how
            parked = {}
            for off, hs, label in wps:
                live.set_banner(f"B: {j}  ->  {label}", "darkgreen")
                ramp(j, env.clamp_goal(j, start[j] + off))
                got = hold(hs, label, j)
                if label.startswith("from_"):
                    tail = got[len(got) * 3 // 10:]
                    # mean, not median: single readings are whole ticks; the jitter makes the
                    # mean resolve fractions of a tick
                    parked[label] = float(np.mean([s.err[j] for s in tail]))
            half = (parked["from_below"] - parked["from_above"]) / 2
            res["joints"][j].update(err_from_above=parked["from_above"], err_from_below=parked["from_below"],
                                    rest_band_ticks=float(half))
        live.set_banner("done: back at the start pose", "black")
        hold(1.0, "done")
    except Aborted as e:
        res["aborted"] = str(e)
        live.set_banner(f"ABORTED: {e}", "red")
        print(f"\n[safety] ABORT: {e}. Sending every joint back to its start pose.")
        for j in J:
            goals[j] = env.clamp_goal(j, start[j])
        src.set_goal({j: goals[j] for j in J})
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 4.0:
            s = src.read()
            log.write(s, {"stage": "abort"})
            live.push(s.t, {**{f"goal/{j}": 100 * env.frac(j, src.goals[j]) for j in J},
                            **{f"pos/{j}": 100 * env.frac(j, s.pos[j]) for j in J}})
    finally:
        s = src.read()
        res["max_excursion_pct"] = {j: round(100 * maxdev[j], 2) for j in J}
        res["max_commanded_pct"] = {j: round(100 * maxcmd[j], 2) for j in J}
        res["final_offset_ticks"] = {j: round(s.pos[j] - start[j], 1) for j in J}
        worker.res = res


panels = [Panel(f"{j}: % of its range from the start pose", [f"goal/{j}", f"pos/{j}"], ylabel="%",
                ylim=(-2.4, 2.4), hlines=(-2.0, -1.8, 1.8, 2.0)) for j in J]
live = Live("self_calibrate: hard limit +-2 % (red dashed: abort at 1.8 %, limit 2 %)", panels, window_s=30)
try:
    if args.no_window:
        run_headless(worker, args.duration)
    else:
        live.run(worker, args.duration)
finally:
    log.close()
    src.close(release=args.release_on_exit)

res = getattr(worker, "res", None)
if res:
    K = config.NM_PER_TICK_P16
    print(f"\n{'joint':14s} {'commanded':>10s} {'measured':>9s} {'jitter':>8s} {'rest band':>10s} {'friction':>14s}  sweep")
    for j in J:
        r = res["joints"].get(j, {})
        band = r.get("rest_band_ticks")
        # The joint parks anywhere its drive can't beat friction: band = max(dead zone, friction / K).
        if band is None:
            fr_txt = ""
        elif band > config.DEAD_ZONE_TICKS + 0.15:
            r["friction_nm"] = K * band
            fr_txt = f"{K * band:9.3f} N.m"
        else:
            r["friction_nm_max"] = K * config.DEAD_ZONE_TICKS
            fr_txt = f"<{K * config.DEAD_ZONE_TICKS:8.3f} N.m"     # hidden by the dead zone
        print(f"{j:14s} {res['max_commanded_pct'][j]:9.2f}% {res['max_excursion_pct'][j]:8.2f}% "
              f"{r.get('jitter_ticks', float('nan')):7.2f}t {'' if band is None else f'{band:9.2f}t'} "
              f"{fr_txt:>14s}  {r.get('sweep', '')}")
    print(f"commanded: at most {max(res['max_commanded_pct'].values()):.2f} % of any joint's range (clamp 1.5 %, "
          f"hard limit 2 %); measured: at most {max(res['max_excursion_pct'].values()):.2f} %. "
          f"Final offsets from start (ticks): {res['final_offset_ticks']}")
    if res["aborted"]:
        print(f"ABORTED: {res['aborted']}")
    out = DATA_DIR / f"self_cal{'_fake' if args.fake else ''}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")
