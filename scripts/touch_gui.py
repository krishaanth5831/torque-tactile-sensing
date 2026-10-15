"""The SO-101's sense of touch, live in 3D.

A MuJoCo window shows the arm mirroring the real follower. When something pushes it:
  * an arrow on every joint axis shows the external torque that joint feels
    (grey = noise level, yellow -> red = clearly more than noise)
  * a HEAT CLOUD on the arm's surface: every point where a finger could be pressing to
    produce exactly these torques (red = most likely). Small cloud = it knows where.
    A cloud smeared along a link = physics can't tell (see USAGE.md: what it can know)
  * a red ball + arrow at the most likely point: where, and which way you push
  * --fake only: cyan = the TRUE simulated push, and the error in cm
  * top: TOUCH <link>  +-<cm>  ~<N>. Bottom-left: per-joint torque ~N.m and z.

Modes:
    python scripts/touch_gui.py --fake                    # simulated arm + simulated finger pushes
    python scripts/touch_gui.py                           # real follower holds its pose: push it
    python scripts/touch_gui.py --soft 4                  # same, softer servos: finer touch
    python scripts/touch_gui.py --teleop --soft 4 --poke  # pose it with the leader, hold the leader
                                                          # still: it softens, then poke the follower
    python scripts/touch_gui.py --poke                    # accuracy test: double-click a spot on
                                                          # the model, poke it on the real arm
    python scripts/touch_gui.py --teleop                  # leader drives follower, push the follower
    python scripts/touch_gui.py --teleop --record         # record no-touch motion for fit_model.py
    python scripts/touch_gui.py --teleop --model models/load_model.json   # touch while MOVING

Keys (window focused): R = re-learn the baseline now (don't touch while it learns).
(R also toggles MuJoCo's floor reflection: harmless.)

States: LEARNING (don't touch) -> WATCHING (still, most sensitive) -> TOUCH.
While the command moves: MOVING (with --model), or "no model" (no detection while moving).
"""
import json
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from _common import RateMeter, base_parser, open_and_report

from tactile import config
from tactile.contact import ContactLocalizer
from tactile.dynmodel import LoadModel
from tactile.estimator import TouchEstimator
from tactile.logger import DATA_DIR, CsvLogger
from tactile.model import SO101Model
from tactile.source import soft_problems
from tactile.teleop import FakeLeader, LeaderArm, Teleop

p = base_parser(__doc__)
p.add_argument("--teleop", action="store_true", help="drive the follower from the leader arm")
p.add_argument("--leader-port", default=config.LEADER_PORT)
p.add_argument("--model", default=None, help="LoadModel json from fit_model.py (touch while moving)")
p.add_argument("--record", action="store_true", help="label the log as no-touch motion for fit_model.py")
p.add_argument("--torque-limit", type=int, default=500, help="follower Torque_Limit, 0..1000")
p.add_argument("--soft", type=int, default=None, metavar="P",
               help="lower the servos' P gain to P (lerobot uses 16) while this runs. Softer = more "
                    "deflection per newton = finer touch. Hold mode: at start (arm must be off its end "
                    "stops). Teleop: as soon as the follower is still and off its stops. Restored on exit")
p.add_argument("--max-step", type=float, default=40, help="teleop: max goal change per tick, ticks")
p.add_argument("--d-on", type=float, default=6.0, help="touch starts when the noise-weighted torque exceeds this")
p.add_argument("--d-off", type=float, default=3.5, help="... and ends below this")
p.add_argument("--poke", action="store_true",
               help="accuracy test: double-click the model where you will poke, poke the real arm there")
p.add_argument("--no-truth", action="store_true", help="fake: hide the true-push arrow")
p.add_argument("--limp", action="store_true",
               help="never enable torque (mirror check: move the arm by hand). SUPPORT THE ARM")
args = p.parse_args()
if args.limp and args.teleop:
    p.error("--limp and --teleop don't mix (teleop has to drive the follower)")
if args.soft is not None and args.limp:
    p.error("--soft and --limp don't mix (limp never turns torque on)")
if args.soft is not None and not 1 <= args.soft <= 32:
    p.error("--soft P: 1..32 (lerobot 16, firmware default 32)")

J = config.JOINTS
src = open_and_report(args, full=False, cartesian=args.fake, pushes=not args.record)  # record = no touch
lm = LoadModel.load(args.model) if args.model else None
if lm:
    print("[tactile] load model R^2:", {j: round(v, 2) for j, v in lm.r2.items()})
t_build = time.time()
loc = ContactLocalizer(SO101Model(ranges=src.ranges))
print(f"[tactile] touch surface: {loc.s.n} points on {', '.join(loc.links)} ({time.time() - t_build:.1f} s)")
if args.soft is not None and not args.teleop:
    # Hold mode keeps the pose it starts in, so check it BEFORE opening anything.
    s0 = src.read()
    bad = soft_problems(s0.pos, {}, src.ranges, J)
    if bad:
        src.close()
        print("\n[tactile] --soft refused, nothing was changed:\n    " + "\n    ".join(bad) +
              "\n  A joint on its end stop can't feel a push, and softening it is unsafe. Either put the"
              "\n  arm in free space first, or let the leader pose it:\n"
              f"      python scripts/touch_gui.py --teleop --soft {args.soft}" + (" --poke" if args.poke else ""),
              file=sys.stderr)
        sys.exit(2)
soft_now = args.soft if (args.soft is not None and not args.teleop) else None   # teleop: later
est = TouchEstimator(SO101Model(ranges=src.ranges), lm, p_gain=soft_now or config.P_DEFAULT,
                     d_on=args.d_on, d_off=args.d_off, localizer=loc)
leader = None
if args.teleop:
    leader = FakeLeader() if args.fake else LeaderArm(args.leader_port)
tp = Teleop(leader, src.ranges, max_step=args.max_step) if leader else None
name = "free_motion" if args.record else "poke" if args.poke else "touch_gui"
log = CsvLogger(name, J, extra=["state", "touch", "d", "contact_link", "cx", "cy", "cz", "spread_cm"]
                + [f"z_{j}" for j in J], fake=args.fake)

lock = threading.Lock()
shared = {"pos": None, "out": None, "truth": None, "hz": 0.0, "status": {}, "contact": None,
          "soft": (f"soft P={args.soft}" if soft_now else
                   f"soft P={args.soft} pending: hold the leader still" if args.soft else "")}
stop = threading.Event()
relearn = threading.Event()


worker_error = []


def worker():
    try:
        _worker()
    except BaseException as e:     # surfaced after the window closes; never die silently
        worker_error.append(e)
        raise


def _worker():
    if not args.limp:
        src.hold_here(J, args.torque_limit)
        if soft_now is not None:
            print(f"[tactile] softening servos: P_Coefficient -> {args.soft} (restored on exit)")
            src.set_p_gain(J, args.soft)
    soft_pending = args.soft is not None and soft_now is None
    rate = RateMeter()
    n = 0
    status = {}
    while not stop.is_set():
        s = src.read()
        if tp:
            src.set_goal(tp.goals(s.t, s.pos))
        if n % 10 == 0 and not args.fake:
            status = {j: v for j, v in src.read_status().items() if v}
        n += 1
        if relearn.is_set():
            est.relearn()
            relearn.clear()
        o = est.update(s.t, dict(src.goals), s.pos, s.load)
        if soft_pending and o.state == "WATCHING":
            # teleop: soften once the follower holds a pose clear of its stops. The leader keeps
            # setting the goals, so no re-aim; a soft follower sags a little below the leader.
            bad = soft_problems(s.pos, src.goals, src.ranges, J)
            if bad:
                with lock:
                    shared["soft"] = f"soft P={args.soft} WAITING: " + bad[0].split(" (")[0]
            else:
                print(f"[tactile] follower still and clear of its stops: P_Coefficient -> {args.soft}")
                src.set_p_gain(J, args.soft, reaim=False)
                est.set_p_gain(args.soft)
                soft_pending = False
                with lock:
                    shared["soft"] = f"soft P={args.soft}"
        with lock:
            c = shared["contact"] if o.touching else None
            shared.update(pos=dict(s.pos), out=o, hz=rate.tick(), status=status,
                          truth=getattr(src, "true_push", None))
        log.write(s, {"state": o.state, "touch": int(o.touching), "d": round(o.d, 2),
                      "contact_link": c.link if c else "",
                      **({"cx": round(float(c.point[0]), 4), "cy": round(float(c.point[1]), 4),
                          "cz": round(float(c.point[2]), 4), "spread_cm": round(c.spread_cm, 2)} if c else {}),
                      **{f"z_{j}": round(float(o.z[k]), 2) for k, j in enumerate(J)}})


# ---------------------------------------------------------------------- drawing
vm = SO101Model(ranges=src.ranges)          # the viewer's own model + data
m, d = vm.m, vm.d
orig_rgba = m.geom_rgba.copy()
visual = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
RED, CYAN, GREEN = np.array([1, .1, .1, 1]), np.array([0, .9, 1, .9]), np.array([.1, 1, .2, .9])
STOP_MARGIN = 25     # ticks: closer than this to a calibrated end = probably resting on the stop


def add_arrow(scn, a, b, rgba, width):
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, a, b)
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p,
                        np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def force_arrow(scn, point, force, rgba):
    """Arrow pointing AT the contact point along the push (tail outside the arm)."""
    f = np.linalg.norm(force)
    if f < 1e-6:
        return
    length = min(0.04 + 0.03 * f, 0.18)
    add_arrow(scn, point - force / f * length, point, rgba, 0.007)


def heat(z):
    a = min(max((abs(z) - 2) / 6, 0), 1)
    if abs(z) < 2:
        return np.array([.6, .6, .6, .5])
    return np.array([1, 1 - a, 0, .95])


def on_stops(pos):
    out = []
    for j in J[:5]:
        lo, hi = src.ranges[j]
        if pos[j] < lo + STOP_MARGIN or pos[j] > hi - STOP_MARGIN:
            out.append(j)
    return out


# ---------------------------------------------------------------------- poke test
poke = {"target": None, "results": [], "last": None, "last_id": 0}


def poke_target():
    """The spot double-clicked on the model: (body, body-local point)."""
    pert = viewer_ref[0].perturb
    if pert.select <= 0:
        return None
    return int(pert.select), np.array(pert.localpos)


def target_world(tgt):
    b, local = tgt
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ local


def poke_update(o, c):
    """Score the final estimate of every finished touch against the clicked target."""
    if o is not None and o.touching and c is not None and poke["target"] is not None:
        poke["last"] = (o.touch_id, c, target_world(poke["target"]).copy(), poke["target"])
    if poke["last"] and (o is None or not o.touching or o.touch_id != poke["last"][0]):
        tid, c, tw, (b, local) = poke["last"]
        poke["last"] = None
        link_true = loc.s.link_of_body.get(b, m.body(b).name)
        r = {"target_body": m.body(b).name, "target_link": link_true, "target_local": local.tolist(),
             "est_link": c.link, "err_cm": float(np.linalg.norm(c.point - tw) * 100),
             "spread_cm": c.spread_cm, "force_n": float(np.linalg.norm(c.force))}
        poke["results"].append(r)
        same = [x["err_cm"] for x in poke["results"] if x["target_local"] == r["target_local"]]
        print(f"[poke] #{len(poke['results'])}: target {link_true}, estimate {c.link}, error "
              f"{r['err_cm']:.1f} cm (claimed +-{c.spread_cm:.1f}), ~{r['force_n']:.1f} N  | this spot: "
              f"median {np.median(same):.1f} cm over {len(same)}")


def draw(viewer, pos, o, c, truth, hz, status):
    d.qpos[:] = vm.raw_to_qpos(pos)
    mujoco.mj_forward(m, d)
    m.geom_rgba[:] = orig_rgba
    scn = viewer.user_scn
    scn.ngeom = 0
    if o is not None:
        for k, j in enumerate(J[:5]):                 # torque arrows along each joint axis
            jid = vm.jid[k]
            tau = o.tau_nm[k] * vm.torque_tick_to_model[k]
            L = float(np.clip(abs(tau) / 0.3 * 0.08, 0, 0.12))
            axis = d.xaxis[jid] * np.sign(tau)
            add_arrow(scn, d.xanchor[jid], d.xanchor[jid] + axis * L, heat(o.z[k]), 0.004)
    if c is not None:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c.link)
        for g in visual:
            if m.geom_bodyid[g] == b:
                m.geom_rgba[g] = [1, .55, .55, 1]
        for w, pt in zip(c.heat_w[::-1], c.heat_pts[::-1]):       # faint first, hottest on top
            if w > 0.02:
                add_sphere(scn, pt, 0.003 + 0.003 * w, np.array([1, 1 - w, 0, 0.25 + 0.6 * w]))
        add_sphere(scn, c.point, 0.008, RED)
        force_arrow(scn, c.point, c.force, RED)
    if truth is not None and not args.no_truth:
        _, tp_, tf = truth
        add_sphere(scn, tp_, 0.006, CYAN)
        force_arrow(scn, tp_, tf, CYAN)
    if args.poke and poke["target"] is not None:
        add_sphere(scn, target_world(poke["target"]), 0.007, GREEN)

    mode = ("FAKE " if args.fake else "") + ("TELEOP" if args.teleop else "LIMP (mirror check)" if args.limp
                                               else "HOLD") + (f"  {shared['soft']}" if shared["soft"] else "")
    if args.record:
        mode += "  RECORDING no-touch motion"
    if args.poke:
        mode += "  POKE TEST"
    state = (o.state + (f"  ({o.note})" if o.note else "")) if o else "starting"
    big = state
    if c is not None:
        fN = np.linalg.norm(c.force)
        big = f"TOUCH  {c.link}  +-{c.spread_cm:.1f} cm  ~{fN:.1f} N"
        top = sorted(c.link_prob.items(), key=lambda kv: -kv[1])
        if top[0][1] < 0.85:
            big += "\n" + "  ".join(f"{k} {v:.0%}" for k, v in top[:3] if v > 0.05)
        if truth is not None and not args.no_truth:
            big += f"\nerror vs true push: {np.linalg.norm(c.point - truth[1]) * 100:.1f} cm"
    stops = on_stops(pos)
    if stops and not args.limp:
        big = (f"{', '.join(stops)} resting on its END STOP: the stop takes the push, not the motor.\n"
               "Lift the arm into free space (or move the leader), then press R")
    if status:
        big = f"SERVO STATUS ERROR {status}"
    left = "\n".join(J)
    right = "\n".join(
        f"{o.tau_nm[k]:+.3f} N.m  z{o.z[k]:+6.1f}  " + "|" * int(min(abs(o.z[k]), 30)) if o else "-"
        for k in range(len(J)))
    br = "~ = uncalibrated scale\nR = re-learn baseline"
    if args.poke:
        res = poke["results"]
        br = ("double-click the model where you will poke (green),\nthen poke that spot on the real arm.\n"
              + (f"{len(res)} pokes, median error {np.median([r['err_cm'] for r in res]):.1f} cm" if res
                 else "no pokes yet"))
    texts[0] = [
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "mode\nstate\nrate",
         f"{mode}\n{state}\n{hz:.0f} Hz"),
        (mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOP, big, ""),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, left, right),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, br, ""),
    ]
    new_texts.set()


# viewer.set_texts() waits for the render thread to take the previous text. When the window
# is not being drawn (hidden, other workspace, screen locked) that wait never ends; in the
# main loop it froze the window, --duration and Ctrl+C. So texts go through this thread.
texts = [None]
new_texts = threading.Event()


def text_pump(viewer):
    while not stop.is_set():
        if new_texts.wait(0.1):
            new_texts.clear()
            viewer.set_texts(texts[0])


def on_key(keycode):
    if keycode == ord("R"):
        relearn.set()


viewer_ref = [None]
th = threading.Thread(target=worker, daemon=True)
t_end = time.time() + args.duration if args.duration else None
next_loc = 0.0
contact = None
try:
    th.start()
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer_ref[0] = viewer
        threading.Thread(target=text_pump, args=(viewer,), daemon=True).start()
        viewer.cam.lookat[:] = [0.15, 0.0, 0.15]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.85, 135, -22
        while viewer.is_running() and th.is_alive():
            if t_end and time.time() > t_end:
                break
            with lock:
                pos, o, truth = shared["pos"], shared["out"], shared["truth"]
                hz, status = shared["hz"], shared["status"]
            if o is not None and time.time() >= next_loc:       # where: ~25 Hz, in this thread
                next_loc = time.time() + 0.04
                contact = est.localize() if o.touching else None
                with lock:
                    shared["contact"] = contact
            if args.poke:
                with viewer.lock():
                    tgt = poke_target()
                if tgt is not None:
                    poke["target"] = tgt
                poke_update(o, contact)
            if pos is not None:
                with viewer.lock():
                    draw(viewer, pos, o, contact if (o is not None and o.touching) else None, truth, hz, status)
                viewer.sync()
            time.sleep(1 / 60)
except KeyboardInterrupt:
    print("\n[tactile] Ctrl+C: closing")
finally:
    # Leaving the `with` only ASKS the viewer to close. Its render thread is a daemon
    # with no handle; if Python's atexit glfw.terminate() runs while it still renders,
    # the process segfaults. Wait for it (MuJoCo 3.8.1, names it "..._launch_internal").
    # In `finally` so Ctrl+C waits too: skipping this made Ctrl+C segfault / hang.
    for t in threading.enumerate():
        if "_launch_internal" in t.name:
            t.join(timeout=5)
    stop.set()
    th.join(timeout=3)
    log.close()
    if leader:
        leader.close()
    src.close(release=args.release_on_exit)
    if args.poke and poke["results"]:
        out = DATA_DIR / f"poke_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(poke["results"], indent=1))
        e = np.array([r["err_cm"] for r in poke["results"]])
        ok = np.mean([r["est_link"] == r["target_link"] for r in poke["results"]])
        print(f"[poke] {len(e)} pokes: median error {np.median(e):.1f} cm, 80% within "
              f"{np.percentile(e, 80):.1f} cm, link right {ok:.0%}  -> {out}")
    if args.record and not worker_error:
        print(f"\nNext: python scripts/fit_model.py {log.path}")
if worker_error:
    print(f"[tactile] the sensing loop crashed: {worker_error[0]!r} (traceback above)", file=sys.stderr)
    if "status packet" in str(worker_error[0]):
        print("[tactile] The follower's bus did not answer properly. Most likely another program is using it\n"
              "          (lerobot-teleoperate / ~/teleop.sh, another touch_gui): close it and run again.\n"
              "          Otherwise check the 12 V PSU and the USB cable, and just run it again.", file=sys.stderr)
# Everything that matters is closed above (CSV, servo P gain + torque limit, ports). Exit
# NOW: a viewer thread stuck inside MuJoCo's C++ (window not being drawn) would otherwise
# hang interpreter shutdown, and glfw's atexit hook can segfault on a live render thread.
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if worker_error else 0)
