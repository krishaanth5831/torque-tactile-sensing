"""The SO-101's sense of touch, live in 3D.

A MuJoCo window shows the arm mirroring the real follower. When something pushes it:
  * an arrow on every joint axis shows the external torque that joint feels
    (grey = normal, yellow -> red = more and more sure it is a touch)
  * the touched link turns red
  * a red arrow shows WHERE the push is and WHICH WAY it pushes (orange + a big
    halo = ambiguous: only the link is known, the point and direction are a guess)
  * --fake only: a cyan arrow shows the TRUE simulated push, to compare
  * top-left text: mode, state, loop rate. Bottom-left: per-joint torque ~N.m and z.

Modes:
    python scripts/touch_gui.py --fake                    # simulated arm holding + 3D pushes
    python scripts/touch_gui.py --fake --teleop           # simulated leader moving it around
    python scripts/touch_gui.py                           # real follower holds its pose: push it
    python scripts/touch_gui.py --teleop                  # leader drives follower, push the follower
    python scripts/touch_gui.py --teleop --record         # record no-touch motion for fit_model.py
    python scripts/touch_gui.py --teleop --model models/load_model.json   # touch while MOVING

Keys (window focused): R = re-learn the baseline now (don't touch while it learns).
(R also toggles MuJoCo's floor reflection: harmless.)

States: LEARNING (don't touch) -> WATCHING (still, most sensitive) -> TOUCH.
While the command moves: MOVING (with --model), or "no model" (no detection while moving).
"""
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from _common import RateMeter, base_parser, open_and_report

from tactile import config
from tactile.dynmodel import LoadModel
from tactile.estimator import TouchEstimator
from tactile.logger import CsvLogger
from tactile.model import SO101Model
from tactile.teleop import FakeLeader, LeaderArm, Teleop

p = base_parser(__doc__)
p.add_argument("--teleop", action="store_true", help="drive the follower from the leader arm")
p.add_argument("--leader-port", default=config.LEADER_PORT)
p.add_argument("--model", default=None, help="LoadModel json from fit_model.py (touch while moving)")
p.add_argument("--record", action="store_true", help="label the log as no-touch motion for fit_model.py")
p.add_argument("--torque-limit", type=int, default=500, help="follower Torque_Limit, 0..1000")
p.add_argument("--max-step", type=float, default=40, help="teleop: max goal change per tick, ticks")
p.add_argument("--z-on", type=float, default=6.0)
p.add_argument("--z-off", type=float, default=3.0)
p.add_argument("--z-loc", type=float, default=2.0, help="joints above this |z| count when picking the link")
p.add_argument("--no-truth", action="store_true", help="fake: hide the true-push arrow")
p.add_argument("--limp", action="store_true",
               help="never enable torque (mirror check: move the arm by hand). SUPPORT THE ARM")
args = p.parse_args()
if args.limp and args.teleop:
    p.error("--limp and --teleop don't mix (teleop has to drive the follower)")

J = config.JOINTS
src = open_and_report(args, full=False, cartesian=args.fake, pushes=not args.record)  # record = no touch
lm = LoadModel.load(args.model) if args.model else None
if lm:
    print("[tactile] load model R^2:", {j: round(v, 2) for j, v in lm.r2.items()})
est = TouchEstimator(SO101Model(ranges=src.ranges), lm, args.z_on, args.z_off, z_loc=args.z_loc)
leader = None
if args.teleop:
    leader = FakeLeader() if args.fake else LeaderArm(args.leader_port)
tp = Teleop(leader, src.ranges, max_step=args.max_step) if leader else None
name = "free_motion" if args.record else "touch_gui"
log = CsvLogger(name, J, extra=["state", "touch", "contact_link"] + [f"z_{j}" for j in J], fake=args.fake)

lock = threading.Lock()
shared = {"pos": None, "out": None, "truth": None, "hz": 0.0, "status": {}}
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
    rate = RateMeter()
    n = 0
    status = {}
    while not stop.is_set():
        s = src.read()
        if tp:
            src.set_goal(tp.goals(s.t, s.pos))
        if n % 10 == 0:
            status = {j: v for j, v in src.read_status().items() if v}
        n += 1
        if relearn.is_set():
            est.relearn()
            relearn.clear()
        o = est.update(s.t, dict(src.goals), s.pos, s.load)
        log.write(s, {"state": o.state, "touch": int(o.active.any()),
                      "contact_link": o.contact.link if o.contact else "",
                      **{f"z_{j}": round(float(o.z[k]), 2) for k, j in enumerate(J)}})
        with lock:
            shared.update(pos=dict(s.pos), out=o, hz=rate.tick(), status=status,
                          truth=getattr(src, "true_push", None))


# ---------------------------------------------------------------------- drawing
vm = SO101Model(ranges=src.ranges)          # the viewer's own model + data
m, d = vm.m, vm.d
orig_rgba = m.geom_rgba.copy()
visual = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
RED, ORANGE, CYAN = np.array([1, .1, .1, 1]), np.array([1, .55, 0, 1]), np.array([0, .9, 1, .9])


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
    a = min(max((abs(z) - args.z_off) / (args.z_on * 1.5 - args.z_off), 0), 1)
    if abs(z) < args.z_off:
        return np.array([.6, .6, .6, .5])
    return np.array([1, 1 - a, 0, .95])


def draw(viewer, pos, o, truth, hz, status):
    d.qpos[:] = vm.raw_to_qpos(pos)
    mujoco.mj_forward(m, d)
    m.geom_rgba[:] = orig_rgba
    scn = viewer.user_scn
    scn.ngeom = 0
    if o is not None:
        for k, j in enumerate(J[:5]):                 # torque arrows along each joint axis
            jid = vm.jid[k]
            tau = o.tau_nm[k] * config.JOINT_SIGN[j]        # +tick direction -> model axis
            L = float(np.clip(abs(tau) / 0.4 * 0.08, 0, 0.12))
            axis = d.xaxis[jid] * np.sign(tau)
            add_arrow(scn, d.xanchor[jid], d.xanchor[jid] + axis * L, heat(o.z[k]), 0.004)
        c = o.contact
        if c is not None:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c.link)
            for g in visual:
                if m.geom_bodyid[g] == b:
                    m.geom_rgba[g] = [1, .25, .25, 1]
            if c.ambiguous:
                add_sphere(scn, c.point, 0.035, np.array([1, .55, 0, .25]))
                force_arrow(scn, c.point, c.force, ORANGE)
            else:
                add_sphere(scn, c.point, 0.01, RED)
                force_arrow(scn, c.point, c.force, RED)
    if truth is not None and not args.no_truth:
        _, tp_, tf = truth
        add_sphere(scn, tp_, 0.007, CYAN)
        force_arrow(scn, tp_, tf, CYAN)

    mode = ("FAKE " if args.fake else "") + ("TELEOP" if args.teleop else "LIMP (mirror check)" if args.limp
                                               else "HOLD")
    if args.record:
        mode += "  RECORDING no-touch motion"
    state = o.state if o else "starting"
    big = state
    if o is not None and o.contact is not None:
        c = o.contact
        fN = np.linalg.norm(c.force)
        big = f"TOUCH  {c.link}  ~{fN:.1f} N" + ("  (link only: point/direction a guess)" if c.ambiguous else "")
    if status:
        big = f"SERVO STATUS ERROR {status}"
    left = "\n".join(J)
    right = "\n".join(
        f"{o.tau_nm[k]:+.2f} N.m  z{o.z[k]:+6.1f}  " + "|" * int(min(abs(o.z[k]), 30)) if o else "-"
        for k in range(len(J)))
    viewer.set_texts([
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "mode\nstate\nrate",
         f"{mode}\n{state}\n{hz:.0f} Hz"),
        (mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOP, big, ""),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, left, right),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
         "~ = uncalibrated (load/1000 x 2.94 N.m)\nR = re-learn baseline", ""),
    ])


def on_key(keycode):
    if keycode == ord("R"):
        relearn.set()


th = threading.Thread(target=worker, daemon=True)
t_end = time.time() + args.duration if args.duration else None
try:
    th.start()
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.15, 0.0, 0.15]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.85, 135, -22
        while viewer.is_running() and th.is_alive():
            if t_end and time.time() > t_end:
                break
            with lock:
                pos, o, truth = shared["pos"], shared["out"], shared["truth"]
                hz, status = shared["hz"], shared["status"]
            if pos is not None:
                with viewer.lock():
                    draw(viewer, pos, o, truth, hz, status)
                viewer.sync()
            time.sleep(1 / 40)
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
    if args.record and not worker_error:
        print(f"\nNext: python scripts/fit_model.py {log.path}")
if worker_error:
    raise SystemExit(f"[tactile] the sensing loop crashed: {worker_error[0]!r} (traceback above)")
