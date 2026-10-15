# torque-tactile-sensing

Touch for the SO-101 **without adding sensors**: read what the STS3215 servos already
report and turn it into "someone is touching me, here, this hard, this way" (and, parked
for later, "I am squeezing a tomato this hard").

- **[USAGE.md](USAGE.md)**: every command. Testing without the arm, the real arm on its
  own, the poke accuracy test, teleoperation, how it works and what physics allows.
- **[IDEAS.md](IDEAS.md)**: the physics, the idea ladder, and what's next.

Current focus: a sense of touch for the arm (where am I touched, which way, how hard).
The tomato / grip-force work is parked for now (`squeeze.py` still works).

## The touch GUI

```bash
conda activate lerobot
python scripts/touch_gui.py --fake               # no arm needed: simulated finger pushes
python scripts/touch_gui.py --dither 4           # real follower holds still: touch it
python scripts/touch_gui.py --teleop --dither 4 --poke   # pose with the leader, then measure accuracy
python scripts/touch_gui.py --teleop             # leader drives it (replaces lerobot-teleoperate)
python scripts/touch_gui.py --replay data/poke_<stamp>.csv   # a recorded run, re-analysed in 3D
```

A MuJoCo window shows the SO-101 mirroring the real arm:
- arrows on each joint axis for the torque it feels
- a **heat cloud on the arm's surface**: every point where a fingertip could be pressing
  to produce exactly these torques. Small cloud = it knows where. Smeared = physics can't
  tell (proximal links, see USAGE.md §4)
- a red ball + arrow at the most likely point: where, and which way you push, with
  `+-cm` and `~N` on top
- in `--fake`, the true push in cyan and the error in cm

How: each servo's position error is a torque sensor (it is a P controller). Every point of
the real link surfaces is scored by how well a push-only, finger-sized force there explains
all six joint torques (`τ = Jᵀ F`). The real limit is gear friction (~0.05 N·m per joint,
measured from recordings): it hides light pokes and parks joints off after every poke.
The estimator now undoes that parking, and `--dither` (goals shaking ±4 ticks) makes the
friction average out. On a simulation calibrated to the real arm: ~75 % right link,
3.5–4 cm median error, no stuck touches (`scripts/bench_touch.py`, USAGE.md §4). The real-arm
number comes from a poke test.

**Improving it without the arm:** `scripts/replay_eval.py` replays any recorded run
through the current code (time in TOUCH, touches, poke error per poke), and
`touch_gui.py --replay` shows it in 3D. Poke runs log their targets in every row.

## Older step-by-step tools (watch first)

Everything runs in the `lerobot` conda env. Every script opens a live window and
works with `--fake` (a simulated arm) so you can try it without the arm plugged in.

```bash
conda activate lerobot
cd ~/Desktop/personal_projects/github/vinea-so101/torque-tactile-sensing

# 0. look at the raw signals. Fake arm gets random "hand pushes"
python scripts/live_view.py --fake

# 1. push detection: hold still, learn, then flag pushes (red banner)
python scripts/push_detect.py --fake
python scripts/push_detect.py --fake --mode wiggle --joint elbow_flex --react stop

# 2. gripper squeeze: contact, stiffness, stop at target, back off on crush
python scripts/squeeze.py --fake
python scripts/squeeze.py --fake --torque-limit 600 --target 500 --max-lead 400   # fake tomato gets crushed
```

### On the real arm (follower, 12 V PSU on)

The scripts only talk to the **follower** (the white arm) on its by-id port
(`tactile/config.py`). Nothing can share the serial port with `lerobot-teleoperate`,
so close teleop first.

```bash
# torque untouched, just watch (move the limp arm by hand)
python scripts/live_view.py

# hold the pose at 30 % torque, then push it with your hand
python scripts/live_view.py --hold arm --torque-limit 300
python scripts/push_detect.py --torque-limit 300                      # don't touch for the first 3 s
python scripts/push_detect.py --signal err                            # try load / cur / err, compare
python scripts/push_detect.py --mode wiggle --joint shoulder_pan --amp 150 --period 4 --react stop
python scripts/push_detect.py --react comply                          # lead it around by hand

# gripper: FIRST find which raw end is closed, write it into tactile/config.py
python scripts/probe_gripper.py
python scripts/squeeze.py --torque-limit 250 --target 150 --speed 80  # sponge / foam first!
```

Every run writes a **new** `data/<script>_<timestamp>.csv` with every register for all
joints, never appending to an old one. `data/` is gitignored. `--no-window` logs without a
window, and `--duration N` stops after N seconds.

### Safety, read once

- `--torque-limit` writes the servo's **Torque_Limit** (RAM, 0..1000 = 0..100 %). The
  servo enforces it itself, so it is the real force ceiling. It is restored on exit
  and resets on power-up anyway.
- At startup each script prints `Torque_Limit`, `Max_Torque_Limit`, `Protection_Current`,
  `Overload_Torque`, `P_Coefficient`, `Minimum_Startup_Force` and the dead zones. A bare bus does not run lerobot's
  `configure()`, so lerobot's gripper caps (50 %) and P=16 are only there if the EEPROM
  has them. Check the printout.
- All goals are raw ticks clamped to `range_min/range_max` from
  `follower_white.json`. There is no lerobot normalisation, so the `use_degrees` trap
  doesn't apply.
- `touch_gui.py --soft P` writes `P_Coefficient` while torque is on (Lock=1, so RAM
  only), re-aims the goals so the pose holds, and restores the original P on exit.
  lerobot also rewrites P=16 every time it connects.
- On exit torque stays **on** (the arm holds instead of dropping). `--release-on-exit`
  makes it go limp, so support the arm.
- A non-zero `Status` register (overload/heat/voltage trip) is shown in purple/red. A
  tripped servo goes soft, which looks exactly like a slip or a crushed fruit.

## Branches

- `dev`: work in progress. Commit and push here.
- `main`: finished, working code. Changes only by PR from `dev`. It's protected: no
  direct pushes, no force-pushes, no deletion. The `only-dev-into-main` check fails
  any PR into `main` from another branch.

## Layout

```
assets/so101/    MuJoCo SO-101 model (Menagerie, Apache-2.0, see SOURCE.md)
tactile/
  config.py      ports, joint ids, ranges, raw<->model signs/offsets, err->N.m (K per P gain),
                 rest band (dead zone + friction), touchable bodies
  source.py      ArmSource (real bus, raw units, --soft P gain with re-aim) + FakeSource (2D scripts)
  sim.py         ServoArmSim / SimArmSource: simulated follower calibrated to the real one
                 (encoder flicker, 0.05 N.m friction, dead zone) + fingertip pushes
  replay.py      ReplaySource: a recorded CSV played back as if it were the arm
  envelope.py    SafetyEnvelope: hard +-1.5 % / abort 1.8 % motion limit for self_calibrate
  surface.py     touchable surface points + normals, sampled from the link meshes
  contact.py     ContactLocalizer: push-only force fit at every surface point -> probability
  model.py       MuJoCo model: raw->angles, gravity, Jacobians, servo<->model torque
  estimator.py   TouchEstimator: LEARNING / WATCHING / MOVING / TOUCH on windowed err
  dynmodel.py    expected err while moving, from COMMANDED motion only (fit_model.py)
  teleop.py      leader -> follower teleop in our own loop (ramp, step cap, clamp)
  detect.py      Baseline, PhaseBaseline, Hysteresis, analyse_squeeze (pure numpy, tested)
  liveplot.py    threaded matplotlib window (the bus loop never waits for drawing)
  logger.py      one new timestamped CSV per run
scripts/
  touch_gui.py   the 3D touch GUI: hold / soft / poke / teleop / record, --fake
  bench_touch.py localisation accuracy per link on the simulated arm (no window)
  replay_eval.py the current code on a recorded run: TOUCH time, touches, poke cm error
  self_calibrate.py  per-joint jitter + friction band from tiny moves (sim-tested only)
  fit_model.py   fit the moving-arm err model from a no-touch recording
  live_view.py   step 0, look at signals (optionally hold joints)
  push_detect.py step 1, hold / wiggle push detection, stop / comply reactions
  probe_gripper.py  which raw end of the gripper is closed
  squeeze.py     step 2, contact, stiffness, target grip, yield (crush) back-off
tests/           detectors, estimator, contact localisation, teleop + motion model
```

Tests: `env -u PYTHONPATH python -m pytest -q tests`. The `-u PYTHONPATH` keeps the
ROS install's pytest plugins out.

## Status

- Verified:
  - all unit tests
  - every script end to end on `--fake`, with a window and headless
  - on the real follower (2026-10-15): mirror check, loop rate ~320 Hz in hold mode /
    ~240 Hz in teleop, P_Coefficient 16, gripper Torque_Limit 500, Present_Current
    unsigned and only 1–7 raw units (unused)
  - from the real recordings: readings sit on one integer for seconds (~5 % flicker),
    static gear friction ~0.05 N·m per joint, light pokes move a joint only 1–4 ticks.
    The simulation is calibrated to these numbers.
  - the real recordings replayed through the current code: time in TOUCH 81 % → 12 %
    (rest-pose hold), 49 % → 28 % (hold), 36 % → 3 % (poke run), with the same number of
    real touches and no 15–40 s stuck ones
- First real poke test (2026-10-15, P=16, elbow and gripper resting on their stops):
  median 12 cm, link right 27 %. Its targets had to be paired with touches by order and
  the old estimator's stuck touches got some of them, so it can't score the new code
  reliably (5–8 cm on replay). Poke runs now log their targets per row.
- Not yet on the real arm: `--dither`, `--soft`, `self_calibrate.py`, a clean poke test.
