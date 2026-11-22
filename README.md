# torque-tactile-sensing

Touch for the SO-101 arm **without adding sensors**. The STS3215 servos in the arm already
report how far each joint is from where it was told to be. This repo turns those readings
into "someone is touching me, here, this hard, this way", shown live on a 3D model of the
arm. It is meant for anyone with an SO-101 follower (and optionally a leader) who wants a
rough sense of touch on the arm for free.

> **Licence: there is no licence file yet.** The code is public to read, but until a
> licence is added no permission to use, copy, modify or redistribute it is granted.
> The SO-101 model in `assets/so101/` is from MuJoCo Menagerie and stays under its own
> Apache-2.0 licence.
>
> **Hardware status:** the touch GUI, the mirror check and the signal readings have run on
> a real follower. `--dither`, `--soft`, `self_calibrate.py` and a clean poke test (the
> real accuracy number) are **not yet tested on the real arm**. Accuracy figures below
> come from a simulation calibrated to the real arm, not from the arm itself. Details in
> [Status](#status).

## What it does

- **Touch GUI** (`scripts/touch_gui.py`): a MuJoCo window with the SO-101 mirroring the
  real arm. When something pushes the arm you see
  - an arrow on each joint axis for the torque it feels,
  - a **heat cloud** on the arm's surface: every point where a fingertip could be pressing
    to produce these torques. Small cloud = it knows where. Smeared = physics can't tell,
  - a red ball and arrow at the most likely point and push direction, with `+-cm` and
    `~N` on top.
- **Hold mode** (the arm holds still, you push it) and **teleop mode** (a leader arm drives
  the follower in the same loop, so it replaces `lerobot-teleoperate`).
- **Poke test**: double-click a spot on the model, poke that spot on the real arm, and it
  prints the error in cm per poke.
- **Simulated arm** (`--fake`): try everything without hardware. The simulation copies the
  real servos' quirks (integer encoder ticks, gear friction, dead zone).
- **Replay**: play any recorded run back through the current code, in 3D or as numbers.
- Older, simpler step-by-step tools: raw signal plots, a 2D push detector with stop /
  comply reactions, and a gripper squeeze test (parked work on gripping tomatoes).

## Requirements

- Linux. Tested on Ubuntu with Python 3.12 in a conda env with LeRobot 0.6.2,
  MuJoCo 3.8.1, NumPy 2.2.6, Matplotlib 3.11.1 and pandas 2.3.3. There is no
  `requirements.txt` or `pyproject.toml` yet.
- For the real arm: an SO-101 follower (12 V servos, power supply on) calibrated with
  LeRobot. The leader is only needed for `--teleop`.
- No hardware is needed for `--fake`, `--replay` or the tests.

## Setup

1. **A Python env with LeRobot.** Install LeRobot by its own instructions (with the
   Feetech extra for the SO-101 bus). Then check that the rest is there:

   ```bash
   conda activate lerobot        # or whatever your LeRobot env is called
   python -c "import lerobot, mujoco, numpy, matplotlib, pandas; print('ok')"
   ```

   If one is missing, `pip install` it into the same env. LeRobot is only imported when a
   real arm is opened, so `--fake` works without it.

2. **Get the code.**

   ```bash
   git clone https://github.com/krishaanth5831/torque-tactile-sensing.git
   cd torque-tactile-sensing
   ```

3. **Try it without an arm.** This opens the MuJoCo window with a simulated arm and
   simulated finger pushes (cyan arrow = the true push, the error in cm on top):

   ```bash
   python scripts/touch_gui.py --fake
   ```

4. **Point it at your arm.** Edit the top of `tactile/config.py`:
   - `FOLLOWER_PORT` / `LEADER_PORT`: your arms' stable serial paths. Find them with
     `ls -l /dev/serial/by-id/`. Use these, not `/dev/ttyACM*`, which swap with plug
     order. You can also pass `--port` / `--leader-port` on each run.
   - `CALIBRATION_JSON` / `LEADER_CALIBRATION_JSON`: the calibration files LeRobot wrote
     when you calibrated the arms. The defaults point at
     `~/.cache/huggingface/lerobot/calibration/.../follower_white.json` and
     `leader_black.json`; change the file names to your arms' ids. These give the joint
     ranges every goal is clamped to. The real arm, `--replay`, `replay_eval.py` and
     `bench_touch.py` need the follower file; `--fake` falls back to generic ranges.

5. **Mirror check, once.** With torque off, move each joint by hand and watch the model
   follow. Support the arm: it is limp.

   ```bash
   python scripts/touch_gui.py --limp
   ```

   If a joint moves the wrong way or sits at a constant angle off, fix `JOINT_SIGN` /
   `JOINT_OFFSET_TICKS` in `tactile/config.py` ([USAGE.md §2A](USAGE.md#first-time-checks-once)).

## Quick start

Close `lerobot-teleoperate` and anything else using the arm first: only one program can
use the follower's serial bus.

```bash
python scripts/touch_gui.py --fake                         # no arm: simulated pushes
python scripts/touch_gui.py                                # real follower holds its pose: push it
python scripts/touch_gui.py --dither 4                     # same, goals shake +-4 ticks (not yet tried on the real arm)
python scripts/touch_gui.py --teleop                       # leader drives the follower, push the follower
python scripts/touch_gui.py --teleop --dither 4 --poke     # pose it with the leader, then measure accuracy
python scripts/touch_gui.py --replay data/poke_<stamp>.csv # a recorded run, re-analysed in 3D
```

1. It starts in **LEARNING** for about 2 s. Don't touch the arm while it learns what
   "nothing touching me" looks like.
2. **WATCHING**: push the arm with a finger. The top line says `TOUCH <link> +-<cm> ~<N>`.
3. Press **R** in the window to re-learn, for example after moving the arm by hand.

The arm must be in free space, not resting on an end stop (the folded rest pose rests
shoulder_lift, the elbow and a closed gripper on their stops). The GUI shows an `END STOP`
warning. The easiest way to get a good pose is `--teleop`: move the leader until the
follower is clear, then hold the leader still and touch the follower.

Every command, the poke test, touch detection while moving and troubleshooting are in
**[USAGE.md](USAGE.md)**. The physics and next ideas are in **[IDEAS.md](IDEAS.md)**.

## Safety

These scripts move real servos. Read once:

- `--torque-limit` writes the servo's own Torque_Limit (RAM, 0..1000 = 0..100 %). The servo
  enforces it, so it is the real force ceiling. It is restored on exit and resets on
  power-up anyway.
- All goals are raw ticks clamped to the calibrated range. In teleop the follower ramps to
  the leader pose over 2 s and each tick moves at most `--max-step` ticks. The leader is
  only read; its torque is never touched.
- On exit torque stays **on** (the arm holds instead of dropping). `--release-on-exit`
  makes it go limp, so support the arm.
- `--soft P` lowers the servos' P gain while the GUI runs and restores it on exit. In
  hold mode it refuses to start if a joint rests on an end stop; with `--teleop` it waits
  until the follower is still and clear of its stops.

More detail in [USAGE.md §8](USAGE.md#8-safety-details).

## How it works, in short

Each servo is a P controller, so its position error `err = goal - pos` is a torque sensor:
at rest, external torque ≈ −K × (err − err at rest). Every point of the arm's surface
(~4600 points from the link meshes) is scored by how well a push-only, finger-sized force
there explains all six joint torques (`τ = Jᵀ F`). That gives the heat cloud.

Two limits are physics, not code:
- **Gear friction** (~0.05 N·m per joint, measured from recordings) hides light pokes and
  leaves joints parked a few ticks off after each poke. The estimator undoes that parking,
  and `--dither` (goals shaking ±4 ticks) keeps the joints moving so the friction averages
  out.
- **Joints near the base** can't locate a touch well: a push on the shoulder or upper arm
  moves only one or two joints. Pushes parallel to the pitch axes give those joints
  nothing at all.

On the simulation calibrated to the real arm (`scripts/bench_touch.py`): ~75 % right link,
3.5–4 cm median error, no stuck touches with `--dither 4`. The real-arm number needs a
clean poke test. Full tables in [USAGE.md §4](USAGE.md#4-how-it-works-and-what-it-can-and-cant-know).

## Recordings

Every run writes a **new** `data/<script>_<timestamp>.csv` with every register for every
joint; nothing is appended or overwritten. `data/` is gitignored. Poke runs log their
target in every row, so `scripts/replay_eval.py` can score a run again after any code
change, without the arm.

## Project layout

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
  touch_gui.py   the 3D touch GUI: hold / soft / poke / teleop / record / replay, --fake
  bench_touch.py localisation accuracy per link on the simulated arm (no window)
  replay_eval.py the current code on a recorded run: TOUCH time, touches, poke cm error
  self_calibrate.py  per-joint jitter + friction band from tiny moves (sim-tested only)
  fit_model.py   fit the moving-arm err model from a no-touch recording
  live_view.py   step 0, look at signals (optionally hold joints)
  push_detect.py step 1, hold / wiggle push detection, stop / comply reactions
  probe_gripper.py  which raw end of the gripper is closed
  squeeze.py     step 2, contact, stiffness, target grip, yield (crush) back-off
  _common.py     shared argparse flags (--fake, --port, --duration, ...) and arm opening
tests/           detectors, estimator, contact localisation, replay, teleop + motion model
data/            your recordings (gitignored)
```

## Tests

```bash
env -u PYTHONPATH python -m pytest -q tests
```

The tests need no hardware. `env -u PYTHONPATH` keeps a ROS install's pytest plugins
out; drop it if you have no ROS.

## Status

- Verified:
  - all unit tests
  - every script end to end on `--fake`, with a window and headless
  - on a real follower (2026-10-15): mirror check, loop rate ~320 Hz in hold mode /
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
- **Not yet tested on the real arm:** `--dither`, `--soft`, `self_calibrate.py`, a clean
  poke test, touch detection while moving (`--model`).

## Branches

- `dev`: work in progress.
- `main`: finished, working code. Changes only by PR from `dev`. It's protected, and the
  `only-dev-into-main` check fails any PR into `main` from another branch.

## Licence

No licence file yet, so no permission to reuse the code is granted (see the note at the
top). `assets/so101/` is Apache-2.0, see `assets/so101/LICENSE`.
