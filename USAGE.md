# Usage: every command, in order

Everything runs from the repo root in the `lerobot` conda env:

```bash
conda activate lerobot
cd ~/Desktop/personal_projects/github/vinea-so101/torque-tactile-sensing
```

Every script opens a window. Every script has `--fake` (a simulated arm, no hardware
needed) and `--duration N` (stop after N seconds). Every run writes a **new**
`data/<script>_<timestamp>.csv`; nothing is ever appended or overwritten.

---

## 1. Test it by itself (no arm needed)

Start here. The simulated arm behaves like the real one: integer encoder ticks with
jitter, a P servo whose stiffness follows its gain, gear friction, the Present_Load dead
band. A simulated fingertip presses random points of the real surface (1–5 N).

```bash
# The touch GUI: heat cloud on the surface where the touch can be, red ball + arrow at
# the most likely point, cyan = the TRUE simulated push, and the error in cm on top
python scripts/touch_gui.py --fake
python scripts/touch_gui.py --fake --soft 4        # softer servos (see §2)
python scripts/touch_gui.py --fake --teleop        # simulated leader moving it around

# Accuracy in numbers: random poses x random pushes, cm error per link (no window, ~6 min)
systemd-run --user --scope -p MemoryMax=3G python scripts/bench_touch.py

# Plain signal plots and the simpler detectors from step 0/1
python scripts/live_view.py --fake
python scripts/push_detect.py --fake

# Unit tests (the -u keeps the ROS install's pytest plugins out)
env -u PYTHONPATH python -m pytest -q tests
```

---

## 2. The real arm on its own (follower holds still, you push it)

Plug in the follower (white arm, 12 V PSU on). The leader is not needed. Close
`lerobot-teleoperate` first: only one program can open the follower's port.

**Hold the arm up in free space before starting.** If a joint rests on its end stop
(the folded rest pose does this for shoulder_lift and elbow), the stop takes the push
instead of the motor. The GUI shows an END STOP warning when that happens.

```bash
python scripts/touch_gui.py                    # holds its current pose at Torque_Limit 500
python scripts/touch_gui.py --soft 4           # softer servos: finer touch, weaker hold
```

1. It starts in **LEARNING** (about 2 s). **Don't touch it** while it learns what
   "nothing touching me" looks like.
2. **WATCHING**: push the arm with a finger. Joint arrows light up, a heat cloud appears
   on the surface, and the red ball + arrow show the most likely point and push direction.
   The top line says `TOUCH <link> +-<cm> ~<N>`. When physics can't decide between links
   it also lists their probabilities.
3. Keep pressing: the estimate sharpens while the same touch lasts, fastest if your push
   direction wobbles a little.
4. Press **R** in the window to re-learn, e.g. after moving it by hand.

`--soft P` lowers the servos' P gain (lerobot uses 16) while the GUI runs. A push then
deflects each joint more ticks, so the signal is finer, and weaker pushes are felt. Each
goal is re-aimed so the arm keeps its pose as P drops, and P is restored (with the same
re-aim) on exit. With torque on, the servo keeps the new P in RAM only; lerobot also
rewrites P=16 every time it connects. In the simulation `--soft 4` catches more pushes and
is sharper on the wrist, but less sure on the upper arm (§4).

### First-time checks (once)

**A. Mirror check.** Re-do it after any recalibration:

```bash
python scripts/touch_gui.py --limp     # never enables torque: move it by hand. SUPPORT the arm
```

Move one joint at a time by hand and watch the model.
- Moves the **wrong way**: flip that joint in `tactile/config.py` → `JOINT_SIGN`.
- Moves the right way but sits at a **constant angle off**: set `JOINT_OFFSET_TICKS`
  (4096 ticks = 360°, so 11 ticks ≈ 1°).
- Gripper jaw opens when the real one closes: `GRIPPER_MODEL_FLIP = True`.

**B. Push-direction check.** The force direction now comes from physics (a push toward
+ticks lowers `err`), not from a sign setting, so this check should pass as is. Hold mode,
wait for WATCHING, rotate the view to match where you stand, push the gripper sideways:
the red arrow must point the way your finger pushes. If a single joint's arrow is
backwards, its `JOINT_SIGN` is wrong (that would also fail check A).

**C. Poke test: how accurate is it really?**

```bash
python scripts/touch_gui.py --poke              # or --poke --soft 4, to compare
```

1. Double-click the 3D model where you will poke. A green dot marks the spot.
2. Poke that spot on the real arm, ~1 s each, 5–10 times. The terminal prints each poke's
   error in cm and the median for that spot.
3. Double-click a new spot, repeat. Cover every link: gripper, jaw, wrist, forearm, upper
   arm, shoulder.
4. Ctrl+C. It prints the overall median error and link accuracy and saves
   `data/poke_results_<stamp>.json`.

Send the summary back: it is the only honest accuracy number for the real arm, and it
tells whether `FRICTION_NM` / `TAU_FLOOR_NM` in `tactile/config.py` need tuning.

---

## 3. With teleoperation (leader drives, you push the follower)

`lerobot-teleoperate` cannot run at the same time, because it owns the follower's port.
`touch_gui.py --teleop` **replaces it**. It reads the leader and drives the follower
itself (same mapping as lerobot with `use_degrees=false`), and senses touch in the same
loop.

```bash
python scripts/touch_gui.py --teleop
```

- The leader is only ever read; its torque is never touched.
- The follower ramps to the leader pose over 2 s (no snap). Each tick moves at most
  `--max-step 40` ticks, and every goal is clamped to the calibrated range.
- Both arms use the by-id ports in `tactile/config.py`, never `/dev/ttyACM*`.
- `--soft` is not allowed here: a soft arm sags off the leader's pose.

What you'll see:
- **When you pause the leader**, the follower is still → LEARNING → WATCHING, with
  hold-mode sensitivity. Keep the leader still with one hand and poke the follower
  with the other. This is the most sensitive way to use it.
- **While you move the leader**, without a model it says "MOVING (no model)" and does
  not detect. Gravity and friction change with every motion, so it needs the model below.

### Touch detection while moving: record → fit → use

```bash
# 1. 2-3 minutes of teleop where NOBODY touches the follower. Cover the whole workspace,
#    slow and fast, every joint. The window shows RECORDING.
python scripts/touch_gui.py --teleop --record

# 2. Fit expected position error while moving (step 1 prints this command at the end).
#    The window shows measured vs predicted err per joint.
python scripts/fit_model.py data/free_motion_<stamp>.csv

# 3. Teleop with touch detection during motion
python scripts/touch_gui.py --teleop --model models/load_model.json
```

The model uses only the **commanded** motion (leader positions and their speed and
acceleration), never the follower's measured position. A push shows up in that, so a
model that used it would learn to "predict" your hand. Re-fit after you change
`--torque-limit`, P gain or hardware (e.g. the fin-ray gripper). An old fit is kept as
`load_model_prev.json`. Models fitted before 2026-10-15 predicted Present_Load and are
refused: re-record.

---

## 4. How it works, and what it can and can't know

**The signal.** Each servo's position error `err = goal - pos`. The servo is a P
controller, so at rest external torque = −K × (err − err at rest), K ≈ 0.026 N·m per tick
at P=16 (4× less at P=4). Present_Load turned out to be ≈ −9 × err with a dead band
(it reads 0, then jumps to ±20), so it carries less; Present_Current tops out at 1–7 raw
units, useless. err is averaged over 0.1 s, which resolves fractions of a tick.

**Detection.** One number over all joints, `d = sqrt(sum (torque_j / noise_j)^2)`, on/off
at `--d-on 6` / `--d-off 3.5`. Only the part of each joint's error beyond its **rest
band** counts: after a push a servo can settle anywhere inside its 1-tick dead zone plus
what gear friction holds (`DEAD_ZONE_TICKS`, `FRICTION_NM` in `tactile/config.py`).
Without that band a released push looked like a touch that never ended. A "touch" that
stays perfectly steady for 5 s is re-learned as the new baseline (a finger is never that
steady).

**Where.** `tactile/contact.py` scores ~4600 points covering the real surface of every
moving link (5 mm apart, from the meshes; points inside the arm or facing a neighbouring
part are dropped). For each point it fits the force a fingertip could apply there, which
can only **push into the surface** (a friction cone around the inward normal), and asks how
well `torque = J(point)^T force` explains all six joints, including the ones reading ~0.
That gives a probability per point: the heat cloud. Evidence adds up while one touch lasts.

**The physics ceiling** (no algorithm can beat it with joint torques alone):

| touched | joints that feel it | what can be known |
|---|---|---|
| shoulder | shoulder_pan only | that it's the shoulder (or a sideways push further out) |
| upper arm | pan + lift | a curve of points; the link is usually right |
| forearm | pan + lift + elbow | roughly the point |
| wrist, gripper, jaw | 4–6 joints | the point and the push direction |

Pushes **parallel to the pitch axes** (shoulder_lift, elbow, wrist_flex are parallel) give
those joints nothing, wherever they land, so they all look like shoulder pushes. The heat
cloud and the `+-cm` show the uncertainty honestly instead of hiding it.

**Measured on the simulated arm** (`bench_touch.py`, 40 random 1–5 N pushes per setting,
gear friction 0.005–0.02 N·m; median error of the most likely point):

| | P=16 (default) | P=4 (`--soft 4`) |
|---|---|---|
| pushes detected | 78–80 % | 88 % |
| false touches / stuck touches | 0 / 0 | 0 / 0 (19/40 stuck at 0.02 N·m friction) |
| gripper + jaw | ~1 cm | 1–2 cm |
| wrist | 2.5–3 cm | 1–2 cm |
| forearm | ~4 cm | ~4 cm |
| upper arm | ~4 cm, link right 83 % | ~5 cm, link right 38–50 % |

The real arm's friction is not measured yet; the poke test (§2C) gives the real numbers.

---

## 5. All commands

| command | what |
|---|---|
| `touch_gui.py [--fake] [--teleop] [--model F] [--record]` | the 3D touch GUI (hold or teleop) |
| `touch_gui.py --soft P` | hold mode with softer servos (P gain, lerobot 16), restored on exit |
| `touch_gui.py --poke` | accuracy test: double-click a spot, poke it, cm error per poke |
| `touch_gui.py --torque-limit N --max-step N --d-on D --d-off D` | tuning: hold strength, teleop speed cap, detection thresholds |
| `bench_touch.py [--p 16 4] [--friction ...] [--trials N]` | accuracy on the simulated arm, per link |
| `fit_model.py CSV... [--out F]` | fit expected err while moving from no-touch recordings |
| `live_view.py [--hold arm]` | raw load / current / position error plots |
| `push_detect.py [--mode hold\|wiggle] [--react none\|stop\|comply] [--signal load\|cur\|err]` | 2D detector, collision stop, lead-through |
| `probe_gripper.py` | which raw end of the gripper is closed (for `squeeze.py`) |
| `squeeze.py` | gripper squeeze test (tomato work, parked for now) |
| any `--release-on-exit` | torque OFF at the end. The arm goes limp and drops, so **support it** |
| any `--port P` / `--leader-port P` | override the by-id ports |

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| `Follower not found at /dev/serial/by-id/...` | not plugged in / PSU off. `ls -l /dev/serial/by-id/` |
| `Could not connect` / port busy | `lerobot-teleoperate` or another script still has the port |
| `END STOP` warning | a joint rests on its mechanical stop: lift the arm into free space, press R |
| constant TOUCH right after LEARNING | you touched it while learning, or it was still settling: press **R** |
| touch never ends after a push | friction bigger than `FRICTION_NM`: raise it in `tactile/config.py` (or wait 5 s) |
| touches while moving, nobody touching | no / old model. Record more varied motion and re-fit |
| arrow points backwards | that joint's `JOINT_SIGN` (mirror check A) |
| model moves differently from the arm | mirror check (A): `JOINT_SIGN` / `JOINT_OFFSET_TICKS` |
| `servo did not take P_Coefficient` | `--soft` could not set the gain: run without `--soft` |
| `--soft refused, nothing changed: ...` | a joint is on / near its end stop or far from its goal. Hold the arm up in free space and restart (softening there would make it drive off once lifted) |
| `SERVO STATUS ERROR` banner | overload / heat / voltage protection tripped; the servo goes soft. Lower `--torque-limit`, let it cool |
| window frozen, terminal still logging | the window isn't being drawn (hidden / other workspace / screen locked). Bring it to front; Ctrl+C works either way |
| Wayland `GLFWError: window position` warning | cosmetic, the window still opens |
| `Failed to sync read ... no status packet` / `Incorrect status packet` | another program is using the arm (`~/teleop.sh` / `lerobot-teleoperate`, another touch_gui): close it. Otherwise just run it again |
| low Hz in the banner | each register read is a bus round trip; touch_gui reads 3 per tick + Status every 10th |
