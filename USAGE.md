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

## 1. Without the arm: simulation and your own recordings

**The simulated arm is calibrated to the real follower** (2026-10-15 recordings): integer
encoder ticks that sit still for seconds (only ~5 % of readings flicker), a P servo whose
stiffness follows its gain, ~0.05 N·m of static gear friction per joint (after a poke a
real joint stays parked up to 2–3 ticks off), the servo's 1-tick dead zone, the
Present_Load dead band. A simulated fingertip presses random points of the real surface
(1–5 N).

```bash
# The touch GUI: heat cloud on the surface where the touch can be, red ball + arrow at
# the most likely point, cyan = the TRUE simulated push, and the error in cm on top
python scripts/touch_gui.py --fake
python scripts/touch_gui.py --fake --dither 4      # with dither (see §2)
python scripts/touch_gui.py --fake --teleop        # simulated leader moving it around

# Any recorded run, played back through the CURRENT code (green dot = poke target, if logged)
python scripts/touch_gui.py --replay data/poke_<stamp>.csv --replay-speed 2

# Numbers on recorded runs: time in TOUCH, touches, and for poke runs the cm error per poke
python scripts/replay_eval.py data/poke_<stamp>.csv              # add --variants to compare settings

# Accuracy on the simulated arm, per link (no window; 2 earlier pokes leave friction offsets)
systemd-run --user --scope -p MemoryMax=3G python scripts/bench_touch.py --p 16 --dither 0 4 --window 0.25

# Plain signal plots and the simpler detectors from step 0/1
python scripts/live_view.py --fake
python scripts/push_detect.py --fake

# Unit tests (the -u keeps the ROS install's pytest plugins out)
env -u PYTHONPATH python -m pytest -q tests
```

---

## 2. The real arm on its own (follower holds still, you push it)

Plug in the follower (white arm, 12 V PSU on). The leader is not needed. Close
`lerobot-teleoperate` / `~/teleop.sh` first: only one program can use the follower's bus.

**The arm must be in free space, not on an end stop.** If a joint rests on its stop (the
folded rest pose does this for shoulder_lift, the elbow and the closed gripper), the stop
takes pushes in that direction and the joint can only report pushes that lift it off. The
GUI shows an END STOP warning. Hold mode keeps whatever pose the arm starts in, so the
easiest way to get a good pose is to let the leader set it: `--teleop`, move the leader so
the follower is in free space (elbow not folded, gripper a little open), then hold the
leader still and touch the follower.

```bash
python scripts/touch_gui.py --dither 4         # recommended: holds its pose, goals shake +-4 ticks
python scripts/touch_gui.py                    # no dither (P=16)
python scripts/touch_gui.py --soft 4           # softer servos (arm must already be off its stops)
python scripts/touch_gui.py --teleop --dither 4 --soft 4    # pose it with the leader first
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

**`--dither A`** (try 4): each goal shakes ±A ticks at `--dither-hz` (8). The real arm's
gear friction (~0.05 N·m) otherwise holds light pokes without the joint moving and parks
joints off after every poke; kept moving, the friction averages out. On the calibrated
simulation it removes every stuck touch, ends a touch 0.18 s after release (0.7 s without)
and catches more pushes (§4). Sensing uses the unshaken goal, joints resting on a stop are not shaken, and the
arm hums faintly (±4 ticks ≈ ±0.35°). **Not tried on the real arm yet.**

**`--soft P`** lowers the servos' P gain (lerobot uses 16) while the GUI runs, so a push
deflects each joint more ticks. In hold mode it softens at start, re-aiming each goal so
the arm keeps its pose; it refuses (before opening the window) if a joint is on its end
stop. With `--teleop --soft 4` it softens as soon as the follower is still and clear of its
stops (the banner says `WAITING: <joint> on its end stop` until then); a soft follower then
sags a little below the leader. P is restored on exit; with torque on the servo keeps the
new P in RAM only, and lerobot rewrites P=16 every time it connects. Note: the friction is
the same in N·m at any P, so `--soft` alone helps less than `--dither`.

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

**B. Push-direction check.** The force direction comes from physics (a push toward +ticks
lowers `err`), not from a sign setting. Hold mode, wait for WATCHING, rotate the view to
match where you stand, push the gripper sideways: the red arrow must point the way your
finger pushes. If a single joint's arrow is backwards, its `JOINT_SIGN` is wrong (that
would also fail check A).

**C. Poke test: how accurate is it really?**

```bash
python scripts/touch_gui.py --teleop --dither 4 --poke     # pose it with the leader, hold the leader still
python scripts/touch_gui.py --poke                         # hold mode, arm already in free space
```

1. Double-click the 3D model where you will poke. A green dot marks the spot.
2. Poke that spot on the real arm firmly, ~1 s each, 5–10 times. The terminal prints each
   poke's error in cm and the median for that spot.
3. Double-click a new spot, repeat. Cover every link: gripper, jaw, wrist, forearm, upper
   arm, shoulder.
4. Ctrl+C. It prints the overall median error and link accuracy and saves
   `data/poke_results_<stamp>.json`.

Every row of the poke CSV logs the current target, so `replay_eval.py` can score the run
again after any code change, without the arm. (The 2026-10-15 poke run predates this: its
targets had to be paired with touches by order, and the old estimator's stuck "touches"
got some of them, so its numbers are unreliable.)

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
- `--soft 4` softens the follower once it is still and clear of its stops; it then sags a
  little below the leader. `--dither` works here too.

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
at P=16 (4× less at P=4). Present_Load is ≈ −9 × err with a dead band (0, then ±20), so it
carries less; Present_Current tops out at 1–7 raw units, useless. err is averaged over
0.1 s (two dither periods with `--dither`). On the real arm the readings sit on one integer
for seconds, so averaging buys little without dither.

**What limits it: gear friction.** About 0.05 N·m per joint on the real follower (from the
recordings). A poke smaller than that doesn't move the joint at all, and after any poke
the joint stays parked up to `friction / K` ticks off (about 2 ticks at P=16). Light real
pokes moved joints only 1–4 ticks, so friction is the same size as the signal. Dither is
the fix (it keeps every joint moving, so the friction averages out); a softer P gain alone
doesn't change the friction in N·m.

**Detection.** One number over all joints, `d = sqrt(sum (torque_j / noise_j)^2)`, on/off
at `--d-on 6` / `--d-off 3.5`, where each joint only counts beyond its **rest band**
`max(1-tick dead zone, FRICTION_NM / K)`. Three rules stop friction from faking touches
(the GUI said TOUCH 36–81 % of the time on the real recordings; now 3–28 %):
- while nothing touches, the baseline follows wherever friction parks each joint, so a
  touch is measured from just before it (an earlier poke's leftover offset on another joint
  no longer looks like part of this one);
- a touch is over once every joint has been dead still for 0.6 s within two rest bands
  (a finger is never that still) and the parked pose becomes the baseline;
- the spring-back when a finger lifts after such a re-zero (about minus the touch) is
  recognised and not reported as a new touch.
With `--dither` the last two are off: there is no parking to undo.

**Where.** `tactile/contact.py` scores ~4600 points covering the real surface of every
moving link (5 mm apart, from the meshes; points inside the arm or facing a neighbouring
part are dropped). For each point it fits the force a fingertip could apply there, which
can only **push into the surface** (a friction cone around the inward normal), and asks how
well `torque = J(point)^T force` explains all six joints, including the ones reading ~0.
Points that would need an implausible finger force (0.2 N or 40 N; log-normal around 3 N)
count less: the old "smallest force wins" rule put shoulder pokes on the gripper. A joint
resting on an end stop gives one-sided evidence: it can't feel a push into its stop, but a
push that would have lifted it off would have shown. That gives a probability per point:
the heat cloud. Evidence adds up while one touch lasts.

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

**Measured on the calibrated simulation** (`bench_touch.py`, 30 random 1–5 N pushes per
setting at random poses, friction 0.05 N·m, two earlier pokes leave the joints parked;
median error of the most likely point):

| | P=16 | P=16 + `--dither 4` | P=4 + `--dither 8` |
|---|---|---|---|
| pushes detected | 73 % | 83 % | 83 % |
| touches stuck after release | 1 / 30 | 0 / 30 | 0 / 30 |
| touch ends after release (median) | 0.70 s | 0.18 s | 0.22 s |
| right link | 73 % | 76 % | 76 % |
| median error / 80 % within | 3.9 / 6.6 cm | 3.5 / 7.3 cm | 3.3 / 6.9 cm |

Before the friction fixes (same simulation): 20/40 stuck touches, 55 % right link, 5.6 cm.
The real arm's accuracy needs a clean poke test (§2C).

---

## 5. All commands

| command | what |
|---|---|
| `touch_gui.py [--fake] [--teleop] [--model F] [--record]` | the 3D touch GUI (hold or teleop) |
| `touch_gui.py --dither A [--dither-hz F]` | shake goals ±A ticks so friction averages out (try 4) |
| `touch_gui.py --soft P` | softer servos (P gain, lerobot 16), restored on exit |
| `touch_gui.py --poke` | accuracy test: double-click a spot, poke it, cm error per poke |
| `touch_gui.py --replay CSV [--replay-speed S]` | a recorded run through the current code, in 3D |
| `touch_gui.py --torque-limit N --max-step N --d-on D --d-off D` | tuning: hold strength, teleop speed cap, detection thresholds |
| `replay_eval.py CSV [--variants]` | numbers on a recorded run: TOUCH time, touches, poke cm error |
| `bench_touch.py [--p ...] [--friction ...] [--dither ...] [--trials N]` | accuracy on the simulated arm, per link |
| `self_calibrate.py --fake` | measures each joint's jitter + friction band by tiny moves (sim; the real-arm run stays within ±1.5 % of each range, not run yet) |
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
| touch never ends after a push | friction bigger than `FRICTION_NM`: raise it in `tactile/config.py`, or use `--dither` |
| a long steady press ends by itself | without dither, a press held dead still looks like friction parking: use `--dither` |
| light pokes not felt | below the gear friction: poke firmer, or use `--dither` |
| touches while moving, nobody touching | no / old model. Record more varied motion and re-fit |
| arrow points backwards | that joint's `JOINT_SIGN` (mirror check A) |
| model moves differently from the arm | mirror check (A): `JOINT_SIGN` / `JOINT_OFFSET_TICKS` |
| `servo did not take P_Coefficient` | `--soft` could not set the gain: run without `--soft` |
| `--soft refused, nothing was changed` | a joint is on its end stop. Use `--teleop --soft 4` and pose it with the leader (softening on a stop is unsafe: it would drive off once lifted) |
| `SERVO STATUS ERROR` banner | overload / heat / voltage protection tripped; the servo goes soft. Lower `--torque-limit`, let it cool |
| window frozen, terminal still logging | the window isn't being drawn (hidden / other workspace / screen locked). Bring it to front; Ctrl+C works either way |
| Wayland `GLFWError: window position` warning | cosmetic, the window still opens |
| `Failed to sync read ... no status packet` / `Incorrect status packet` | another program is using the arm (`~/teleop.sh` / `lerobot-teleoperate`, another touch_gui): close it. Otherwise just run it again |
| low Hz in the banner | each register read is a bus round trip; touch_gui reads 3 per tick + Status every 10th (+1 goal write with `--dither`) |
