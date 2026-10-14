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

Start here. These all run on the simulated arm, which gets random 3D "hand pushes".

```bash
# The touch GUI: SO-101 model, joint-torque arrows, touched link in red, estimated
# push arrow (red, or orange if ambiguous), and the TRUE simulated push in cyan
python scripts/touch_gui.py --fake

# Same, with a simulated leader moving the arm (moves ~4 s, pauses ~3 s)
python scripts/touch_gui.py --fake --teleop

# Plain signal plots and the simpler detectors from step 0/1
python scripts/live_view.py --fake
python scripts/push_detect.py --fake
python scripts/push_detect.py --fake --mode wiggle --joint elbow_flex --react stop

# Unit tests (the -u keeps the ROS install's pytest plugins out)
env -u PYTHONPATH python -m pytest -q tests
```

In the fake GUI the red/orange arrow should land on or next to the cyan one. On the
wrist and gripper the direction matches closely. On the shoulder / upper arm / lower arm
only the link is really known (see "What the GUI can and can't know" below).

---

## 2. The real arm on its own (follower holds still, you push it)

Plug in the follower (white arm, 12 V PSU on). The leader is not needed. Close
`lerobot-teleoperate` first: only one program can open the follower's port.

```bash
python scripts/touch_gui.py                    # holds its current pose at Torque_Limit 500
python scripts/touch_gui.py --torque-limit 300 # softer: more deflection per push, weaker hold
```

1. It starts in **LEARNING** (orange, about 2 s). **Don't touch it** while it learns
   what "nothing touching me" looks like.
2. **WATCHING**: push any link with a finger. The joint arrows light up and the link
   turns red. The arrow shows where and which way.
3. Press **R** in the window to re-learn, e.g. after moving it by hand.

At startup the terminal prints the servo protection registers (`Torque_Limit`,
`Max_Torque_Limit`, `Protection_Current`, `Overload_Torque`, `P_Coefficient`). If
`P_Coefficient` shows 32 instead of lerobot's 16, the servos are stiffer, so a push moves
them less but load still rises. That's fine, just note it.

### First-time checks (do these once, in this order)

**A. Mirror check: does the model move like the real arm?**

```bash
python scripts/touch_gui.py --limp     # never enables torque: move it by hand. SUPPORT the arm,
                                       # it doesn't hold itself up (start with it resting)
```

Move one joint at a time by hand and watch the model.
- Moves the **wrong way**: flip that joint in `tactile/config.py` → `JOINT_SIGN`.
- Moves the right way but sits at a **constant angle off**: set `JOINT_OFFSET_TICKS`
  (4096 ticks = 360°, so 11 ticks ≈ 1°).
- Gripper jaw opens when the real one closes: `GRIPPER_MODEL_FLIP = True`.

The model's zero pose is lerobot's calibration middle pose, so these should already be
close. wrist_roll is the one to watch: its zero was moved in EEPROM on 2026-09-20.

**B. Push-direction check: does the arrow point the way you push?**

Hold mode (`python scripts/touch_gui.py`), wait for WATCHING. Rotate the 3D view
(drag) until it looks like the real arm from where you stand. Push the gripper firmly
**sideways**, horizontally. The red arrow must point **the same way your finger
pushes**. If it points the opposite way, set `FORCE_SIGN = +1` in `tactile/config.py`.
Then push the gripper **down**: the arrow should point down. Re-check on the forearm
(orange arrow there: direction is only partly known, but it shouldn't point backwards).

**C. Is Present_Current signed?** Look at `cur_raw_*` columns in the CSV from any run. If
values ever exceed 32768 the servo reports a sign bit; `tactile/source.py` already
decodes it, just confirm.

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

# 2. Fit expected-load-while-moving (prints the next command at the end of step 1).
#    The window shows measured vs predicted load per joint.
python scripts/fit_model.py data/free_motion_<stamp>.csv

# 3. Teleop with touch detection during motion
python scripts/touch_gui.py --teleop --model models/load_model.json
```

The model uses only the **commanded** motion (leader positions and their speed and
acceleration), never the follower's own position error. A push shows up in that error,
so a model that used it would learn to "predict" your hand. Re-fit after you change
`--torque-limit`, P gain or hardware (e.g. the fin-ray gripper). An old fit is kept as
`load_model_prev.json`.

Moving detection is less sensitive than still detection, because the model is never
perfect. The `sigma` that `fit_model.py` prints is the noise floor per joint. Still
periods always switch back to the sharper static baseline automatically.

---

## 4. What the GUI can and can't know

- **Joint torque arrows are always valid.** They show what each joint feels, in
  ≈N·m (load/1000 × 2.94 N·m stall torque; uncalibrated, hence the `~`).
- **The red link** comes from the most distal joint that felt the push: a push on a link
  can only load the joints between it and the base. Usually right; when wrong, it's one
  link too close to the base. A push whose line runs straight through a joint axis gives
  that joint nothing.
- **Point and direction:** from the wrist out (4–5 joints respond), both are fitted and
  usually good. On the shoulder / upper arm / lower arm only 1–3 joints respond, which
  isn't enough to pin down a 3D force. The GUI then shows an **orange** arrow with a halo
  at the link middle, and the force shown is the *smallest* force that explains the
  torques (a lower bound).
- **On the fake arm** this was measured over 87 random pushes: 82 detected, link exactly
  right 63×, one link toward the base 17×, direction cos ≈ 0.99 where the point is
  constrained. The fake uses the same model, so this proves the pipeline is consistent,
  not that the real arm's signs are right. That's what checks A and B are for.

---

## 5. All commands

| command | what |
|---|---|
| `touch_gui.py [--fake] [--teleop] [--model F] [--record]` | the 3D touch GUI (hold or teleop) |
| `touch_gui.py --torque-limit N --max-step N --z-on Z --z-off Z --z-loc Z` | tuning: hold strength, teleop speed cap, detection thresholds |
| `fit_model.py CSV... [--out F]` | fit expected-load-while-moving from no-touch recordings |
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
| constant TOUCH right after LEARNING | you touched it while learning, or it was still settling: press **R** |
| touches while moving, nobody touching | no / old model. Record more varied motion and re-fit |
| arrow points backwards | push-direction check (B): `FORCE_SIGN` |
| model moves differently from the arm | mirror check (A): `JOINT_SIGN` / `JOINT_OFFSET_TICKS` |
| `SERVO STATUS ERROR` banner | overload / heat / voltage protection tripped; the servo goes soft. Lower `--torque-limit`, let it cool |
| Wayland `GLFWError: window position` warning | cosmetic, the window still opens |
| low Hz in the banner | each register read is a bus round trip; touch_gui reads 3 per tick + Status every 10th |
