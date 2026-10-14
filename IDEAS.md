# Ideas: torque-based tactile sensing on the SO-101

Status per idea: **built** = a script in this repo does it (tested on `--fake` only so
far), **next** = worth doing soon, **later** = once the basics work.

---

## 1. What the motors can actually tell you

Each STS3215 is a position servo: you send `Goal_Position`, an internal loop (mostly P)
drives the motor through a big gearbox. Readable registers that carry force information:

| signal | register | what it really is | why it is useful |
|---|---|---|---|
| `load` | Present_Load (60) | drive duty (PWM %) the controller is applying, ±1000 | what the servo is *trying* to push |
| `cur` | Present_Current (69) | measured motor current | motor torque ≈ Kt × current. Closest to real torque |
| `err` | Goal − Present_Position | position error | on a P loop, drive ≈ P × err, so err is a force sensor too, and often the cleanest one |
| `vel` | Present_Velocity (58) | joint speed | needed to separate friction from touch |
| `status` | Status (65) | error bits | an overload trip makes the servo go soft and **looks like a slip**. Always watch it |

Three physical facts decide what is possible:

1. **Gear friction is the enemy.** About 1:345 reduction means lots of static friction. A
   small push can be absorbed entirely by friction and never show up in current. Expect a
   dead band and measure it (experiment E1). Pushes above the dead band show up clearly.
2. **P gain is a sensitivity knob.** lerobot sets `P_Coefficient` 16 (firmware default
   32). Lower P gives more position error per newton, so it's more sensitive and softer,
   but positions less precisely. A "soft mode" for gripping and a "stiff mode" for moving
   is a legit trick.
3. **Units are raw.** Load units are not newtons. Calibrate against a kitchen scale (§3A)
   before trusting any force number.

Loop rate: every `sync_read` is one bus round trip, and we read 3–6 registers per tick.
The Hz shows in the window banner. **next**: one block read of registers 56–70 (they are
contiguous) gets everything in a single transaction. lerobot's `getData` only unpacks
1/2/4-byte values, so this needs a small custom reader.

---

## 2. "Someone is pushing me": the arm level

The trick in every version: **touch = what you measure − what you expected**. The versions
differ only in how they predict "expected".

| level | expected comes from | works when | status |
|---|---|---|---|
| L0 hold | mean + noise learned for 3 s while still | arm holds a pose | **built**: `push_detect.py` |
| L1 phase | expected signal at each point of a repeated motion | arm repeats a motion (wiggle, a fixed pick path) | **built**: `push_detect.py --mode wiggle` |
| L2 model | gravity(q) + friction(q̇) + inertia(q̈) | any motion | **next** |
| L3 observer | momentum observer (De Luca), no q̈ needed | any motion, the "proper" cobot way | later |

**L2, two ways:**
- *Physics*: compute gravity torque per joint from the SO-101 URDF masses with MuJoCo
  `mj_inverse` (the Vinea sim already loads the arm), then fit one scale and offset per
  joint, because load units are not N·m.
- *Data*: record ~10 min of slow random motion with nobody touching it. Fit, per joint,
  `load ≈ a·sin q + b·cos q + c·q̇ + d·sign(q̇) + e·q̈` by least squares. That is classic
  robot identification. A small MLP on (q, q̇, err) is the lazy upgrade. Then residual =
  measured − predicted, thresholded with the same `Hysteresis`.

**Reactions** (what to do once touch is detected):
- **built** `--react stop`: pause the motion. This is the collision stop, and it matters
  for a robot working next to greenhouse staff.
- **built** `--react comply`: the goal follows your hand, which gives hand-guiding /
  lead-through.
- **next** admittance: `goal += K × residual` each tick, a smooth version of comply.
- **later** kinesthetic teaching: record demonstrations by pushing the follower directly,
  with no leader arm. That's a cheaper way to collect LeRobot data.
- **later** where is the force? The pattern of residuals across joints is τ = Jᵀ F.
  Assume the contact is at the gripper and solve F = (Jᵀ)⁺ τ for the force direction
  ("the truss is pushing me left").

---

## 3. The tomato: grip without crushing (the gripper level)

The window you want: **enough force that it doesn't slip out, less than what bruises it.**
For a light fruit the "don't drop it" force is small, so the window is probably wide.
Measure both ends instead of trusting textbook values.

**A. Calibrate units to grams (next, do this early).** Put one jaw against a kitchen
scale and squeeze at several `--torque-limit` values. Log load/current against grams and
fit a line. Hanging known weights does the same for the arm joints.

**B. Hardware force cap (built).** `Torque_Limit` is enforced by the servo itself. Pick it
so that the maximum squeeze is below the bruise force, then just close "past" the fruit.
The servo stalls at the limit, which gives a **constant-force grip by design**.
`squeeze.py` reports this as `STALL`. It is the simplest robust strategy and works even
if all the clever detection fails. Note that `--max-lead` caps force too, because
drive ≈ P × lead.

**C. Contact + stiffness (built).** `squeeze.py` finds the contact point and the slope of
force vs further closing. **Firmness is a ripeness cue**: growers and QC labs use fruit
firmness testers. Combine it with the B.L.U.S.H vision gate. Vision says "red", touch says
"firm, not over-ripe", and the claw gate gets a second, independent veto.

**D. Crush / yield detection (built).** When the skin gives way, the force stops rising,
or drops, while the jaw keeps closing (or surges forward). `analyse_squeeze` flags both
the sudden and the gradual case, and the script backs off. Goal: never see this on a good
fruit. Use it to *find* the bruise limit on sacrificial supermarket tomatoes, then set the
target well below it.

**E. Slip during lift (next).** Three free signals:
- the gripper position creeps closed at a constant goal, so the fruit is sliding out
- the grip load drops
- **weight check**: after lifting, `shoulder_lift`/`wrist_flex` load rises by about the
  fruit's weight. "Did I actually pick it?" comes for free, so missed picks get detected
  without a camera.

**F. Pluck detection (next, very Vinea).** While pulling or twisting the fruit off the
truss, the load builds and then **drops suddenly when the pedicel snaps**. So "it's off
the plant" means stop pulling. If the load keeps rising past a limit, abort: you are
pulling the vine.

**G. Fin-ray caveat.** The TPU fingers are a spring in series with the fruit:
`1/k_measured = 1/k_finger + 1/k_fruit`. Squeeze a rigid wooden block first to get
`k_finger`, then `k_fruit = 1 / (1/k_measured − 1/k_finger)`. Calibrate on the stock jaws
first, then redo it with the fin-ray. Upside: the soft finger turns gripper position after
contact into a force reading, so it acts as a spring scale.

---

## 4. Learning-based (later)

- **Effort as policy input.** Append load/current to `observation.state` when recording
  LeRobot datasets. The stock `so_follower` only exposes position, so this needs a small
  robot subclass. Force-aware inputs help contact-rich tasks, and π0.5 / ACT can take
  them as extra state dims.
- **Squeeze-curve classifier.** Record 20–50 squeezes with ripeness labels from the vision
  gate, and classify ripe / unripe / not-tomato / crushed from the curve alone.
- **Residual model** for L2 above (MLP), trained on contact-free motion logs.

---

## 5. Experiments, in order

| # | what | command | you learn |
|---|---|---|---|
| E0 | signals sanity | `live_view.py`, then `--hold arm` and push | noise per joint/signal, loop Hz, is current signed (`cur_raw_*` in CSV) |
| E1 | dead band | `push_detect.py`, push with a luggage/kitchen scale, increasing slowly | smallest detectable force per joint; load vs cur vs err, which is cleanest |
| E2 | drift | hold 10 min, `--no-window --duration 600` | does the baseline drift with motor temperature |
| E3 | moving | `push_detect.py --mode wiggle`, vary amp/period | false alarms vs speed |
| E4 | units → grams | gripper on a kitchen scale at 3–4 torque limits | the calibration line |
| E5 | squeeze ladder | sponge → foam ball → wooden block (k_finger) → cherry tomatoes green vs red | stiffness separates ripeness? where do they split? |
| E6 | lift + shake | grip, lift, shake | weight check and slip signals |
| E7 | pluck | in the greenhouse rig, on the vine | the snap signature |

---

## 6. Known limits (be honest about them)

- Not a torque sensor: gear friction, a hysteresis dead band, quantisation.
- Drifts with motor temperature and supply voltage (the 12 V PSU sags under load).
- The register semantics are Feetech-firmware specific, and the current unit (≈6.5 mA) is
  from their docs, not verified here.
- Overload protection (`Protection_Current`, `Overload_Torque`) can trip during a long
  stall. Watch `status`.
- **If it isn't enough**, cheap add-ons that still fit the fin-ray: an FSR or barometric
  sensor inside the finger, a small load cell under one jaw, a piezo disc for slip
  vibration. Try sensorless first, because it's free.

## 7. Code TODO

- [ ] block read of 56–70 (speed)
- [ ] `fit_model.py`: L2 gravity + friction regression from free-motion CSVs
- [ ] `calibrate_grams.py` + a `calibration.json` mapping units to grams
- [ ] lift/slip/weight-check script (§3E)
- [ ] lerobot robot subclass exposing load/current as observations (§4)
