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

**Measured on the real follower (2026-10-15):** `err` wins. Present_Load is ≈ −9 × err
(P=16) with a dead band: it reads 0, then jumps straight to ±20, so it carries strictly
less than err. Present_Current never exceeded 1–7 raw units at rest and under light
pushes: useless for touch. err jitters ±0.5 tick at rest, so averaging it over 0.1 s
resolves fractions of a tick. The touch GUI now runs on err alone
(`tactile/config.py`: `NM_PER_TICK_P16`).

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
- *Data*: record no-touch motion and fit, per joint,
  `load ≈ a·gravity(q) + b·q̇ + c·sign(q̇) + d·q̈ + e` by least squares. **built**:
  `touch_gui.py --record` then `fit_model.py`, with gravity from MuJoCo. Inputs must
  come from the **commanded** motion only. On a P servo, load ≈ P × err whatever caused
  the error, including a hand, so a model fed with err (or measured velocity) would
  predict the push and the residual would stay flat. A small MLP on the same commanded
  inputs is the possible upgrade.

**Reactions** (what to do once touch is detected):
- **built** `--react stop`: pause the motion. This is the collision stop, and it matters
  for a robot working next to greenhouse staff.
- **built** `--react comply`: the goal follows your hand, which gives hand-guiding /
  lead-through.
- **next** admittance: `goal += K × residual` each tick, a smooth version of comply.
- **later** kinesthetic teaching: record demonstrations by pushing the follower directly,
  with no leader arm. That's a cheaper way to collect LeRobot data.
- **built** where is the force? `tactile/contact.py`, shown by `touch_gui.py`. ~4600
  points on the real link surfaces (from the meshes, unreachable faces dropped); at each,
  the best **push-only** fingertip force (friction cone around the inward normal, FISTA,
  all points at once) and how well τ = Jᵀ F explains all six joints, including the
  zero-reading ones. Probability per point = heat cloud; evidence adds up while a touch
  lasts. Same idea as the Contact Particle Filter (Manuelli & Tedrake, 2016). On the
  realistic sim (`bench_touch.py`): gripper ~1 cm, wrist 2–3 cm, forearm / upper arm
  ~4 cm. The 1–3-joint links stay ambiguous by physics, and the cloud shows it.
- **built** `--poke`: double-click a spot on the model, poke it on the arm, cm error per
  poke. The real accuracy number.
- **built (sim only)** beat friction with **dither**: `touch_gui.py --dither 4` shakes every
  goal ±4 ticks at 8 Hz (triangle, joints out of step) and senses against the unshaken goal
  over two periods. On the simulation calibrated to the real arm it removes every stuck
  touch, ends touches 0.18 s after release (0.7 s without) and detects 83 % instead of
  73 % of pushes at P=16. Costs a faint hum. **Next: try
  it on the real arm** — the unknown is whether the STS3215 follows an 8 Hz ±4-tick goal
  (its own acceleration limits); try 4–6 Hz if not.
- **built** friction-aware estimator, from the real 2026-10-15 recordings: the readings sit
  on one integer for seconds, gear friction is ~0.05 N·m per joint and parks joints up to
  ~2 ticks off after every poke. The baseline now follows parked offsets while nothing
  touches, a touch ends once every joint is dead still within two rest bands, and the
  spring-back after such a re-zero is not a new touch. Real recordings: time in TOUCH
  81 → 12 %, 36 → 3 %, no more 15–40 s stuck touches. (Simulation: stuck touches 20/40 →
  1/30, right link 55 → 73 %.)
- **built** a finger-sized force prior (log-normal around 3 N; "smallest force wins" put
  shoulder pokes on the gripper) and one-sided evidence from joints resting on end stops.
- **built** working without the arm: `replay_eval.py` scores the current code on any
  recorded run, `touch_gui.py --replay` shows it in 3D, poke runs log their targets in
  every row, and `sim.py` is calibrated to the real arm (encoder flicker ~5 %, 0.05 N·m).
- **next** fit per-joint torque gains (and `FRICTION_NM`) from a clean poke-test run (targets
  now logged per row): the points are known, so the gains that minimise the poke error
  can be solved for. `self_calibrate.py` (sim-tested, ±1.5 % moves) measures each joint's
  friction band directly once the arm may move.
- **next** active touch for the proximal links: when a touch is ambiguous, wiggle the
  wrist/elbow a few ticks while the finger stays. New poses give new Jacobians, which
  pin the point down on the upper arm too.
- **next** a calibrated force scale (kitchen scale), so the ~N become real N.
- **later** a tactile "map" over a whole session: accumulate where it gets touched.

---

## 3. The tomato: grip without crushing (the gripper level). PARKED for now

The focus is the arm's sense of touch first (§2). Everything below still stands for later.

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

- Not a torque sensor: gear friction, a hysteresis dead band, quantisation. After a push
  a joint can settle up to (dead zone + friction / K) ticks off; the estimator ignores
  offsets inside that rest band, which also means pushes smaller than it are not felt.
- Joint torques alone can't locate a touch on links that move only 1–2 joints (shoulder,
  upper arm), nor any push parallel to the pitch axes. See USAGE.md §4.
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
- [x] `fit_model.py`: L2 gravity + friction regression from free-motion CSVs (now on err)
- [x] surface contact localisation + poke test + realistic sim benchmark
- [x] dither mode (§2, sim-validated), replay tools, friction-aware estimator
- [ ] dither on the real arm, per-joint gains from clean poke data, active touch
- [ ] `calibrate_grams.py` + a `calibration.json` mapping units to grams
- [ ] lift/slip/weight-check script (§3E)
- [ ] lerobot robot subclass exposing load/current as observations (§4)
