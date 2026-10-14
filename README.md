# torque-tactile-sensing

Touch for the SO-101 **without adding sensors**: read what the STS3215 servos already
report (load, current, position error) and turn it into "someone is pushing me" and
"I am squeezing a tomato this hard".

Ideas, physics and the full plan are in **[IDEAS.md](IDEAS.md)**.

## Run it (watch first)

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
  `Overload_Torque` and `P_Coefficient`. A bare bus does not run lerobot's
  `configure()`, so lerobot's gripper caps (50 %) and P=16 are only there if the EEPROM
  has them. Check the printout.
- All goals are raw ticks clamped to `range_min/range_max` from
  `follower_white.json`. There is no lerobot normalisation, so the `use_degrees` trap
  doesn't apply.
- On exit torque stays **on** (the arm holds instead of dropping). `--release-on-exit`
  makes it go limp, so support the arm.
- A non-zero `Status` register (overload/heat/voltage trip) is shown in purple/red. A
  tripped servo goes soft, which looks exactly like a slip or a crushed fruit.

## Layout

```
tactile/
  config.py      ports, joint ids, calibration ranges, GRIPPER_CLOSED_END
  source.py      ArmSource (real bus, raw units) + FakeSource (P-servo sim), same interface
  detect.py      Baseline, PhaseBaseline, Hysteresis, analyse_squeeze (pure numpy, tested)
  liveplot.py    threaded matplotlib window (the bus loop never waits for drawing)
  logger.py      one new timestamped CSV per run
scripts/
  live_view.py   step 0, look at signals (optionally hold joints)
  push_detect.py step 1, hold / wiggle push detection, stop / comply reactions
  probe_gripper.py  which raw end of the gripper is closed
  squeeze.py     step 2, contact, stiffness, target grip, yield (crush) back-off
tests/           detector tests on synthetic signals
```

Tests: `env -u PYTHONPATH python -m pytest -q tests`. The `-u PYTHONPATH` keeps the
ROS install's pytest plugins out.

## Status

- Verified so far: detector unit tests, plus every script end to end on `--fake`, with a
  window and headless.
- Not run on the real arm yet. Check these first:
  1. Is `Present_Current` signed? Look at `cur_raw_*` in the CSV.
  2. Achieved loop Hz. It's shown in the banner.
  3. The gripper closed end, from `probe_gripper.py`.
