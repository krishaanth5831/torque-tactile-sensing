"""Hardware constants for the SO-101 follower (white arm).

Everything here is in RAW servo units on purpose. We never go through
lerobot's normalisation, so the use_degrees trap (no clamping in DEGREES mode)
cannot bite. Instead we clamp every goal to range_min/range_max ourselves.
"""
import json
from pathlib import Path

# The 12 V arm is the follower. Only the follower has motors strong enough to push
# back, so it is the only arm we sense with. Use the by-id path, never /dev/ttyACM*:
# those swap with plug order.
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B014343-if00"
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B014877-if00"

CALIBRATION_JSON = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower_white.json"
)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(JOINTS)}

# Which end of the gripper's raw range is CLOSED. Do not guess this. Run
# scripts/probe_gripper.py once, close the jaws by hand, and it tells you.
# Then write the answer here ("min" or "max").
GRIPPER_CLOSED_END = None

# STS3215 safety registers. Torque_Limit (addr 48) is RAM. It resets on power-up
# and is the live knob we turn. The rest are EEPROM and we only READ them.
#   Torque_Limit / Max_Torque_Limit : 0..1000 = 0..100 % of stall torque
#   Protection_Current              : current cut-off, raw units
#   Overload_Torque                 : % torque the servo drops to after an overload trip
#   P_Coefficient                   : position-loop P gain. lerobot sets 16 (firmware default 32).
#                                     Lower P = softer servo = more position error per newton.
SAFETY_REGISTERS = ["Torque_Limit", "Max_Torque_Limit", "Protection_Current", "Overload_Torque", "P_Coefficient",
                    "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone"]


LEADER_CALIBRATION_JSON = (
    Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader_black.json"
)

# ------------------------------------------------------------------ raw ticks <-> 3D model
# The MuJoCo model (assets/so101, Menagerie, derived from so101_new_calib) has its zero
# pose at lerobot's calibration MIDDLE pose, which lerobot's homing puts at raw ~2048.
#   model angle [rad] = JOINT_SIGN * (raw - 2048 - JOINT_OFFSET_TICKS) * 2*pi/4096
# Check it with the MIRROR CHECK in USAGE.md (torque off, move each joint by hand, the
# model must follow). If a joint moves the wrong way, flip its sign. If it is off by a
# constant angle, set its offset in ticks.
JOINT_SIGN = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1}
# wrist_roll +1024 (90 deg): mirror check 2026-10-15, its EEPROM zero was moved on 2026-09-20.
# +offset turns the model wrist clockwise seen from the arm looking out (axis is -x).
JOINT_OFFSET_TICKS = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 1024}
# The gripper's model range is not symmetric (-0.17..1.75 rad), so it is mapped linearly
# from the calibrated raw range. Flip if the model jaw opens when the real one closes.
GRIPPER_MODEL_FLIP = False

# ------------------------------------------------------------------ position error -> torque
# STS3215 stall torque ~2.94 N.m at 12 V (also the Menagerie model's forcerange).
# Present_Load is drive duty, so at standstill torque ~= load/1000 * stall.
NM_PER_LOAD = 2.94 / 1000
# The touch signal is the POSITION ERROR err = goal - pos, not Present_Load. On the real
# arm (2026-10-15 CSV) load is exactly ~-9 x err with a dead band (0, then jumps to +-20),
# so it carries less information than err, and err's +-0.5 tick jitter averages below
# one tick while load's dead band does not. The servo is a P controller, so at rest
#     external torque (N.m, +tick direction) = -K * (err - err_at_rest),  K = K16 * P / 16
# K16 = 9 load units per tick (measured slope) x NM_PER_LOAD. The sign is physics, not a
# setting: a push toward +ticks moves pos up, so err goes down. The overall scale only
# changes the N shown, never WHERE the touch is (that depends on torque ratios).
NM_PER_TICK_P16 = 9 * NM_PER_LOAD
P_DEFAULT = 16                 # lerobot's position_p_coefficient for the follower
# Torque the joint cannot report, whatever the averaging (quantisation). Added to the
# learned noise.
TAU_FLOOR_NM = 0.004
# THE REST BAND. After a push ends, a servo does not return exactly: inside its dead zone
# (CW/CCW_Dead_Zone, 1 tick) it does not drive at all, and gear friction holds it until
# the net torque exceeds FRICTION_NM. So it can settle anywhere within
#     band = DEAD_ZONE_TICKS + FRICTION_NM / K   ticks   of where it started.
# Offsets inside the band are "no touch"; only what exceeds it counts. In N.m the dead-zone
# part is K * 1 tick: 4x smaller at --soft 4 than at P=16, the main gain of soft servos.
DEAD_ZONE_TICKS = 1.0
FRICTION_NM = 0.01             # ESTIMATE until measured on the arm (bench_touch.py sweeps it)

# ------------------------------------------------------------------ contact localisation
# Bodies whose surface can be touched (the fixed base never moves a joint). camera_mount
# is the wrist camera mount; drop it here if your gripper has none.
TOUCH_BODIES = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "camera_mount",
                "moving_jaw_so101_v1"]
SURFACE_SPACING_M = 0.005      # one candidate point per 5 x 5 mm of surface
FINGER_MU = 0.6                # friction cone: how far a push may lean from the surface normal
FORCE_PRIOR_N = 10.0           # finger pushes are a few N: mild preference for small forces


def load_ranges(path: Path = CALIBRATION_JSON) -> dict[str, tuple[int, int]]:
    """(range_min, range_max) per joint in raw ticks, from the lerobot calibration file."""
    cal = json.loads(Path(path).read_text())
    return {name: (cal[name]["range_min"], cal[name]["range_max"]) for name in JOINTS}
