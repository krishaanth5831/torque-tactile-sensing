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
SAFETY_REGISTERS = ["Torque_Limit", "Max_Torque_Limit", "Protection_Current", "Overload_Torque", "P_Coefficient"]


def load_ranges(path: Path = CALIBRATION_JSON) -> dict[str, tuple[int, int]]:
    """(range_min, range_max) per joint in raw ticks, from the lerobot calibration file."""
    cal = json.loads(Path(path).read_text())
    return {name: (cal[name]["range_min"], cal[name]["range_max"]) for name in JOINTS}
