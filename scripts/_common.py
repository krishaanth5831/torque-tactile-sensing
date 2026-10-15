"""Shared argparse flags + startup printout for every script."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.source import open_source  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="simulated arm, no hardware needed")
    p.add_argument("--port", default=config.FOLLOWER_PORT, help="follower serial port (by-id path)")
    p.add_argument("--no-window", action="store_true", help="no plot window (log only, Ctrl+C to stop)")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--release-on-exit", action="store_true",
                   help="torque OFF when done. The arm goes limp and drops, so SUPPORT IT")
    return p


def open_and_report(args, full: bool = True, pushes: bool = True, cartesian: bool = False):
    src = open_source(fake=args.fake, port=args.port, full=full, pushes=pushes, cartesian=cartesian)
    print(f"[tactile] source: {'FAKE' if args.fake else args.port}")
    try:
        rep = src.safety_report()
        for reg, vals in rep.items():
            print(f"  {reg:20s} {vals}")
    except Exception as e:  # never block a run on the report
        print(f"  (safety report failed: {e})")
    return src


class RateMeter:
    def __init__(self):
        self.n, self.t0, self.hz = 0, time.perf_counter(), 0.0

    def tick(self) -> float:
        self.n += 1
        dt = time.perf_counter() - self.t0
        if dt > 1.0:
            self.hz, self.n, self.t0 = self.n / dt, 0, time.perf_counter()
        return self.hz


def parse_joints(s: str, allow_gripper: bool = True) -> list[str]:
    if s in ("", "none"):
        return []
    if s == "all":
        js = list(config.JOINTS)
    elif s == "arm":
        js = [j for j in config.JOINTS if j != "gripper"]
    else:
        js = [x.strip() for x in s.split(",")]
    bad = [j for j in js if j not in config.JOINTS]
    if bad:
        raise SystemExit(f"unknown joint(s) {bad}; choose from {config.JOINTS}, or 'all' / 'arm'")
    if not allow_gripper and "gripper" in js:
        js.remove("gripper")
    return js
