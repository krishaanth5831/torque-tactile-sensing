"""A hard motion limit around the pose a run starts in.

Every goal a self-calibration run sends goes through `clamp_goal`, and every reading
through `violations`. No joint is commanded more than `cmd_frac` of its calibrated range
away from where it started, and if one is MEASURED more than `abort_frac` away (a push, a
slip, a sag) the run aborts and sends every goal back to the start.
"""
from __future__ import annotations


class SafetyEnvelope:
    HARD_MAX = 0.02            # the user's limit: never more than 2 % of a joint's range

    def __init__(self, start: dict[str, float], ranges: dict[str, tuple[int, int]],
                 cmd_frac: float = 0.015, abort_frac: float = 0.018):
        if not 0 < cmd_frac < abort_frac < self.HARD_MAX:
            raise ValueError(f"need 0 < cmd_frac < abort_frac < {self.HARD_MAX}")
        self.start = {j: float(v) for j, v in start.items()}
        self.ranges = ranges
        self.span = {j: ranges[j][1] - ranges[j][0] for j in self.start}
        self.cmd = {j: cmd_frac * self.span[j] for j in self.start}
        self.abort = {j: abort_frac * self.span[j] for j in self.start}

    def clamp_goal(self, j: str, goal: float) -> int:
        lo, hi = self.ranges[j]
        a = max(self.start[j] - self.cmd[j], lo)
        b = min(self.start[j] + self.cmd[j], hi)
        return int(round(min(max(goal, a), b)))

    def frac(self, j: str, pos: float) -> float:
        """Signed distance from the start, as a fraction of the joint's range."""
        return (pos - self.start[j]) / self.span[j]

    def violations(self, pos: dict[str, float]) -> list[str]:
        return [f"{j} moved {abs(pos[j] - self.start[j]):.0f} ticks ({abs(self.frac(j, pos[j])):.1%} of range)"
                for j in self.start if abs(pos[j] - self.start[j]) > self.abort[j]]
