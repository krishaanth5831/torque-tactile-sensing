"""The hard 2 % motion limit used by self_calibrate.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile.envelope import SafetyEnvelope  # noqa: E402

RANGES = {"a": (1000, 3000), "b": (2030, 3481)}


def test_goals_never_leave_the_window():
    e = SafetyEnvelope({"a": 2000, "b": 2040}, RANGES)
    for g in (-1e9, 0, 1960, 2000, 2040, 1e9):
        assert abs(e.clamp_goal("a", g) - 2000) <= 0.015 * 2000 + 0.5
    # near the end of the range: the range wins
    assert e.clamp_goal("b", 0) == 2030
    assert e.clamp_goal("b", 1e9) <= 2040 + 0.015 * 1451 + 0.5


def test_abort_before_two_percent():
    e = SafetyEnvelope({"a": 2000, "b": 2040}, RANGES)
    assert e.violations({"a": 2035, "b": 2040}) == []            # 1.75 %
    assert e.violations({"a": 2037, "b": 2040})                  # 1.85 %: abort
    assert e.violations({"a": 2000, "b": 2040 - 27})             # gripper 1.86 %


def test_cannot_be_configured_past_the_hard_limit():
    with pytest.raises(ValueError):
        SafetyEnvelope({"a": 2000}, RANGES, cmd_frac=0.019, abort_frac=0.025)
    with pytest.raises(ValueError):
        SafetyEnvelope({"a": 2000}, RANGES, cmd_frac=0.019, abort_frac=0.018)
