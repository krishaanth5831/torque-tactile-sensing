"""A recorded CSV plays back through ReplaySource exactly as it was logged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.logger import CsvLogger  # noqa: E402
from tactile.replay import ReplayEnded, ReplaySource  # noqa: E402
from tactile.source import FakeSource  # noqa: E402

J = config.JOINTS


def test_replay_reproduces_the_log(tmp_path):
    src = FakeSource(pushes=False, rate_hz=1000)
    src.hold_here(J, 500)
    log = CsvLogger("t", J, extra=[f"nom_{j}" for j in J], out_dir=tmp_path)
    rec = []
    for _ in range(40):
        s = src.read()
        log.write(s, {f"nom_{j}": src.goals[j] for j in J})
        rec.append(s)
    log.close()
    rp = ReplaySource(str(log.path), speed=1000.0)
    for s in rec:
        r = rp.read()
        assert r.pos == s.pos and r.err == s.err
    try:
        rp.read()
        raise AssertionError("should have ended")
    except ReplayEnded:
        pass
