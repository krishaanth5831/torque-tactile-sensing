"""Detector tests on synthetic signals (no hardware)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile.detect import Baseline, Hysteresis, PhaseBaseline, analyse_squeeze  # noqa: E402
from tactile.source import decode_current  # noqa: E402

rng = np.random.default_rng(0)


def test_baseline_flags_push_not_noise():
    b = Baseline()
    for _ in range(200):
        b.add([100 + rng.normal(0, 5), -40 + rng.normal(0, 5)])
    b.freeze()
    h = Hysteresis(2, on=6, off=3)
    quiet = [h.update(b.z([100 + rng.normal(0, 5), -40 + rng.normal(0, 5)])) for _ in range(500)]
    assert sum(q.any() for q in quiet) == 0
    assert h.update(b.z([100, -40 + 80])).tolist() == [False, True]


def test_hysteresis_holds_between_thresholds():
    h = Hysteresis(1, on=5, off=2)
    assert not h.update([4])[0]
    assert h.update([6])[0]
    assert h.update([3])[0]        # still on: above off
    assert not h.update([1])[0]


def test_phase_baseline_cancels_motion_profile():
    pb = PhaseBaseline(1, bins=40)
    prof = lambda ph: 150 * np.sin(2 * np.pi * ph) + 60 * np.sign(np.cos(2 * np.pi * ph))  # gravity + friction
    for i in range(3 * 400):
        ph = (i / 400) % 1
        pb.add(ph, [prof(ph) + rng.normal(0, 5)])
    pb.freeze()
    zs = [abs(pb.z(ph, [prof(ph) + rng.normal(0, 5)])[0]) for ph in np.linspace(0, 0.999, 300)]
    assert np.median(zs) < 3
    assert abs(pb.z(0.3, [prof(0.3) + 120])[0]) > 6


def _squeeze(contact=200, k=1.5, crush_at=None, n=500):
    travel = np.linspace(0, 400, n)
    f = np.where(travel > contact, k * (travel - contact), 0.0)
    if crush_at is not None:
        i = np.argmax(f > crush_at)
        f[i:] = f[i] + 0.1 * k * (travel[i:] - travel[i])
    return travel, f + rng.normal(0, 3, n)


def test_squeeze_contact_and_stiffness():
    travel, f = _squeeze(contact=200, k=1.5)
    r = analyse_squeeze(travel, f, free_load_std=3)
    assert abs(r.contact_pos - 200) < 15
    assert 1.2 < r.stiffness < 1.8
    assert r.yield_idx is None


def test_squeeze_softer_object_reads_softer():
    a = analyse_squeeze(*_squeeze(k=2.5), free_load_std=3)
    b = analyse_squeeze(*_squeeze(k=0.8), free_load_std=3)
    assert a.stiffness > b.stiffness


def test_squeeze_detects_yield():
    travel, f = _squeeze(contact=150, k=1.5, crush_at=200)
    r = analyse_squeeze(travel, f, free_load_std=3)
    assert r.yield_idx is not None


def test_squeeze_detects_sudden_collapse_with_jaw_surge():
    # what the fake crush looks like: load drops while travel jumps ~30 ticks/sample
    travel = np.concatenate([np.linspace(0, 400, 300), [430, 480, 530, 560, 580, 590, 600, 610]])
    f = np.where(travel > 200, 1.9 * (travel - 200), 0.0)
    f[300:] = [380, 300, 200, 140, 125, 122, 121, 122]
    r = analyse_squeeze(travel, f + rng.normal(0, 3, len(f)), free_load_std=3)
    assert r.yield_idx is not None and r.yield_idx >= 300


def test_squeeze_stall_is_not_a_yield():
    # torque limit reached: travel stops, load flat with noise -> no yield
    travel = np.concatenate([np.linspace(0, 300, 250), np.full(150, 300.0)])
    f = np.where(travel > 150, 1.6 * (travel - 150), 0.0)
    r = analyse_squeeze(travel + rng.normal(0, 0.5, 400), f + rng.normal(0, 3, 400), free_load_std=3)
    assert r.yield_idx is None


def test_squeeze_empty_gripper():
    travel = np.linspace(0, 400, 300)
    r = analyse_squeeze(travel, rng.normal(0, 3, 300), free_load_std=3)
    assert r.contact_idx is None


def test_decode_current():
    assert decode_current(120) == 120
    assert decode_current(0x8000 | 120) == -120
