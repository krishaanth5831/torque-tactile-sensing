"""Surface contact localisation with exact synthetic pushes: tau = J(p)^T F.

The truth points come from a FINER, differently seeded surface map than the localiser's,
so a hit is not just "found its own grid point"."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tactile import config  # noqa: E402
from tactile.contact import ContactLocalizer  # noqa: E402
from tactile.model import SO101Model  # noqa: E402
from tactile.surface import build_surface  # noqa: E402

RANGES = {j: (900, 3100) for j in config.JOINTS}
POSE = {"shoulder_pan": 2300, "shoulder_lift": 1700, "elbow_flex": 2500, "wrist_flex": 2200,
        "wrist_roll": 2048, "gripper": 2300}
SIGMA = np.full(len(config.JOINTS), 0.005)       # N.m


@pytest.fixture(scope="module")
def setup():
    model = SO101Model(ranges=RANGES)
    loc = ContactLocalizer(model)
    truth = build_surface(model.m, spacing_m=0.003, seed=777)
    return model, loc, truth


def _push(model, truth, i, rng, mu=0.4, f=3.0):
    model.set_pose(POSE)
    b = truth.body[i]
    R = model.d.xmat[b].reshape(3, 3)
    p = model.d.xpos[b] + R @ truth.local[i]
    n = R @ truth.normal[i]
    t = rng.normal(size=3)
    t -= (t @ n) * n
    t *= mu * rng.uniform() / (np.linalg.norm(t) + 1e-12)
    F = (-n + t) / np.linalg.norm(-n + t) * f
    return p, F, model.model_to_tick(model.torque_from_force(int(b), p, F))


def _run(loc, tau):
    loc.reset()
    return loc.update(POSE, tau, SIGMA)


def test_wrist_and_gripper_localised(setup):
    """5 joints respond: the line of action is fixed, the push-only cone picks its entry point."""
    model, loc, truth = setup
    rng = np.random.default_rng(0)
    errs, links_ok = [], []
    want = {"wrist", "gripper", "moving_jaw_so101_v1"}
    idx = [i for i in range(truth.n) if truth.link_of_body[int(truth.body[i])] in want]
    for i in rng.choice(idx, 40, replace=False):
        p, F, tau = _push(model, truth, i, rng)
        c = _run(loc, tau)
        errs.append(np.linalg.norm(c.point - p) * 100)
        links_ok.append(c.link == truth.link_of_body[int(truth.body[i])])
    assert np.median(errs) < 4.0, np.median(errs)     # cm, single noise-free look
    # misses: wrist vs gripper when the push line runs through the wrist_roll axis
    assert np.mean(links_ok) >= 0.75


def test_uncertainty_is_honest(setup):
    """Where physics can't pin the point (1-3 joints respond), the estimate must SAY so:
    the error should be within ~2x the claimed spread."""
    model, loc, truth = setup
    rng = np.random.default_rng(5)
    ok = []
    for i in rng.choice(truth.n, 60, replace=False):
        p, _, tau = _push(model, truth, i, rng)
        c = _run(loc, tau)
        ok.append(np.linalg.norm(c.point - p) * 100 <= 2 * c.spread_cm + 2.0)
    assert np.mean(ok) > 0.8, np.mean(ok)


def test_force_direction_recovered(setup):
    model, loc, truth = setup
    rng = np.random.default_rng(1)
    idx = [i for i in range(truth.n) if truth.link_of_body[int(truth.body[i])] in ("wrist", "gripper")]
    cos = []
    for i in rng.choice(idx, 20, replace=False):
        _, F, tau = _push(model, truth, i, rng)
        c = _run(loc, tau)
        cos.append(F @ c.force / (np.linalg.norm(F) * np.linalg.norm(c.force) + 1e-12))
    assert np.median(cos) > 0.9


def test_force_only_pushes_into_the_surface(setup):
    model, loc, truth = setup
    rng = np.random.default_rng(2)
    _, _, tau = _push(model, truth, rng.integers(truth.n), rng)
    c = _run(loc, tau)
    assert c.force @ c.normal <= 1e-6                 # pushes in, never pulls


def test_upper_arm_is_the_most_probable_link(setup):
    """Only pan + lift respond, so points on several links fit; the touched link must still
    collect the most probability."""
    model, loc, truth = setup
    rng = np.random.default_rng(3)
    idx = [i for i in range(truth.n) if truth.link_of_body[int(truth.body[i])] == "upper_arm"]
    hits = []
    for i in rng.choice(idx, 20, replace=False):
        _, _, tau = _push(model, truth, i, rng)
        hits.append(_run(loc, tau).link == "upper_arm")
    assert np.mean(hits) > 0.7, np.mean(hits)


def test_evidence_accumulates(setup):
    """Same point, the push direction wobbling: the second look is at least as sure."""
    model, loc, truth = setup
    rng = np.random.default_rng(4)
    idx = [i for i in range(truth.n) if truth.link_of_body[int(truth.body[i])] == "lower_arm"]
    i = rng.choice(idx)
    loc.reset()
    spreads = []
    for _ in range(6):
        _, _, tau = _push(model, truth, i, rng)
        spreads.append(loc.update(POSE, tau, SIGMA).spread_cm)
    assert spreads[-1] <= spreads[0] + 1e-6
