"""Turning raw load/current into "something is touching me".

Pure numpy, no hardware, so it is unit-tested in tests/test_detect.py.

The core idea everywhere: a servo always needs SOME drive (gravity, friction,
inertia). Touch is the part you did not expect. So:

    residual = measured - expected
    z        = residual / noise
    touch    = |z| above a threshold (with hysteresis so it does not flicker)

The detectors differ only in how they get "expected":
    Baseline       arm holding still: learn mean/std for a couple of seconds
    PhaseBaseline  arm repeating the same motion: learn expected load per phase
    (IDEAS.md)     arbitrary motion: a model of gravity + friction + inertia
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Baseline:
    """Learn mean/std per channel while nothing touches the arm, then score new samples."""

    def __init__(self, min_std: float = 3.0):
        self.min_std = min_std      # floor so a very quiet channel does not fire on 1 tick
        self._buf: list[np.ndarray] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def add(self, x) -> None:
        self._buf.append(np.asarray(x, float))

    def freeze(self) -> None:
        a = np.stack(self._buf)
        self.mean = a.mean(axis=0)
        self.std = np.maximum(a.std(axis=0), self.min_std)
        self._buf.clear()

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def z(self, x) -> np.ndarray:
        return (np.asarray(x, float) - self.mean) / self.std


class PhaseBaseline:
    """For a repeated motion (e.g. a sine wiggle): expected load depends on WHERE in
    the cycle you are. Bin the cycle into `bins` slots and learn mean/std per slot over
    the first few cycles. Gravity, friction and inertia all repeat with the motion, so
    they all cancel. Only the hand pushing does not."""

    def __init__(self, n_channels: int, bins: int = 40, min_std: float = 4.0):
        self.bins = bins
        self.min_std = min_std
        self.sum = np.zeros((bins, n_channels))
        self.sq = np.zeros((bins, n_channels))
        self.n = np.zeros(bins)
        self.frozen = False

    def _bin(self, phase: float) -> int:
        return int((phase % 1.0) * self.bins) % self.bins

    def add(self, phase: float, x) -> None:
        b = self._bin(phase)
        x = np.asarray(x, float)
        self.sum[b] += x
        self.sq[b] += x * x
        self.n[b] += 1

    def freeze(self) -> None:
        n = np.maximum(self.n, 1)[:, None]
        self.mean = self.sum / n
        var = self.sq / n - self.mean ** 2
        self.std = np.maximum(np.sqrt(np.maximum(var, 0)), self.min_std)
        # an empty bin borrows from its neighbour so it does not read as a huge residual
        for b in np.where(self.n == 0)[0]:
            self.mean[b] = self.mean[b - 1]
            self.std[b] = self.std[b - 1]
        self.frozen = True

    def z(self, phase: float, x) -> np.ndarray:
        b = self._bin(phase)
        return (np.asarray(x, float) - self.mean[b]) / self.std[b]


class Hysteresis:
    """Per-channel on/off with two thresholds, so a signal hovering at the threshold
    does not chatter. Turns on above `on`, off only below `off`."""

    def __init__(self, n_channels: int, on: float = 5.0, off: float = 2.5):
        self.on, self.off = on, off
        self.state = np.zeros(n_channels, bool)

    def update(self, z) -> np.ndarray:
        a = np.abs(np.asarray(z, float))
        self.state = np.where(self.state, a > self.off, a > self.on)
        return self.state.copy()


# ------------------------------------------------------------------ gripper squeeze
@dataclass
class SqueezeResult:
    contact_idx: int | None       # first sample where the object pushes back
    contact_pos: float | None     # closing travel at contact (ticks)
    stiffness: float | None       # d(load)/d(travel) right after contact, load units per tick
    yield_idx: int | None         # sample where stiffness collapsed (skin split / crush), if any
    peak_load: float | None


def _slope(x, y) -> float | None:
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 4 and np.ptp(x) > 0 else None


def analyse_squeeze(pen, grip_load, free_load_std: float = 6.0, k_contact: float = 5.0,
                    fit_ticks: float = 30.0, yield_ratio: float = 0.35, yield_ticks: float = 25.0,
                    yield_persist: int = 3, drop_frac: float = 0.25) -> SqueezeResult:
    """Analysis of a closing gripper (call it on the whole run, or on a growing log).

    pen        closing travel so far in ticks (0 at start, grows as the jaws close)
    grip_load  load projected onto the closing direction (positive = resisting closing)

    Contact = grip_load rises k_contact * noise above its free-closing level and STAYS
    up for 3 samples. Stiffness = least-squares slope of load vs travel over the first
    `fit_ticks` of travel after contact. That is the "how firm is this fruit" number.
    Yield (the object gave way; for a tomato, the skin splitting. Stop BEFORE this) is
    either of:
      sudden   grip falls `drop_frac` below its running peak while still closing
      gradual  the slope over a sliding `yield_ticks` travel window drops below
               yield_ratio * stiffness for `yield_persist` windows in a row
    Windows are in ticks of travel, not samples, so the result does not depend on
    loop rate or closing speed, and a stalled jaw (no travel) never reads as a yield.
    """
    pen = np.asarray(pen, float)
    f = np.asarray(grip_load, float)
    n = len(f)
    if n < 10:
        return SqueezeResult(None, None, None, None, None)
    free = np.median(f[: max(5, n // 10)])
    thr = free + k_contact * free_load_std
    above = f > thr
    contact = None
    for i in range(n - 2):
        if above[i] and above[i + 1] and above[i + 2]:
            contact = i
            break
    if contact is None:
        return SqueezeResult(None, None, None, None, float(f.max()))
    # back off to where the rise began (last sample within 1 sigma of free)
    start = contact
    while start > 0 and f[start - 1] > free + free_load_std:
        start -= 1
    p0 = pen[start]
    fit = (pen >= p0) & (pen <= p0 + fit_ticks)
    fit[:start] = False
    k = _slope(pen[fit], f[fit])
    y = None
    # Sudden yield: grip drops well below its running peak while the jaw keeps closing.
    # (A real skin split makes the jaw SURGE forward, so there are too few samples per
    # travel window for the slope test below to see it.) 5-sample median kills spikes.
    fs = np.array([np.median(f[max(0, i - 4): i + 1]) for i in range(n)])
    peak = -np.inf
    for i in range(start, n):
        peak = max(peak, fs[i])
        drop = peak - fs[i]
        if peak > thr and drop > drop_frac * peak and drop > 4 * free_load_std and pen[i] > pen[int(np.argmax(fs[start:i + 1])) + start]:
            y = i
            break
    if y is None and k and k > 0:
        end_fit = start + int(fit.sum())
        run = 0
        for i in range(end_fit, n):
            w = (pen >= pen[i]) & (pen <= pen[i] + yield_ticks)
            w[:i] = False
            if pen[w].size and pen[w].max() - pen[i] < 0.8 * yield_ticks:
                break                      # not enough travel left after i to judge
            s = _slope(pen[w], f[w])
            run = run + 1 if (s is not None and s < yield_ratio * k) else 0
            if run >= yield_persist:
                y = i - yield_persist + 1
                break
    return SqueezeResult(start, float(p0), k, y, float(f.max()))
