"""A matplotlib window of scrolling traces, fed from a worker thread.

The worker owns the serial bus (read -> detect -> write goals -> log) and calls
`live.push(t, {...})`. The main thread only draws, so plotting never slows the
control loop. Closing the window stops the worker.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402


@dataclass
class Panel:
    title: str
    keys: list[str]                       # series names pushed by the worker
    ylabel: str = "raw units"
    ylim: tuple[float, float] | None = None
    hlines: tuple[float, ...] = ()        # e.g. detection thresholds


class Live:
    def __init__(self, title: str, panels: list[Panel], window_s: float = 10.0, maxlen: int = 4000):
        self.title, self.panels, self.window_s = title, panels, window_s
        keys = {k for p in panels for k in p.keys}
        self._t = deque(maxlen=maxlen)
        self._v = {k: deque(maxlen=maxlen) for k in keys}
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.banner = ("starting...", "grey")
        self.hz = 0.0
        self.error: BaseException | None = None

    def push(self, t: float, values: dict[str, float]) -> None:
        with self._lock:
            self._t.append(t)
            for k, d in self._v.items():
                v = values.get(k)
                d.append(float("nan") if v is None else v)

    def set_banner(self, text: str, color: str = "black") -> None:
        self.banner = (text, color)

    def run(self, worker: Callable[["Live"], None], duration: float | None = None) -> None:
        """Start `worker(live)` in a thread and show the window until it closes
        (or `duration` seconds pass)."""
        if duration:
            timer = threading.Timer(duration, self.stop.set)
            timer.daemon = True
            timer.start()

        def _wrapped():
            try:
                worker(self)
            except BaseException as e:  # surface worker crashes in the main thread
                self.error = e
            finally:
                self.stop.set()

        th = threading.Thread(target=_wrapped, daemon=True)
        fig, axes = plt.subplots(len(self.panels), 1, sharex=True, figsize=(11, 2.3 * len(self.panels) + 1),
                                 squeeze=False)
        axes = axes[:, 0]
        fig.canvas.manager.set_window_title(self.title)
        lines = []
        for ax, p in zip(axes, self.panels):
            ax.set_title(p.title, fontsize=10, loc="left")
            ax.set_ylabel(p.ylabel, fontsize=8)
            ax.grid(alpha=0.3)
            for h in p.hlines:
                ax.axhline(h, color="red", ls="--", lw=0.8, alpha=0.6)
            for k in p.keys:
                (ln,) = ax.plot([], [], lw=1.2, label=k.split("/")[-1] if "/" in k else k)
                lines.append((ax, p, k, ln))
            ax.legend(loc="upper left", fontsize=7, ncol=6)
        axes[-1].set_xlabel("time (s)")
        banner = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=15, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        def _draw(_):
            if self.stop.is_set():
                plt.close(fig)
                return []
            with self._lock:
                t = list(self._t)
                vals = {k: list(d) for k, d in self._v.items()}
            if t:
                t1 = t[-1]
                for ax, p, k, ln in lines:
                    ln.set_data(t, vals[k])
                for ax, p in zip(axes, self.panels):
                    ax.set_xlim(max(0, t1 - self.window_s), max(self.window_s, t1))
                    if p.ylim:
                        ax.set_ylim(*p.ylim)
                    else:
                        ax.relim()
                        ax.autoscale_view(scalex=False)
            text, color = self.banner
            banner.set_text(f"{text}    [{self.hz:.0f} Hz]")
            banner.set_color(color)
            return []

        fig.canvas.mpl_connect("close_event", lambda _e: self.stop.set())
        th.start()
        self._anim = FuncAnimation(fig, _draw, interval=60, cache_frame_data=False)
        plt.show()
        self.stop.set()
        th.join(timeout=3)
        if self.error and not isinstance(self.error, (KeyboardInterrupt, SystemExit)):
            raise self.error


def run_headless(worker: Callable[[Live], None], duration: float | None) -> Live:
    """Same worker, no window (for logging runs or SSH). Ctrl+C to stop."""
    live = Live("headless", [])
    if duration:
        timer = threading.Timer(duration, live.stop.set)
        timer.daemon = True
        timer.start()
    try:
        worker(live)
    except KeyboardInterrupt:
        pass
    finally:
        live.stop.set()
    return live
