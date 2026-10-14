"""One NEW timestamped CSV per run under data/. Never appends to an old file."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .source import SIGNALS, Sample

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class CsvLogger:
    def __init__(self, script: str, joints: list[str], extra: list[str] = (), out_dir: Path = DATA_DIR,
                 fake: bool = False):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = "_fake" if fake else ""
        path = out_dir / f"{script}{tag}_{stamp}.csv"
        i = 1
        while path.exists():                      # two runs in the same second
            path = out_dir / f"{script}{tag}_{stamp}_{i}.csv"
            i += 1
        self.path = path
        self.joints = joints
        self.extra = list(extra)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["t"] + [f"{sig}_{j}" for sig in SIGNALS for j in joints] + self.extra)
        self.rows = 0

    def write(self, s: Sample, extra: dict | None = None) -> None:
        row = [f"{s.t:.4f}"]
        for sig in SIGNALS:
            d = getattr(s, sig)
            row += ["" if d.get(j) is None else d[j] for j in self.joints]
        extra = extra or {}
        row += [extra.get(k, "") for k in self.extra]
        self._w.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._f.close()
        print(f"[tactile] logged {self.rows} rows -> {self.path}")
