#!/usr/bin/env python3
"""Micro-benchmark for CPU sanitize path (optional; not run in CI by default)."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import sanitizer as s  # noqa: E402


class _Buf:
    __slots__ = ("_tensor", "device", "nbytes")

    def __init__(self, arr, device: str = "cpu") -> None:
        self._tensor = arr
        self.device = device
        self.nbytes = int(arr.nbytes)


def main() -> None:
    import numpy as np

    sizes = (1024, 4096, 65536)
    reps = 50
    for n in sizes:
        times: list[float] = []
        for _ in range(reps):
            arr = (np.arange(n, dtype="float32") % 7).astype("float32")
            buf = _Buf(arr)
            t0 = time.perf_counter()
            s.sanitize_sync(buf, verify=True, samples=16)
            times.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"n={n:6d}  p50={statistics.median(times):.3f}ms  "
            f"p95={sorted(times)[int(0.95 * (len(times) - 1))]:.3f}ms"
        )


if __name__ == "__main__":
    main()
