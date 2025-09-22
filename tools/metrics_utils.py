from __future__ import annotations
from typing import Iterable, Sequence


def percentile(data: Sequence[float], pct: float) -> float:
    """Compute percentile with linear interpolation.

    Raises ValueError for empty data or pct outside [0,100].
    """
    if not data:
        raise ValueError("empty dataset")
    if pct < 0 or pct > 100:
        raise ValueError("percentile out of range")
    arr = sorted(float(x) for x in data)
    if len(arr) == 1:
        return arr[0]
    if pct == 0:
        return arr[0]
    if pct == 100:
        return arr[-1]
    k = (len(arr) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1
