from __future__ import annotations
from typing import Any, List, Optional
import time
import random

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

RNG = random.Random()


class SanitizationResult:
    __slots__ = ("scrubbed_bytes", "duration_ms", "coverage_pct")

    def __init__(self, scrubbed_bytes: int, duration_ms: float, coverage_pct: float):
        self.scrubbed_bytes = scrubbed_bytes
        self.duration_ms = duration_ms
        self.coverage_pct = coverage_pct


def _sample_indices(n: int, samples: int) -> List[int]:
    if samples <= 0:
        return []
    if samples >= n:
        return list(range(n))
    # Stable deterministic sample: shuffle seeded by n & samples
    local = random.Random(n * 31 + samples * 17)
    return sorted(local.sample(range(n), samples))


def zeroize_cpu(buf: Any) -> int:
    """Zeroize a CPU resident tensor/array in-place.

    Supports both NumPy ndarrays and torch CPU tensors. Returns number of bytes scrubbed.
    """
    arr = buf._tensor
    if arr is None:
        return 0
    # Torch tensor (CPU)
    if (
        torch is not None
        and hasattr(arr, "device")
        and getattr(arr.device, "type", None) == "cpu"
        and hasattr(arr, "numel")
    ):
        try:
            nbytes = int(arr.numel() * arr.element_size())
        except Exception:  # pragma: no cover - defensive
            nbytes = 0
        try:
            if hasattr(arr, "zero_"):
                arr.zero_()
            else:  # pragma: no cover - unlikely branch
                # Fallback iterate (slow, but safe)
                for _i in range(arr.numel()):
                    arr.view(-1)[_i] = 0
        except Exception:  # pragma: no cover - defensive
            return 0
        return nbytes
    # NumPy ndarray
    if np is not None and hasattr(arr, "fill") and hasattr(arr, "nbytes"):
        nbytes = int(arr.nbytes)
        try:
            arr.fill(0)
        except Exception:  # pragma: no cover
            return 0
        return nbytes
    # Unsupported type or missing dependency
    return 0


def zeroize_cuda(buf: Any, async_: bool) -> bool:
    if torch is None or not isinstance(
        buf._tensor, type(getattr(torch, "tensor")([0]))
    ):  # pragma: no cover - no torch
        return False
    t = buf._tensor
    if async_:
        t.zero_()  # still synchronous unless stream used; placeholder
        return True
    else:
        t.zero_()
        return False


def verify_zero(buf: Any, samples: Optional[int]) -> bool:
    if samples is None or samples <= 0:
        return True
    tensor = buf._tensor
    if tensor is None:
        return True
    # NumPy ndarray path (explicit isinstance for reliability)
    if np is not None and isinstance(tensor, np.ndarray):
        flat = tensor.reshape(-1)
        idxs = _sample_indices(flat.size, samples)
        return all(flat[i] == 0 for i in idxs)
    if torch is not None and hasattr(
        tensor, "numel"
    ):  # pragma: no cover - GPU/torch path optional
        flat = tensor.view(-1)
        idxs = _sample_indices(flat.numel(), samples)
        return all(flat[i].item() == 0 for i in idxs)
    return True


def sanitize_sync(
    buf: Any, *, verify: bool, samples: Optional[int]
) -> SanitizationResult:
    start = time.time()
    scrubbed = 0
    if buf.device.startswith("cuda"):
        scheduled = zeroize_cuda(buf, async_=False)
        if (
            scheduled
        ):  # pragma: no cover - branch not expected with simplified zeroization
            pass
        # For simplicity CUDA path returns full logical size if tensor exists
        tensor = buf._tensor
        if tensor is not None:
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                scrubbed = tensor.numel() * tensor.element_size()
    else:
        scrubbed = zeroize_cpu(buf)
    duration_ms = (time.time() - start) * 1000.0
    total = buf.nbytes or 1
    coverage_pct = (scrubbed / total) * 100.0
    if verify:
        ok = verify_zero(buf, samples)
        if not ok:
            coverage_pct = 0.0
    return SanitizationResult(scrubbed, duration_ms, coverage_pct)
