from __future__ import annotations

import random
import time
from typing import Any

# Optional dependencies treated as Any when absent to simplify typing
try:  # pragma: no cover - optional dependency
    import numpy as _np  # noqa: F401

    np: Any = _np
except Exception:  # pragma: no cover
    np = None
try:  # pragma: no cover - optional dependency
    import torch as _torch  # noqa: F401

    torch: Any = _torch
except Exception:  # pragma: no cover
    torch = None

RNG = random.Random()  # noqa: S311 - non-crypto RNG sufficient for sampling


class SanitizationResult:
    __slots__ = ("scrubbed_bytes", "duration_ms", "coverage_pct", "scheduled_async")

    def __init__(
        self,
        scrubbed_bytes: int,
        duration_ms: float,
        coverage_pct: float,
        *,
        scheduled_async: bool = False,
    ):
        self.scrubbed_bytes = scrubbed_bytes
        self.duration_ms = duration_ms
        self.coverage_pct = coverage_pct
        self.scheduled_async = scheduled_async


def _sample_indices(n: int, samples: int) -> list[int]:
    if samples <= 0:
        return []
    if samples >= n:
        return list(range(n))
    # Stable deterministic sample: shuffle seeded by n & samples
    # Non-cryptographic deterministic sample selection; seed mixes size & sample count
    local = random.Random(n * 31 + samples * 17)  # noqa: S311
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
        except Exception:  # pragma: no cover - defensive size calc
            import logging

            logging.debug("zeroize_cpu: failed to compute tensor size", exc_info=True)
            nbytes = 0
        try:
            if hasattr(arr, "zero_"):
                arr.zero_()
            else:  # pragma: no cover - unlikely branch
                # Fallback iterate (slow, but safe)
                for _i in range(arr.numel()):
                    arr.view(-1)[_i] = 0
        except Exception:  # pragma: no cover - defensive zeroing
            import logging

            logging.debug("zeroize_cpu: tensor zeroing failed", exc_info=True)
            return 0
        return nbytes
    # NumPy ndarray
    if np is not None and hasattr(arr, "fill") and hasattr(arr, "nbytes"):
        nbytes = int(arr.nbytes)
        try:
            arr.fill(0)
        except Exception:  # pragma: no cover - numpy fill
            import logging

            logging.debug("zeroize_cpu: numpy fill failed", exc_info=True)
            return 0
        return nbytes
    # Unsupported type or missing dependency
    return 0


def zeroize_cuda(buf: Any, async_: bool) -> bool:
    """Zeroize a CUDA tensor.

    Returns True if `zero_` was submitted on ``buf.stream_obj`` (async path).
    """
    if torch is None:  # pragma: no cover
        return False
    t = getattr(buf, "_tensor", None)
    if t is None or not isinstance(t, torch.Tensor) or not t.is_cuda:  # pragma: no cover
        return False
    stream = getattr(buf, "stream_obj", None)
    if async_ and stream is not None:
        with torch.cuda.stream(stream):
            t.zero_()
        return True
    t.zero_()
    return False


def verify_zero(buf: Any, samples: int | None) -> bool:
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
    if torch is not None and hasattr(tensor, "numel"):  # pragma: no cover - GPU/torch path optional
        flat = tensor.view(-1)
        idxs = _sample_indices(flat.numel(), samples)
        return all(flat[i].item() == 0 for i in idxs)
    return True


def sanitize_sync(buf: Any, *, verify: bool, samples: int | None) -> SanitizationResult:
    start = time.time()
    scrubbed = 0
    scheduled_async = False
    if buf.device.startswith("cuda"):
        cuda_async = bool(getattr(buf, "cuda_async_zeroize", False))
        scheduled_async = zeroize_cuda(buf, async_=cuda_async)
        if scheduled_async:
            verify = False
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
    if hasattr(buf, "cuda_async_zeroize"):
        buf.cuda_async_zeroize = False
    return SanitizationResult(scrubbed, duration_ms, coverage_pct, scheduled_async=scheduled_async)
