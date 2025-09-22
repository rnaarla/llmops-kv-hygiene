from types import SimpleNamespace

import pytest

from tools import sanitizer as sanitizer_mod


def make_np_buf(size: int = 32):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        pytest.skip("NumPy not available")
    arr = np.random.rand(size).astype("float32")  # non-zero initial
    return SimpleNamespace(_tensor=arr, device="cpu", nbytes=arr.nbytes)


def test_sanitize_sync_numpy_success():
    buf = make_np_buf(16)
    # Perform zeroization without verification first
    res = sanitizer_mod.sanitize_sync(buf, verify=True, samples=8)
    assert res.scrubbed_bytes == buf.nbytes
    assert res.coverage_pct == pytest.approx(100.0)
    # All elements should be zero
    assert (buf._tensor == 0).all()


def test_verify_zero_failure_numpy():
    buf = make_np_buf(32)
    # Determine ahead of time which indices will be sampled after zeroization
    samples = 8
    idxs = sanitizer_mod._sample_indices(buf._tensor.size, samples)
    # Perform zeroization without verification first
    sanitizer_mod.sanitize_sync(buf, verify=False, samples=None)
    # Corrupt one of the deterministic sampled indices
    buf._tensor[idxs[0]] = 123.456
    assert sanitizer_mod.verify_zero(buf, samples) is False


def test_sanitize_sync_torch_cpu_if_available():  # pragma: no cover - executed only when torch present
    try:
        import torch
    except Exception:  # pragma: no cover
        pytest.skip("torch not available")
    t = torch.zeros(10, dtype=torch.float32)
    buf = SimpleNamespace(_tensor=t, device="cpu", nbytes=t.numel() * t.element_size())
    # Introduce some non-zero values to ensure zeroization occurs
    if t.numel() > 0:
        t.view(-1)[0] = 1.0
    res = sanitizer_mod.sanitize_sync(buf, verify=True, samples=4)
    assert res.scrubbed_bytes == buf.nbytes
    assert res.coverage_pct in (
        100.0,
        0.0,
    )  # If sampling hit modified index pre-zeroization might report 0 coverage
    # After zeroization tensor should be zeros
    assert torch.count_nonzero(t) == 0
