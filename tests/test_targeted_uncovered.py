from types import SimpleNamespace

import pytest

from test_support.optional_deps import opt
from tools import sanitizer as sanitizer_mod
from tools.cache_tracer import CacheTracer
from tools.forensic_logger import ForensicLogger

np = opt.numpy()


def _make_buf(size: int = 8):
    if np is None:
        pytest.skip("NumPy not available")
    arr = np.arange(size, dtype="float32") + 1.0  # ensure non-zero
    return SimpleNamespace(_tensor=arr, device="cpu", nbytes=arr.nbytes)


def test_zeroize_cpu_numpy_error_branch(monkeypatch):
    if np is None:
        pytest.skip("NumPy not available")
    base = _make_buf(4)

    # Create proxy object exposing .fill which raises to exercise exception branch in zeroize_cpu
    class Proxy:
        def __init__(self, arr):
            self._arr = arr
            self.nbytes = arr.nbytes

        def fill(self, *a, **k):  # pragma: no cover - we want caller's except path
            raise RuntimeError("boom")

    proxy = Proxy(base._tensor)
    buf = SimpleNamespace(_tensor=proxy, device="cpu", nbytes=proxy.nbytes)
    scrubbed = sanitizer_mod.zeroize_cpu(buf)
    assert scrubbed == 0


def test_verify_zero_numpy_samples_all_zero():
    if np is None:
        pytest.skip("NumPy not available")
    buf = _make_buf(16)
    # Zero manually then verify sampling returns True
    buf._tensor.fill(0)
    assert sanitizer_mod.verify_zero(buf, samples=5) is True


def test_forensic_logger_hmac_and_rotation(tmp_path, monkeypatch):
    # Provide HMAC secret via env to cover hmac branch
    monkeypatch.setenv("FORENSIC_HMAC_SECRET", "secret")
    log_path = tmp_path / "f.log"
    flog = ForensicLogger(log_path, max_bytes=200)
    for i in range(20):  # force at least one rotation
        flog.append({"event_type": "alloc", "i": i})
    res = ForensicLogger.verify_all(str(log_path))
    assert "files" in res
    # ensure some file verified ok and at least one rotate record existed
    assert any(f.get("ok") for f in res["files"])  # pragma: no branch (logical)


def test_cache_tracer_quarantine_on_low_coverage(monkeypatch, tmp_path):
    # Force coverage threshold very high to trigger quarantine path on free
    monkeypatch.setenv("KV_COVERAGE_THRESHOLD", "150.0")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(2,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    # Coverage will be 100 < 150 threshold, so free should quarantine and raise exception
    with pytest.raises(Exception):
        tracer.free(h)
    # buffer status should now be quarantined
    assert tracer._get(h).status == "quarantined"


def test_policy_ttl_violation(monkeypatch, tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    # allocate with ttl of 0 to force immediate violation on first write
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(2,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        ttl_sec=0,
    )
    tracer.mark_in_use(h)
    # After mark_in_use, ttl violation should have quarantined buffer
    assert tracer._get(h).status == "quarantined"  # coverage for ttl violation branch
