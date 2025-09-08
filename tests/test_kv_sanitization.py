import os
import json
import time
import pytest

from tools.cache_tracer import CacheTracer, ForensicLogger, FreeWithoutSanitize


def make_tracer(tmp_path):
    log_file = tmp_path / "kv_cache.log"
    return CacheTracer(log_path=log_file)


def test_cpu_numpy_sanitize_and_free(tmp_path):
    tracer = make_tracer(tmp_path)
    handle = tracer.allocate(
        tenant_id="t1",
        request_id="r1",
        model_id="m1",
        shape=(256,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(handle)
    cov = tracer.sanitize(handle, async_=False, verify=True)
    assert cov >= tracer.COVERAGE_THRESHOLD
    assert pytest.approx(tracer.attest_coverage(handle), rel=1e-6) == cov
    tracer.free(handle)

    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 0
    assert metrics["unsanitized_regions_count"] == 0


def test_free_without_sanitize_quarantines_and_raises(tmp_path):
    tracer = make_tracer(tmp_path)
    handle = tracer.allocate(
        tenant_id="t2",
        request_id="r2",
        model_id="m2",
        shape=(64,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    with pytest.raises(FreeWithoutSanitize):
        tracer.free(handle)
    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 1


def test_verification_failure_quarantines(tmp_path, monkeypatch):
    tracer = make_tracer(tmp_path)
    handle = tracer.allocate(
        tenant_id="t3",
        request_id="r3",
        model_id="m3",
        shape=(128,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(handle)

    # Force verification to report failure
    monkeypatch.setattr(tracer, "_verify_zero", lambda buf, samples=8: False)

    cov = tracer.sanitize(handle, async_=False, verify=True)
    assert cov == 0.0
    with pytest.raises(FreeWithoutSanitize):
        tracer.free(handle)

    # Forensic log should be valid
    res = ForensicLogger.verify_chain(str(tmp_path / "kv_cache.log"))
    assert res["ok"] is True


def test_prometheus_export_contains_metrics(tmp_path):
    tracer = make_tracer(tmp_path)
    # Create some activity so metrics are populated
    h = tracer.allocate(
        tenant_id="t4",
        request_id="r4",
        model_id="m4",
        shape=(32,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False)
    tracer.free(h)

    prom_path = tmp_path / "metrics.prom"
    tracer.export_metrics_prometheus(prom_path)
    text = prom_path.read_text()
    assert "kv_hygiene_unsanitized_regions" in text
    assert "kv_hygiene_quarantine_count" in text
    assert "kv_hygiene_min_coverage_pct" in text


def test_ttl_expiry_triggers_quarantine(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h = tracer.allocate(
        tenant_id="t5",
        request_id="r5",
        model_id="m5",
        shape=(16,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        ttl_sec=0.0,
    )
    tracer.mark_in_use(h)
    # Immediate TTL violation
    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 1


def test_reuse_policy_quarantine(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h = tracer.allocate(
        tenant_id="t6",
        request_id="r6",
        model_id="m6",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        max_reuse=0,
    )
    # First reuse increments to 1 > 0 → quarantine
    tracer.mark_reuse(h)
    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 1


def test_double_free_idempotent(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h = tracer.allocate(
        tenant_id="t7",
        request_id="r7",
        model_id="m7",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False)
    tracer.free(h)
    # Double free should be a no-op
    tracer.free(h)


def test_prometheus_lines(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h = tracer.allocate(
        tenant_id="t8",
        request_id="r8",
        model_id="m8",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False)
    tracer.free(h)

    p = tmp_path / "metrics.prom"
    tracer.export_metrics_prometheus(p)
    lines = p.read_text().strip().splitlines()
    assert any(l.startswith("kv_hygiene_unsanitized_regions ") for l in lines)
    assert any(l.startswith("kv_hygiene_quarantine_count ") for l in lines)
    assert any(l.startswith("kv_hygiene_min_coverage_pct ") for l in lines)
