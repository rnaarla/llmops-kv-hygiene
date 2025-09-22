import json

import pytest

from tools.cache_tracer import CacheTracer, FreeWithoutSanitizeError, UnknownHandleError


def test_unknown_handle_error():
    tracer = CacheTracer()
    with pytest.raises(UnknownHandleError):
        tracer.attest_coverage("bad-handle")


def test_free_without_coverage_quarantines(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv.log", coverage_threshold=99.9)
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    # Intentionally skip sanitize to force quarantine on free
    with pytest.raises(FreeWithoutSanitizeError):
        tracer.free(h)
    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 1


def test_reuse_limit_quarantine(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv.log")
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        max_reuse=0,
    )
    tracer.mark_in_use(h)
    tracer.mark_reuse(h)  # reuse_count=1 > max_reuse=0 triggers quarantine
    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 1


def test_metrics_percentiles_and_export(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv.log")
    # Create multiple buffers to produce durations list
    handles = [
        tracer.allocate(
            tenant_id="t",
            request_id=str(i),
            model_id="m",
            shape=(2,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        for i in range(3)
    ]
    for h in handles:
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    metrics = tracer.get_metrics()
    assert metrics["sanitize_duration_p50_ms"] >= 0
    assert metrics["sanitize_duration_p95_ms"] >= metrics["sanitize_duration_p50_ms"]
    out_json = tmp_path / "metrics.json"
    tracer.export_metrics(out_json)
    data = json.loads(out_json.read_text())
    assert "avg_coverage_pct" in data


def test_export_prometheus(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv.log")
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
    tracer.free(h)
    prom = tmp_path / "metrics.prom"
    tracer.export_metrics_prometheus(prom)
    text = prom.read_text()
    assert "kv_hygiene_total_buffers" in text
