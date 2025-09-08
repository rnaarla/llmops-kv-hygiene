import json
import os
import subprocess
import sys
import time
import pytest

from tools.cache_tracer import CacheTracer, ForensicLogger, UnknownHandle, FreeWithoutSanitize
from tools.activation_logger import ActivationLogger


def test_unique_handles_and_unknown_handle(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h1 = tracer.allocate(tenant_id="t1", request_id="r1", model_id="m1", shape=(8,), dtype="float32", device="cpu", framework="numpy")
    h2 = tracer.allocate(tenant_id="t2", request_id="r2", model_id="m2", shape=(8,), dtype="float32", device="cpu", framework="numpy")
    assert h1 != h2
    with pytest.raises(UnknownHandle):
        tracer.attest_coverage("nope")


def test_quarantine_when_threshold_unreachable(tmp_path):
    # Set threshold slightly above 100 to force quarantine despite full scrub
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log", coverage_threshold=100.01)
    h = tracer.allocate(tenant_id="t1", request_id="r1", model_id="m1", shape=(4,), dtype="float32", device="cpu", framework="numpy")
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=True)
    assert cov == 100.0
    with pytest.raises(FreeWithoutSanitize):
        tracer.free(h)

    res = ForensicLogger.verify_chain(str(tmp_path / "kv_cache.log"))
    assert res["ok"] is True


def test_forensic_rotation_and_chain(tmp_path, monkeypatch):
    # Force tiny max_bytes to trigger rotation
    log_file = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=log_file)
    # Monkeypatch logger to force rotation threshold small
    tracer.logger._max_bytes = 200

    for i in range(20):
        h = tracer.allocate(tenant_id="t", request_id=f"r{i}", model_id="m", shape=(4,), dtype="float32", device="cpu", framework="numpy")
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False)
        tracer.free(h)

    res = ForensicLogger.verify_chain(str(log_file))
    assert res["ok"] is True


def test_forensic_hmac_chain(tmp_path, monkeypatch):
    os.environ["FORENSIC_HMAC_SECRET"] = "secret"
    log_file = tmp_path / "kv_cache_hmac.log"
    t = CacheTracer(log_path=log_file)
    h = t.allocate(tenant_id="t", request_id="r", model_id="m", shape=(4,), dtype="float32", device="cpu", framework="numpy")
    t.mark_in_use(h)
    t.sanitize(h, async_=False)
    t.free(h)
    res = ForensicLogger.verify_chain(str(log_file))
    assert res["ok"] is True


def test_eviction_checker_cli_exit(tmp_path):
    # Generate metrics
    from tools.cache_tracer import CacheTracer
    t = CacheTracer(log_path=tmp_path / "kv_cache.log")
    h = t.allocate(tenant_id="t", request_id="r", model_id="m", shape=(16,), dtype="float32", device="cpu", framework="numpy")
    t.mark_in_use(h)
    t.sanitize(h, async_=False)
    try:
        t.free(h)
    except Exception:
        pass
    out_json = tmp_path / "coverage.json"
    t.export_metrics(out_json)

    # Run eviction_checker with strict thresholds to pass
    proc = subprocess.run([sys.executable, "tools/eviction_checker.py", str(out_json), "--coverage-min", "99.0", "--unsanitized-max", "0", "--quarantine-max", "0"], capture_output=True)
    assert proc.returncode == 0

    # Run with impossible coverage to force failure
    proc2 = subprocess.run([sys.executable, "tools/eviction_checker.py", str(out_json), "--coverage-min", "100.01"], capture_output=True)
    assert proc2.returncode != 0


def test_activation_logger_anomaly_and_rate_limit(tmp_path):
    path = tmp_path / "activations.jsonl"
    logger = ActivationLogger(out_path=str(path), z_threshold=3.0)
    import numpy as np

    # warm-up normal values
    for _ in range(50):
        logger.observe("layer1", (0.0, 1.0, 2.0, 128))
    # anomaly
    flagged = logger.observe("layer1", (100.0, 1.0, 100.0, 128), rate_limit_hz=None)
    assert flagged is True

    # rate limit: many logs quickly should not create too many entries
    before = sum(1 for _ in open(path))
    for _ in range(20):
        logger.observe("layer1", (0.1, 1.0, 2.0, 128), rate_limit_hz=5.0)
    after = sum(1 for _ in open(path))
    assert after - before <= 5
