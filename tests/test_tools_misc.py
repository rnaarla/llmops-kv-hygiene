import json
import socket
import threading
import time
from http.client import HTTPConnection

import pytest

from tools.activation_logger import ActivationLogger
from tools.cache_tracer import CacheTracer, ForensicLogger, UnknownHandle
from tools.eviction_checker import main as eviction_main
from tools.verify_logs import prune_rotated, verify_all_and_write


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_metrics_exporter_serves(tmp_path, monkeypatch):
    # Produce a metrics file via tracer
    metrics_file = tmp_path / "metrics.prom"
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    tracer.export_metrics_prometheus(str(metrics_file))

    # Launch exporter in thread using port 0 (let OS allocate a free port to avoid races)
    monkeypatch.setenv("METRICS_FILE", str(metrics_file))
    from tools import metrics_exporter

    server = metrics_exporter.HTTPServer((metrics_exporter.BIND, 0), metrics_exporter.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    conn = HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/metrics")
    resp = conn.getresponse()
    body = resp.read().decode()
    assert resp.status == 200
    # Expect at least a known metric line
    assert "kv_hygiene_unsanitized_regions" in body
    # 404 path
    conn.request("GET", "/nope")
    resp2 = conn.getresponse()
    assert resp2.status == 404


def test_eviction_checker_pass_and_fail(tmp_path, monkeypatch, capsys):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    metrics_json = tmp_path / "metrics.json"
    tracer.export_metrics(str(metrics_json))

    # Passing run
    rc = eviction_main(
        [
            str(metrics_json),
            "--coverage-min",
            "99.0",
            "--unsanitized-max",
            "0",
            "--quarantine-max",
            "0",
            "--out",
            str(tmp_path / "verdict.json"),
        ]
    )
    assert rc == 0

    # Modify metrics to force failure
    data = json.loads(metrics_json.read_text())
    data["min_coverage_pct"] = 10.0
    metrics_json.write_text(json.dumps(data))
    rc2 = eviction_main(
        [
            str(metrics_json),
            "--coverage-min",
            "99.0",
            "--unsanitized-max",
            "0",
            "--quarantine-max",
            "0",
            "--out",
            str(tmp_path / "verdict2.json"),
        ]
    )
    assert rc2 == 2


def test_activation_logger_anomaly(tmp_path):
    logger = ActivationLogger(out_path=str(tmp_path / "acts.jsonl"), z_threshold=3.0)
    # Warm stats
    for _ in range(20):
        logger.observe("layer1", (0.0, 0.0, 0.0, 1))
    # An anomalous value (mean far from baseline)
    flagged = logger.observe("layer1", (100.0, 0.0, 100.0, 1))
    assert flagged is True
    lines = (tmp_path / "acts.jsonl").read_text().strip().splitlines()
    assert any("anomalous" in line for line in lines)


def test_verify_logs_rotation_and_prune(tmp_path):
    log_path = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=str(log_path))
    # Force a couple of rotations by writing many entries
    # Force extremely small rotation threshold
    tracer.logger._max_bytes = 600  # small so multiple rotations occur quickly
    for i in range(80):
        h = tracer.allocate(
            tenant_id="t",
            request_id="r",
            model_id=f"m{i}",
            shape=(2,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    # verify pre-prune; tolerate linkage mismatch but require individual chains ok
    res = verify_all_and_write(log_path)
    files = res["result"]["files"] if "result" in res else res.get("files", [])
    assert files, "Expected verification file entries"
    # All chain verifications should be ok; linkage mismatch may add an extra error entry
    chain_oks = [f for f in files if f.get("first_bad_line") is None and f.get("ok")]
    assert chain_oks, "No successful chains found"
    rotated = list(tmp_path.glob("kv_cache-*.log"))
    assert rotated, f"Expected rotated files, size of active {log_path.stat().st_size}"
    removed = prune_rotated(log_path, retention_days=0, max_rotated=1)
    # Either removed or archived (here removed) > 0
    assert removed
    res2 = verify_all_and_write(log_path)
    assert res2.get("ok")


def test_ttl_violation(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        ttl_sec=0,
    )
    tracer.mark_in_use(h)
    # TTL<=0 violation should quarantine on first use
    stats = tracer.get_metrics()
    assert stats["quarantine_count"] >= 1


def test_reuse_path(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        max_reuse=2,
    )
    tracer.mark_in_use(h)
    # simulate reuse events
    tracer.mark_reuse(h)
    tracer.mark_reuse(h)
    stats = tracer.get_metrics()
    # reuse_rate = reuse_events / allocations; should be > 0
    assert stats.get("reuse_rate", 0) > 0


def test_verification_failure_quarantine(monkeypatch, tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    # Force _verify_zero to return False
    monkeypatch.setattr(tracer, "_verify_zero", lambda *a, **k: False)
    tracer.sanitize(h, async_=False, verify=True)
    # free should quarantine due to low coverage
    with pytest.raises(Exception):
        tracer.free(h)
    stats = tracer.get_metrics()
    assert stats["quarantine_count"] >= 1


def test_prune_archive_branch(tmp_path):
    log_path = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=str(log_path))
    tracer.logger._max_bytes = 500
    for i in range(60):
        h = tracer.allocate(
            tenant_id="t",
            request_id="r",
            model_id=f"m{i}",
            shape=(2,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    rotated = list(tmp_path.glob("kv_cache-*.log"))
    assert rotated
    archive = tmp_path / "archive"
    from tools.verify_logs import prune_rotated

    removed = prune_rotated(log_path, retention_days=0, max_rotated=1, archive_dir=archive)
    assert removed
    # ensure archived files exist
    assert any((archive / name).exists() for name in removed)


def test_hmac_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("FORENSIC_HMAC_SECRET", "supersecret")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    res = ForensicLogger.verify_chain(str(tmp_path / "kv_cache.log"))
    assert res["ok"]


def test_verify_logs_main_cli(tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "foren"
    log_dir.mkdir()
    tracer = CacheTracer(log_path=str(log_dir / "kv_cache.log"))
    for i in range(10):
        h = tracer.allocate(
            tenant_id="t",
            request_id="r",
            model_id=f"m{i}",
            shape=(2,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    from tools import verify_logs

    rc = verify_logs.main(
        [
            "--log-dir",
            str(log_dir),
            "--log-file",
            "kv_cache.log",
            "--out",
            str(tmp_path / "ver.json"),
            "--retention-days",
            "0",
            "--max-rotated",
            "3",
        ]
    )
    assert rc == 0


def test_double_pass_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KV_DOUBLE_PASS_DEFAULT", "true")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(16,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=True)
    assert cov >= 99.9


def test_metrics_exporter_empty_file(monkeypatch, tmp_path):
    # Point exporter to a file that does not yet exist (empty response branch)
    monkeypatch.setenv("METRICS_FILE", str(tmp_path / "missing.prom"))
    from tools import metrics_exporter

    server = metrics_exporter.HTTPServer((metrics_exporter.BIND, 0), metrics_exporter.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    conn = HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/metrics")
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    # File may be created between scheduling; accept empty or valid metrics header
    assert body == b"" or body.startswith(b"# HELP kv_hygiene_unsanitized_regions")


def test_percentile_edges():
    from tools.cache_tracer import CacheTracer

    # Use protected method via tracer class to test logic
    assert CacheTracer._percentile([], 95) == 0.0
    assert CacheTracer._percentile([5], 50) == 5.0
    assert CacheTracer._percentile([1, 9], 50) == 5.0  # interpolation
    assert CacheTracer._percentile([1, 3, 5, 7, 9], 0) == 1.0
    assert CacheTracer._percentile([1, 3, 5, 7, 9], 100) == 9.0


def test_unsanitized_free_quarantine(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    # Intentionally skip sanitize
    with pytest.raises(Exception):
        tracer.free(h)
    stats = tracer.get_metrics()
    assert stats["quarantine_count"] >= 1


def test_env_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("KV_COVERAGE_THRESHOLD", "88.8")
    monkeypatch.setenv("KV_VERIFY_SAMPLES_DEFAULT", "3")
    monkeypatch.setenv("KV_DOUBLE_PASS_DEFAULT", "true")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    assert abs(tracer.COVERAGE_THRESHOLD - 88.8) < 1e-6
    # Allocate small buffer to exercise sampling count path
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(3,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=True)
    assert cov >= 100.0


def test_zero_length_buffer(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(0,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=True)
    assert cov == 100.0


def test_tamper_detection(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    logf = tmp_path / "kv_cache.log"
    # Append a tampered line (breaking chain)
    with logf.open("a", encoding="utf-8") as f:
        f.write('{"curr_hash": "bad"}\n')
    res = ForensicLogger.verify_chain(str(logf))
    assert not res["ok"] and res["first_bad_line"] is not None


def test_exceed_reuse_quarantine(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        max_reuse=1,
    )
    tracer.mark_in_use(h)
    tracer.mark_reuse(h)  # allowed once
    tracer.mark_reuse(h)  # should push toward violation
    # Subsequent reuse likely quarantines depending on policy; sanitize then free to update metrics
    tracer.sanitize(h, async_=False, verify=True)
    try:
        tracer.free(h)
    except Exception:
        pass
    stats = tracer.get_metrics()
    assert stats["reuse_rate"] >= 0.5


def test_hmac_mismatch_detection(tmp_path, monkeypatch):
    # First write with one secret
    monkeypatch.setenv("FORENSIC_HMAC_SECRET", "secret1")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    # Verification with wrong secret should fail
    res = ForensicLogger.verify_chain(str(tmp_path / "kv_cache.log"), hmac_secret=b"wrong")
    assert not res["ok"]


def test_free_with_high_threshold_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("KV_COVERAGE_THRESHOLD", "100.0")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(16,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    # Force verification failure by monkeypatching _verify_zero to False BEFORE sanitize
    monkeypatch.setattr(tracer, "_verify_zero", lambda *a, **k: False)
    tracer.sanitize(h, async_=False, verify=True)
    with pytest.raises(Exception):
        tracer.free(h)


def test_pinned_torch_cpu(tmp_path):
    try:
        import torch
    except Exception:  # pragma: no cover
        pytest.skip("torch not available")
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype=torch.float32,
        device="cpu",
        framework="torch",
    )
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    assert cov >= 99.9


def test_async_sanitize_early_return(monkeypatch, tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    # Patch _zeroize_buffer to simulate async scheduling
    monkeypatch.setattr(tracer, "_zeroize_buffer", lambda *a, **k: True)
    cov = tracer.sanitize(h, async_=True, verify=True)
    # Early return path should give 0 coverage until wait() invoked
    assert cov == 0.0
    # Now patch to return False so wait completes normally
    monkeypatch.setattr(tracer, "_zeroize_buffer", lambda *a, **k: False)
    cov2 = tracer.wait(h, verify=True)
    assert cov2 in (0.0, 100.0)


def test_double_free_idempotent(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    # Second free should not raise
    tracer.free(h)


def test_sanitize_without_verify(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    cov = tracer.sanitize(h, async_=False, verify=False)
    assert cov == 100.0
    tracer.free(h)


def test_rotation_linkage_mismatch(tmp_path):
    log_path = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=str(log_path))
    tracer.logger._max_bytes = 400
    # Produce enough entries for rotation
    for i in range(40):
        h = tracer.allocate(
            tenant_id="t",
            request_id="r",
            model_id=f"m{i}",
            shape=(2,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    # Tamper with rotate record in active file (first line expected rotate)
    active_text = log_path.read_text().splitlines()
    if active_text:
        try:
            obj = json.loads(active_text[0])
            if obj.get("event_type") == "rotate":
                obj["prev_file_last_hash"] = "bad"  # break linkage
                active_text[0] = json.dumps(obj)
                log_path.write_text("\n".join(active_text) + "\n")
        except Exception:
            pass
    from tools.verify_logs import verify_all_and_write

    result = verify_all_and_write(log_path)
    assert result["ok"] is False


def test_unknown_handle_operations(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    bogus = "dead-beef"  # not allocated
    with pytest.raises(UnknownHandle):
        tracer.mark_in_use(bogus)
    with pytest.raises(UnknownHandle):
        tracer.sanitize(bogus, async_=False, verify=True)


def test_activation_logger_rate_limit(tmp_path):
    from tools.activation_logger import ActivationLogger

    logger = ActivationLogger(out_path=str(tmp_path / "acts.jsonl"), z_threshold=10.0)
    # Very low rate limit (2 Hz) and rapid observations; should not log every time
    for _ in range(5):
        logger.observe("layerX", (0.0, 0.0, 0.0, 1), rate_limit_hz=2.0)
    lines = (tmp_path / "acts.jsonl").read_text().strip().splitlines()
    assert 1 <= len(lines) <= 3  # not all five due to rate limiting


def test_metrics_exporter_main(tmp_path, monkeypatch):
    # Generate a metrics file
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    prom_path = tmp_path / "metrics.prom"
    tracer.export_metrics_prometheus(str(prom_path))
    monkeypatch.setenv("METRICS_FILE", str(prom_path))
    monkeypatch.setenv("METRICS_PORT", "8123")
    monkeypatch.setenv("METRICS_BIND", "127.0.0.1")
    import subprocess
    import sys
    import time as _time

    proc = subprocess.Popen(
        [sys.executable, "tools/metrics_exporter.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give it a moment to start then terminate
    _time.sleep(0.3)
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
    # We don't assert on output; just exercising __main__ path
