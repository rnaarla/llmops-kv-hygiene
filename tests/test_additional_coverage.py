import json
import os
import time
import subprocess
import sys
from pathlib import Path
from http.client import HTTPConnection

import pytest

from tools.cache_tracer import CacheTracer, ForensicLogger
from tools.eviction_checker import main as eviction_main
from tools.verify_logs import prune_rotated, verify_all_and_write
from tools.activation_logger import ActivationLogger


def test_verify_logs_retention_archive(tmp_path, monkeypatch):
    log_path = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=str(log_path))
    tracer.logger._max_bytes = 500
    # Create several rotations
    for i in range(40):
        h = tracer.allocate(tenant_id="t", request_id="r", model_id=f"m{i}", shape=(4,), dtype="float32", device="cpu", framework="numpy")
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
        if i == 5:
            # Simulate age for early rotations by manipulating mtime
            for p in tmp_path.glob("kv_cache-*.log"):
                os.utime(p, (time.time() - 3 * 86400, time.time() - 3 * 86400))
    # Rename a subset of rotated files to very old timestamps so age pruning triggers
    rotated = sorted(tmp_path.glob("kv_cache-*.log"))
    old_base = int(time.time() - 4 * 86400)
    for idx, p in enumerate(rotated[:3]):
        new_name = p.with_name(f"kv_cache-{old_base - idx}.log")
        p.rename(new_name)
    # Create one malformed name to exercise fallback-to-mtime path
    malformed_source = rotated[3] if len(rotated) > 3 else None
    if malformed_source and malformed_source.exists():
        mal_dest = malformed_source.with_name("kv_cache-foo.log")
        malformed_source.rename(mal_dest)
        # ensure its mtime is old enough
        os.utime(mal_dest, (time.time() - 5 * 86400, time.time() - 5 * 86400))
    archive_dir = tmp_path / "arch"
    # Age+count pruning with archive (should remove old & malformed)
    removed = prune_rotated(log_path, retention_days=1, max_rotated=1, archive_dir=archive_dir)
    assert removed, "Expected some rotated files to be archived or removed"
    assert archive_dir.exists()
    res_after = verify_all_and_write(log_path)
    assert "files" in res_after


def test_eviction_checker_all_failures(tmp_path):
    metrics = {
        "min_coverage_pct": 10.0,
        "unsanitized_regions_count": 5,
        "quarantine_count": 3,
        "reuse_rate": 2.5,
        "sanitize_duration_p95_ms": 9999.0,
        "avg_coverage_pct": 20.0,
        "threshold_pct": 99.9,
        "sanitize_duration_p50_ms": 10.0,
        "active_buffers": 3,
        "total_buffers": 3,
        "allocations": 2,
        "freed_total": 0,
        "reuse_total": 5,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    rc = eviction_main([
        str(metrics_path),
        "--coverage-min", "95.0",
        "--unsanitized-max", "0",
        "--quarantine-max", "0",
        "--reuse-rate-max", "0.1",
        "--sanitize-p95-ms-max", "100.0",
        "--out", str(tmp_path / "verdict.json"),
    ])
    assert rc == 2
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert len(verdict["failures"]) >= 5


def test_metrics_exporter_content_length_and_empty(tmp_path, monkeypatch):
    # Prepare metrics
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(tenant_id="t", request_id="r", model_id="m", shape=(4,), dtype="float32", device="cpu", framework="numpy")
    tracer.mark_in_use(h)
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)
    prom = tmp_path / "metrics.prom"
    tracer.export_metrics_prometheus(str(prom))
    monkeypatch.setenv("METRICS_FILE", str(prom))
    port = 8255
    monkeypatch.setenv("METRICS_PORT", str(port))
    monkeypatch.setenv("METRICS_BIND", "127.0.0.1")
    proc = subprocess.Popen([sys.executable, "tools/metrics_exporter.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(0.25)
        conn = HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        data = resp.read()
        assert resp.status == 200
        assert resp.getheader("Content-Length") == str(len(data))
        # Delete file and request again (empty path)
        prom.unlink(missing_ok=True)  # type: ignore[arg-type]
        time.sleep(0.05)
        conn2 = HTTPConnection("127.0.0.1", port)
        conn2.request("GET", "/metrics")
        resp2 = conn2.getresponse()
        data2 = resp2.read()
        assert resp2.status == 200
        assert resp2.getheader("Content-Length") == str(len(data2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


def test_activation_logger_anomaly_variants(tmp_path):
    logger = ActivationLogger(out_path=str(tmp_path / "acts.jsonl"), z_threshold=1.0)
    # Establish baseline
    for _ in range(10):
        logger.observe("L1", (0.0, 0.0, 0.0, 4), rate_limit_hz=None)  # no rate limit so all logged
    # z-score anomaly: sudden mean shift
    flagged1 = logger.observe("L1", (50.0, 0.0, 50.0, 4), rate_limit_hz=None)
    # max_val anomaly with zero std baseline (new layer)
    flagged2 = logger.observe("L2", (0.0, 0.0, 500.0, 4), rate_limit_hz=None)
    # rate limited layer: only first logs
    logger.observe("L3", (0.0, 0.0, 0.0, 1), rate_limit_hz=5.0)
    logger.observe("L3", (100.0, 0.0, 100.0, 1), rate_limit_hz=5.0)
    lines = (tmp_path / "acts.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(l) for l in lines]
    assert any(p["anomalous"] for p in parsed if p["layer"] == "L1")
    assert any(p["anomalous"] for p in parsed if p["layer"] == "L2")
    # Second L3 anomaly shouldn't have logged due to rate limiting
    l3_count = sum(1 for p in parsed if p["layer"] == "L3")
    assert l3_count == 1


def test_forensic_rotation_mismatch_branch(tmp_path):
    log_path = tmp_path / "kv_cache.log"
    tracer = CacheTracer(log_path=str(log_path))
    tracer.logger._max_bytes = 400
    for i in range(35):
        h = tracer.allocate(tenant_id="t", request_id="r", model_id=f"m{i}", shape=(3,), dtype="float32", device="cpu", framework="numpy")
        tracer.mark_in_use(h)
        tracer.sanitize(h, async_=False, verify=True)
        tracer.free(h)
    # Corrupt rotate record in active file to force mismatch
    active_lines = log_path.read_text().splitlines()
    if active_lines:
        try:
            obj = json.loads(active_lines[0])
            if obj.get("event_type") == "rotate":
                obj["prev_file"] = "nonexistent.log"
                active_lines[0] = json.dumps(obj)
                log_path.write_text("\n".join(active_lines) + "\n")
        except Exception:
            pass
    res = ForensicLogger.verify_all(str(log_path))
    # Expect an error entry for rotation linkage mismatch
    files = res.get("files", [])
    assert any(f.get("error") == "rotation linkage mismatch" for f in files)


def test_multiple_buffer_metrics_edge(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    handles = []
    for i in range(5):
        h = tracer.allocate(tenant_id="t", request_id="r", model_id=f"m{i}", shape=(4,), dtype="float32", device="cpu", framework="numpy")
        tracer.mark_in_use(h)
        if i % 2 == 0:
            tracer.sanitize(h, async_=False, verify=True)
            tracer.free(h)
        handles.append(h)
    metrics = tracer.get_metrics()
    assert metrics["unsanitized_regions_count"] >= 1
    assert metrics["min_coverage_pct"] >= 0.0
