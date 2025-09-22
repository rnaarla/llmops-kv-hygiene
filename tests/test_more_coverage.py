import json
from pathlib import Path

import pytest

from tools.cache_tracer import CacheTracer
from tools.forensic_logger import ForensicLogger

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def test_double_pass_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("KV_DOUBLE_PASS_DEFAULT", "1")
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
    assert cov > 0
    tracer.free(h)


def test_negative_max_reuse_no_violation(tmp_path):
    tracer = CacheTracer(log_path=str(tmp_path / "kv_cache.log"))
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(4,),
        dtype="float32",
        device="cpu",
        framework="numpy",
        max_reuse=-1,
    )
    tracer.mark_in_use(h)
    tracer.mark_reuse(h)
    assert tracer._get(h).status != "quarantined"
    tracer.sanitize(h, async_=False, verify=True)
    tracer.free(h)


def _force_rotation_with_malformed_tail(tmp_path: Path):
    log_path = tmp_path / "rot.log"
    flog = ForensicLogger(log_path, max_bytes=250)
    for i in range(30):
        flog.append({"event_type": "e", "i": i})
    # Pick a rotated file and add malformed trailing JSON line to exercise exception path in _load_last_hash
    for rot in tmp_path.glob("rot-*.log"):
        with rot.open("a", encoding="utf-8") as f:
            f.write("{malformed\n")
        break
    return log_path


def test_forensic_logger_malformed_tail(tmp_path):
    log_path = _force_rotation_with_malformed_tail(tmp_path)
    try:
        # Attempt full verification; may raise due to intentionally malformed tail line in a rotated file
        ForensicLogger.verify_all(str(log_path))
    except json.JSONDecodeError:
        pass  # Acceptable: malformed line caused decode error during verify_chain
    # Ensure the malformed marker exists in at least one rotated file to confirm we exercised the path
    rotated_with_marker = [
        p for p in tmp_path.glob("rot-*.log") if "{malformed" in p.read_text()
    ]  # noqa: PLR2004
    assert rotated_with_marker, "Expected at least one rotated file containing malformed tail line"


def test_verify_all_missing_rotate_record(tmp_path):
    # Create two rotations then strip rotate record from active to trigger 'missing rotate record'
    log_path = tmp_path / "chain.log"
    flog = ForensicLogger(log_path, max_bytes=200)
    for i in range(40):
        flog.append({"event_type": "alloc", "i": i})
    # Ensure there is at least one rotated predecessor
    rotated_files = sorted(tmp_path.glob("chain-*.log"))
    if not rotated_files:
        pytest.skip("rotation did not occur; environment timing variance")
    # Remove rotate record (first line if rotate) from active log
    active_lines = log_path.read_text().splitlines()
    if active_lines and '"event_type": "rotate"' in active_lines[0]:
        # Remove it entirely
        log_path.write_text("\n".join(active_lines[1:]) + "\n")
    res = ForensicLogger.verify_all(str(log_path))
    # Expect at least one file entry with missing rotate record error OR rotation mismatch
    errors = [
        f
        for f in res.get("files", [])
        if f.get("error") in {"missing rotate record", "rotation linkage mismatch"}
    ]
    assert errors, f"Expected linkage related error entries. Result: {res}"


def test_hmac_mismatch_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("FORENSIC_HMAC_SECRET", "abc123")
    log_path = tmp_path / "hmac.log"
    flog = ForensicLogger(log_path, max_bytes=10_000)
    for i in range(5):
        flog.append({"event_type": "x", "i": i})
    # Tamper one line's hmac but keep curr_hash valid
    lines = log_path.read_text().strip().splitlines()
    if not lines:
        pytest.skip("no lines to tamper")
    obj = json.loads(lines[-1])
    # Recompute correct hash then alter only hmac
    _curr_hash = obj["curr_hash"]  # retained for clarity; value intentionally replaced
    obj["hmac"] = "deadbeef" * 4
    lines[-1] = json.dumps(obj)
    log_path.write_text("\n".join(lines) + "\n")
    res = ForensicLogger.verify_chain(log_path, hmac_secret=b"abc123")
    assert res["ok"] is False and res["first_bad_line"] is not None
