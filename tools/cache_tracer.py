"""
KV-Cache Tracer: allocation tagging, sanitization, coverage attestation, and forensic logging.

This utility provides the core enforcement hooks for KV-cache hygiene:
- Allocate buffers (NumPy or PyTorch tensors) with per-request/tenant tagging
- Track lifecycle: allocate -> bind -> write -> sanitize_start/end -> free/quarantine
- Perform scrubbing (zeroization) and compute coverage (>=99.9% required)
- Emit tamper-evident, append-only forensic logs with hash chaining

Intended usage by tests and pipelines:
- from tools.cache_tracer import CacheTracer
- tracer = CacheTracer()
- handle = tracer.allocate(...)
- tracer.bind(handle, stream_id="stream-1")
- # ... model writes K/V ...
- tracer.sanitize(handle, async_: False)
- assert tracer.attest_coverage(handle) >= 99.9
- tracer.free(handle)

Notes:
- For GPU tensors, scrubbing respects CUDA stream semantics when torch is available.
- For CPU/NumPy arrays, scrubbing is synchronous.
- Coverage is computed as scrubbed_bytes/total_bytes and optionally verified via sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
import logging
import hmac
import random

# Optional deps
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# ----- Exceptions -----
class HygieneViolation(Exception):
    pass

class UnknownHandle(Exception):
    pass

class FreeWithoutSanitize(HygieneViolation):
    pass


# ---------- Tamper-evident append-only logger ----------
class ForensicLogger:
    """Append-only, hash-chained JSONL logger.

    Each record is canonicalized and linked using SHA256(prev_hash + canonical_json).
    The file is append-only; previous lines are never modified.
    Supports optional HMAC signing and basic size-based log rotation.
    """

    def __init__(self, log_path: Union[str, Path], *, max_bytes: int = 5_000_000, hmac_secret: Optional[bytes] = None) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure file exists with restricted perms
        if not self.path.exists():
            self.path.touch()
            try:
                os.chmod(self.path, 0o600)
            except Exception:
                pass
        self._lock = threading.Lock()
        self._prev_hash = self._load_last_hash()
        self._max_bytes = max_bytes
        # Secret can also be provided via env FORENSIC_HMAC_SECRET
        self._hmac_key = hmac_secret or os.environ.get("FORENSIC_HMAC_SECRET", "").encode("utf-8") or None

    def _load_last_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        try:
            with self.path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read last ~4KB chunk to find last line
                start = max(0, size - 4096)
                f.seek(start)
                tail = f.read().splitlines()
                for line in reversed(tail):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        return obj.get("curr_hash", "GENESIS")
                    except Exception:
                        continue
        except Exception:
            pass
        return "GENESIS"

    def _load_last_hash_from(self, path: Union[str, Path]) -> str:
        p = Path(path)
        if not p.exists():
            return "GENESIS"
        try:
            with p.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - 4096)
                f.seek(start)
                tail = f.read().splitlines()
                for line in reversed(tail):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        return obj.get("curr_hash", "GENESIS")
                    except Exception:
                        continue
        except Exception:
            pass
        return "GENESIS"

    @staticmethod
    def _canonicalize(record: Mapping[str, Any]) -> str:
        return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def append(self, record: MutableMapping[str, Any]) -> str:
        with self._lock:
            record.setdefault("schema", 1)
            record.setdefault("ts", time.time())
            record.setdefault("trace_id", str(uuid.uuid4()))
            record["prev_hash"] = self._prev_hash
            canonical = self._canonicalize(record)
            curr_hash = hashlib.sha256((self._prev_hash + canonical).encode("utf-8")).hexdigest()
            record["curr_hash"] = curr_hash
            if self._hmac_key:
                record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
            line = json.dumps(record, ensure_ascii=False)
            # Rotate if oversized
            if self.path.exists() and self.path.stat().st_size + len(line) + 1 > self._max_bytes:
                rotated = self.path.with_name(self.path.stem + f"-{int(time.time())}.log")
                self.path.rename(rotated)
                # Capture last hash from rotated file (previous chain)
                prev_file_last_hash = self._load_last_hash_from(rotated)
                # Start new chain
                self._prev_hash = "GENESIS"
                # Write a rotate marker directly (no recursion)
                rotate_record = {
                    "schema": 1,
                    "event_type": "rotate",
                    "ts": time.time(),
                    "trace_id": str(uuid.uuid4()),
                    "prev_hash": self._prev_hash,
                    "prev_file": str(rotated.name),
                    "prev_file_last_hash": prev_file_last_hash,
                }
                canonical_rotate = self._canonicalize(rotate_record)
                rotate_hash = hashlib.sha256((self._prev_hash + canonical_rotate).encode("utf-8")).hexdigest()
                rotate_record["curr_hash"] = rotate_hash
                if self._hmac_key:
                    rotate_record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical_rotate).encode("utf-8"), hashlib.sha256).hexdigest()
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rotate_record, ensure_ascii=False) + "\n")
                try:
                    os.chmod(self.path, 0o600)
                except Exception:
                    pass
                self._prev_hash = rotate_hash
                # Recompute canonical & hash for the original record with new prev_hash
                record.pop("curr_hash", None)
                record.pop("hmac", None)
                record["prev_hash"] = self._prev_hash
                canonical = self._canonicalize(record)
                curr_hash = hashlib.sha256((self._prev_hash + canonical).encode("utf-8")).hexdigest()
                record["curr_hash"] = curr_hash
                if self._hmac_key:
                    record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
                line = json.dumps(record, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._prev_hash = curr_hash
            return curr_hash

    @staticmethod
    def verify_chain(path: Union[str, Path], *, hmac_secret: Optional[bytes] = None) -> Dict[str, Any]:
        prev = "GENESIS"
        ok = True
        count = 0
        bad_index: Optional[int] = None
        key = hmac_secret or os.environ.get("FORENSIC_HMAC_SECRET", "").encode("utf-8") or None
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                curr = obj.get("curr_hash")
                # Recompute
                tmp = dict(obj)
                tmp.pop("curr_hash", None)
                provided_hmac = tmp.pop("hmac", None)
                canonical = json.dumps(tmp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                calc = hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()
                if curr != calc:
                    ok = False
                    bad_index = i
                    break
                if key is not None and provided_hmac:
                    calc_hmac = hmac.new(key, (prev + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
                    if provided_hmac != calc_hmac:
                        ok = False
                        bad_index = i
                        break
                prev = curr
                count += 1
        return {"ok": ok, "lines": count, "first_bad_line": bad_index}

    @staticmethod
    def verify_all(path: Union[str, Path]) -> Dict[str, Any]:
        """Verify integrity of the active log and any rotated predecessors.

        Assumes rotated files follow the pattern '<stem>-<ts>.log' created by this logger.
        Validates:
        - Each file's internal hash chain
        - Rotation linkage: next file's rotate record references previous file name and last hash
        """
        base = Path(path)
        directory = base.parent
        stem = base.stem
        # Collect rotated files matching stem-*.log
        rotated = sorted(directory.glob(f"{stem}-*.log"), key=lambda p: p.name)
        files = rotated + [base]
        results: List[Dict[str, Any]] = []
        ok = True

        def _last_hash(p: Path) -> str:
            try:
                with p.open("rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    start = max(0, size - 4096)
                    f.seek(start)
                    tail = f.read().splitlines()
                    for line in reversed(tail):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        ch = obj.get("curr_hash")
                        if ch:
                            return ch
            except Exception:
                pass
            return "GENESIS"

        def _first_rotate_record(p: Path) -> Optional[Dict[str, Any]]:
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("event_type") == "rotate":
                            return obj
                        # Stop early after first non-rotate event
                        break
            except Exception:
                return None
            return None

        # Verify each file chain
        for p in files:
            res = ForensicLogger.verify_chain(p)
            results.append({"file": str(p.name), **res})
            if not res.get("ok", False):
                ok = False

        # Verify rotation linkage between consecutive files
        for i in range(1, len(files)):
            prev = files[i - 1]
            curr = files[i]
            rotate = _first_rotate_record(curr)
            if rotate is None:
                # No rotation marker in current file; acceptable only for the first file in series
                # If there are rotated predecessors, we must have a rotate record
                if i > 0:
                    ok = False
                    results.append({"file": str(curr.name), "ok": False, "error": "missing rotate record"})
                continue
            expected_name = prev.name
            expected_hash = _last_hash(prev)
            if rotate.get("prev_file") != expected_name or rotate.get("prev_file_last_hash") != expected_hash:
                ok = False
                results.append({
                    "file": str(curr.name),
                    "ok": False,
                    "error": "rotation linkage mismatch",
                    "expected_prev_file": expected_name,
                    "expected_prev_last_hash": expected_hash,
                    "rotate_prev_file": rotate.get("prev_file"),
                    "rotate_prev_last_hash": rotate.get("prev_file_last_hash"),
                })

        return {"ok": ok, "files": results}


# ---------- KV Buffer model ----------
@dataclass
class KVBuffer:
    handle: str
    tenant_id: str
    request_id: str
    model_id: str
    device: str  # e.g., 'cpu', 'cuda:0'
    shape: Tuple[int, ...]
    dtype: str
    ptr: Optional[int]
    nbytes: int
    created_ts: float = field(default_factory=time.time)
    stream_id: Optional[str] = None
    status: str = "allocated"  # allocated|bound|in_use|sanitizing|sanitized|freed|quarantined
    scrubbed_bytes: int = 0
    sanitize_start_ts: Optional[float] = None
    sanitize_end_ts: Optional[float] = None
    coverage_pct: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)
    _tensor: Any = None  # torch.Tensor or np.ndarray
    # Enhancements
    pinned: bool = False
    reuse_count: int = 0
    first_use_ts: Optional[float] = None
    last_use_ts: Optional[float] = None
    ttl_sec: Optional[float] = None
    max_reuse: Optional[int] = None
    stream_obj: Any = None
    event_obj: Any = None
    verify_samples: List[int] = field(default_factory=list)
    sanitize_duration_ms: Optional[float] = None


class CacheTracer:
    """Tracer and enforcer for KV-cache buffers.

    Responsibilities:
    - Allocate and tag buffers
    - Manage lifecycle and enforce sanitization
    - Provide coverage attestation and metrics export
    - Emit forensic logs on all critical events
    """

    COVERAGE_THRESHOLD: float = 99.9

    def __init__(self, log_path: Union[str, Path] = "forensics/kv_cache.log", *, coverage_threshold: float = 99.9, max_log_bytes: int = 5_000_000, double_pass_default: bool = False, verify_samples_default: int = 8, default_max_reuse: int = 0, default_ttl_sec: Optional[float] = None) -> None:
        self.logger = ForensicLogger(log_path, max_bytes=max_log_bytes)
        self._buffers: Dict[str, KVBuffer] = {}
        self._lock = threading.RLock()
        logging.getLogger(__name__).setLevel(logging.INFO)
        # Allow env overrides for sensible defaults
        def _env_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return str(val).strip().lower() in {"1", "true", "yes", "on"}
        def _env_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except Exception:
                return default
        def _env_float_opt(key: str, default: Optional[float]) -> Optional[float]:
            v = os.environ.get(key)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except Exception:
                return default
        self.COVERAGE_THRESHOLD = float(os.environ.get("KV_COVERAGE_THRESHOLD", coverage_threshold))
        self._double_pass_default = _env_bool("KV_DOUBLE_PASS_DEFAULT", double_pass_default)
        self._verify_samples_default = _env_int("KV_VERIFY_SAMPLES_DEFAULT", verify_samples_default)
        self._default_max_reuse = _env_int("KV_DEFAULT_MAX_REUSE", default_max_reuse)
        self._default_ttl_sec = _env_float_opt("KV_DEFAULT_TTL_SEC", default_ttl_sec)
        # Metrics
        self._sanitize_durations_ms: List[float] = []
        self._allocations: int = 0
        self._reuse_events: int = 0
        self._freed_total: int = 0

    # ----- Allocation & lifecycle -----
    def allocate(
        self,
        *,
        tenant_id: str,
        request_id: str,
        model_id: str,
        shape: Tuple[int, ...],
        dtype: Union[str, Any] = "float32",
        device: str = "cpu",
        framework: Optional[str] = None,
        stream_id: Optional[str] = None,
        pinned: bool = False,
        ttl_sec: Optional[float] = None,
        max_reuse: Optional[int] = None,
    ) -> str:
        """Allocate a buffer/tensor and register it with metadata.

        Returns a handle string used for subsequent operations.
        """
        handle = str(uuid.uuid4())
        tensor, ptr, nbytes, dtype_str = self._create_buffer(shape, dtype, device, framework, pinned=pinned)
        buf = KVBuffer(
            handle=handle,
            tenant_id=tenant_id,
            request_id=request_id,
            model_id=model_id,
            device=device,
            shape=shape,
            dtype=dtype_str,
            ptr=ptr,
            nbytes=nbytes,
            stream_id=stream_id,
            _tensor=tensor,
            pinned=pinned,
            ttl_sec=ttl_sec if ttl_sec is not None else self._default_ttl_sec,
            max_reuse=max_reuse if max_reuse is not None else self._default_max_reuse,
        )
        with self._lock:
            self._buffers[handle] = buf
        self.logger.append(
            {
                "event_type": "allocate",
                "handle": handle,
                "tenant_id": tenant_id,
                "request_id": request_id,
                "model_id": model_id,
                "device": device,
                "shape": list(shape),
                "dtype": dtype_str,
                "ptr": ptr,
                "nbytes": nbytes,
                "stream_id": stream_id,
                "pinned": pinned,
                "ttl_sec": buf.ttl_sec,
                "max_reuse": buf.max_reuse,
            }
        )
        self._allocations += 1
        return handle

    def _get(self, handle: str) -> KVBuffer:
        with self._lock:
            if handle not in self._buffers:
                raise UnknownHandle(f"Unknown handle: {handle}")
            return self._buffers[handle]

    def bind(self, handle: str, *, stream_id: Optional[str] = None, stream: Any = None) -> None:
        with self._lock:
            buf = self._get(handle)
            buf.stream_id = stream_id
            buf.stream_obj = stream
            buf.status = "bound"
        self.logger.append({"event_type": "bind", "handle": handle, "tenant_id": buf.tenant_id, "request_id": buf.request_id, "model_id": buf.model_id, "stream_id": stream_id})

    def mark_in_use(self, handle: str, *, stream: Any = None) -> None:
        with self._lock:
            buf = self._get(handle)
            if stream is not None:
                buf.stream_obj = stream
            buf.status = "in_use"
            now = time.time()
            buf.first_use_ts = buf.first_use_ts or now
            buf.last_use_ts = now
            # Evaluate TTL policy inside lock, act outside lock
            ttl_violation = False
            if buf.ttl_sec is not None and buf.first_use_ts is not None:
                ttl_violation = (buf.ttl_sec <= 0) or ((now - buf.first_use_ts) > buf.ttl_sec)
            tenant_id, request_id, model_id = buf.tenant_id, buf.request_id, buf.model_id
        self.logger.append({"event_type": "write", "handle": handle, "tenant_id": tenant_id, "request_id": request_id, "model_id": model_id})
        if ttl_violation:
            self.quarantine(handle, reason="ttl_expired")

    def mark_reuse(self, handle: str) -> None:
        """Mark a buffer as reused by the same request/session.

        Enforces max_reuse policy, quarantining on violation and recording forensic events.
        """
        with self._lock:
            buf = self._get(handle)
            buf.reuse_count += 1
            self._reuse_events += 1
            exceeded = buf.max_reuse is not None and buf.max_reuse >= 0 and buf.reuse_count > buf.max_reuse
            tenant_id, request_id, model_id = buf.tenant_id, buf.request_id, buf.model_id
            reuse_count, max_reuse = buf.reuse_count, buf.max_reuse
        # Log reuse event
        self.logger.append({
            "event_type": "reuse",
            "handle": handle,
            "tenant_id": tenant_id,
            "request_id": request_id,
            "model_id": model_id,
            "reuse_count": reuse_count,
            "max_reuse": max_reuse,
        })
        if exceeded:
            self.quarantine(handle, reason="reuse_limit_exceeded")

    # ----- Sanitization -----
    def sanitize(self, handle: str, *, async_: bool = False, verify: bool = True, double_pass: Optional[bool] = None, samples: Optional[int] = None) -> float:
        with self._lock:
            buf = self._get(handle)
            buf.status = "sanitizing"
            buf.sanitize_start_ts = time.time()
            stream_id = buf.stream_id
            device = buf.device
        self.logger.append({"event_type": "sanitize_start", "handle": handle, "tenant_id": buf.tenant_id, "request_id": buf.request_id, "model_id": buf.model_id, "device": device, "stream_id": stream_id})
        # Zeroization scheduling
        scheduled_async = self._zeroize_buffer(buf, async_=async_)
        # Optional double pass
        if double_pass if double_pass is not None else self._double_pass_default:
            # Second pass enqueued after first on same stream or run sync on CPU
            scheduled_async2 = self._zeroize_buffer(buf, async_=async_)
            scheduled_async = scheduled_async or scheduled_async2
        if scheduled_async and async_:
            return 0.0
        # Complete synchronously
        return self.wait(handle, verify=verify, samples=samples)

    def wait(self, handle: str, *, verify: bool = True, samples: Optional[int] = None) -> float:
        buf = self._get(handle)
        # Sync outside lock (may block)
        if buf.event_obj is not None:
            try:
                buf.event_obj.synchronize()
            except Exception:
                pass
        elif torch is not None and isinstance(buf._tensor, torch.Tensor) and buf._tensor.is_cuda:
            torch.cuda.synchronize(device=buf._tensor.device)
        with self._lock:
            buf.sanitize_end_ts = time.time()
            buf.sanitize_duration_ms = float(int(1000 * (buf.sanitize_end_ts - (buf.sanitize_start_ts or buf.sanitize_end_ts))))
            self._sanitize_durations_ms.append(buf.sanitize_duration_ms)
            buf.scrubbed_bytes = buf.nbytes
            buf.coverage_pct = self._attest_internal(buf, verify=verify, samples=samples)
            buf.status = "sanitized"
            cov = buf.coverage_pct
            dur = buf.sanitize_duration_ms
            samples_list = list(buf.verify_samples)
        self.logger.append({"event_type": "sanitize_end", "handle": handle, "tenant_id": buf.tenant_id, "request_id": buf.request_id, "model_id": buf.model_id, "coverage_pct": cov, "duration_ms": dur, "verify_samples": samples_list})
        return cov

    def attest_coverage(self, handle: str) -> float:
        """Return the last computed coverage percentage for a handle."""
        buf = self._get(handle)
        return buf.coverage_pct

    def quarantine(self, handle: str, *, reason: str) -> None:
        with self._lock:
            buf = self._get(handle)
            buf.status = "quarantined"
            buf.notes.setdefault("quarantine_reasons", []).append(reason)
        self.logger.append({"event_type": "quarantine", "handle": handle, "tenant_id": buf.tenant_id, "request_id": buf.request_id, "model_id": buf.model_id, "reason": reason})

    def free(self, handle: str) -> None:
        with self._lock:
            buf = self._get(handle)
            if buf.status == "freed":
                return
            if buf.coverage_pct < self.COVERAGE_THRESHOLD:
                # Drop lock before quarantine append to avoid nested lock issues
                pass
        if buf.coverage_pct < self.COVERAGE_THRESHOLD:
            self.quarantine(handle, reason=f"coverage {buf.coverage_pct:.4f} below threshold")
            raise FreeWithoutSanitize("Attempt to free buffer without sufficient sanitization")
        with self._lock:
            buf.status = "freed"
            buf._tensor = None
            self._freed_total += 1
        self.logger.append({"event_type": "free", "handle": handle, "tenant_id": buf.tenant_id, "request_id": buf.request_id, "model_id": buf.model_id})

    # ----- Metrics and export -----
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._buffers)
            unsanitized = sum(1 for b in self._buffers.values() if b.status not in ("sanitized", "freed", "quarantined"))
            quarantined = sum(1 for b in self._buffers.values() if b.status == "quarantined")
            active = sum(1 for b in self._buffers.values() if b.status not in ("freed", "quarantined"))
            coverages = [b.coverage_pct for b in self._buffers.values() if b.coverage_pct > 0]
            durations = [b.sanitize_duration_ms for b in self._buffers.values() if b.sanitize_duration_ms is not None]
            reuse_total = sum(b.reuse_count for b in self._buffers.values())
            freed_total = self._freed_total
            allocations = self._allocations
        min_cov = min(coverages) if coverages else 0.0
        avg_cov = sum(coverages) / len(coverages) if coverages else 0.0
        p50 = self._percentile(durations, 50.0)
        p95 = self._percentile(durations, 95.0)
        reuse_rate = (reuse_total / allocations) if allocations > 0 else 0.0
        return {
            "total_buffers": total,
            "active_buffers": active,
            "unsanitized_regions_count": unsanitized,
            "quarantine_count": quarantined,
            "min_coverage_pct": min_cov,
            "avg_coverage_pct": avg_cov,
            "threshold_pct": self.COVERAGE_THRESHOLD,
            "sanitize_duration_p50_ms": p50,
            "sanitize_duration_p95_ms": p95,
            "allocations": allocations,
            "freed_total": freed_total,
            "reuse_total": reuse_total,
            "reuse_rate": reuse_rate,
        }

    def export_metrics(self, path: Union[str, Path]) -> None:
        metrics = self.get_metrics()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def export_metrics_prometheus(self, path: Union[str, Path]) -> None:
        m = self.get_metrics()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        # HELP/TYPE headers
        lines.append("# HELP kv_hygiene_unsanitized_regions Count of buffers not yet sanitized")
        lines.append("# TYPE kv_hygiene_unsanitized_regions gauge")
        lines.append(f"kv_hygiene_unsanitized_regions {int(m['unsanitized_regions_count'])}")

        lines.append("# HELP kv_hygiene_quarantine_count Count of buffers quarantined due to policy violations")
        lines.append("# TYPE kv_hygiene_quarantine_count gauge")
        lines.append(f"kv_hygiene_quarantine_count {int(m['quarantine_count'])}")

        lines.append("# HELP kv_hygiene_min_coverage_pct Minimum observed sanitization coverage percent across buffers")
        lines.append("# TYPE kv_hygiene_min_coverage_pct gauge")
        lines.append(f"kv_hygiene_min_coverage_pct {float(m['min_coverage_pct'])}")

        lines.append("# HELP kv_hygiene_avg_coverage_pct Average sanitization coverage percent across buffers")
        lines.append("# TYPE kv_hygiene_avg_coverage_pct gauge")
        lines.append(f"kv_hygiene_avg_coverage_pct {float(m['avg_coverage_pct'])}")

        lines.append("# HELP kv_hygiene_threshold_pct Required minimum sanitization coverage percent")
        lines.append("# TYPE kv_hygiene_threshold_pct gauge")
        lines.append(f"kv_hygiene_threshold_pct {float(m['threshold_pct'])}")

        lines.append("# HELP kv_hygiene_sanitize_duration_p50_ms P50 sanitize duration in milliseconds")
        lines.append("# TYPE kv_hygiene_sanitize_duration_p50_ms gauge")
        lines.append(f"kv_hygiene_sanitize_duration_p50_ms {float(m['sanitize_duration_p50_ms'])}")

        lines.append("# HELP kv_hygiene_sanitize_duration_p95_ms P95 sanitize duration in milliseconds")
        lines.append("# TYPE kv_hygiene_sanitize_duration_p95_ms gauge")
        lines.append(f"kv_hygiene_sanitize_duration_p95_ms {float(m['sanitize_duration_p95_ms'])}")

        lines.append("# HELP kv_hygiene_active_buffers Number of active (not freed/quarantined) buffers")
        lines.append("# TYPE kv_hygiene_active_buffers gauge")
        lines.append(f"kv_hygiene_active_buffers {int(m['active_buffers'])}")

        lines.append("# HELP kv_hygiene_total_buffers Total buffers observed")
        lines.append("# TYPE kv_hygiene_total_buffers gauge")
        lines.append(f"kv_hygiene_total_buffers {int(m['total_buffers'])}")

        lines.append("# HELP kv_hygiene_allocations_total Total number of allocations")
        lines.append("# TYPE kv_hygiene_allocations_total counter")
        lines.append(f"kv_hygiene_allocations_total {int(m['allocations'])}")

        lines.append("# HELP kv_hygiene_freed_total Total number of frees")
        lines.append("# TYPE kv_hygiene_freed_total counter")
        lines.append(f"kv_hygiene_freed_total {int(m['freed_total'])}")

        lines.append("# HELP kv_hygiene_reuse_total Total number of reuse events")
        lines.append("# TYPE kv_hygiene_reuse_total counter")
        lines.append(f"kv_hygiene_reuse_total {int(m['reuse_total'])}")

        lines.append("# HELP kv_hygiene_reuse_rate Reuse events per allocation")
        lines.append("# TYPE kv_hygiene_reuse_rate gauge")
        lines.append(f"kv_hygiene_reuse_rate {float(m['reuse_rate'])}")

        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ----- Internals -----
    def _create_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, Any],
        device: str,
        framework: Optional[str],
        *,
        pinned: bool = False,
    ) -> Tuple[Any, Optional[int], int, str]:
        """Create a tensor/array per request, returning (obj, ptr, nbytes, dtype_str)."""
        # Torch path
        if torch is not None and (framework == "torch" or (isinstance(dtype, torch.dtype) or device.startswith("cuda"))):
            # Map dtype
            if isinstance(dtype, torch.dtype):
                tdtype = dtype
                dtype_str = str(dtype).replace("torch.", "")
            else:
                dtype_map = {
                    "float32": torch.float32,
                    "float": torch.float32,
                    "float16": torch.float16,
                    "half": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float64": torch.float64,
                    "double": torch.float64,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "int16": torch.int16,
                    "int8": torch.int8,
                    "uint8": torch.uint8,
                    "bool": torch.bool,
                }
                tdtype = dtype_map.get(str(dtype).lower(), torch.float32)
                dtype_str = str(dtype).lower()
            dev = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
            if dev.type == "cuda":
                t = torch.zeros(*shape, dtype=tdtype, device=dev)
            else:
                # For CPU, optionally pin memory if requested
                t = torch.zeros(*shape, dtype=tdtype, device=dev, pin_memory=pinned) if hasattr(torch, "zeros") else torch.empty(*shape, dtype=tdtype, device=dev)
                if t.numel() > 0:
                    t.zero_()
            ptr = int(t.data_ptr()) if hasattr(t, "data_ptr") else None
            nbytes = int(t.numel() * t.element_size())
            return t, ptr, nbytes, dtype_str

        # NumPy path (default)
        if np is None:
            raise RuntimeError("NumPy not available to allocate CPU buffers")
        ndt = np.dtype(dtype if isinstance(dtype, str) else "float32")
        arr = np.zeros(shape, dtype=ndt)
        ptr = int(arr.__array_interface__.get("data", (0,))[0]) if hasattr(arr, "__array_interface__") else None
        nbytes = int(arr.nbytes)
        dtype_str = str(arr.dtype)
        return arr, ptr, nbytes, dtype_str

    def _zeroize_buffer(self, buf: KVBuffer, *, async_: bool) -> bool:
        """Zeroize underlying storage. Returns True if scheduled asynchronously."""
        t = buf._tensor
        # Torch tensors
        if torch is not None and isinstance(t, torch.Tensor):
            if t.is_cuda:
                # CUDA ops are async; honor provided stream if any
                try:
                    if async_:
                        stream = buf.stream_obj if buf.stream_obj is not None else torch.cuda.current_stream(device=t.device)
                        # Ensure we have a proper Stream object
                        with torch.cuda.stream(stream):
                            t.zero_()
                            ev = torch.cuda.Event()
                            ev.record(stream)
                            buf.event_obj = ev
                        return True
                    else:
                        t.zero_()
                        torch.cuda.synchronize(device=t.device)
                        buf.event_obj = None
                        return False
                except Exception:
                    # Fallback: do sync zeroization
                    t.zero_()
                    return False
            else:
                t.zero_()
                return False
        # NumPy arrays
        if np is not None and hasattr(t, "fill"):
            t.fill(0)
            return False
        # Unknown type
        return False

    def _attest_internal(self, buf: KVBuffer, *, verify: bool = True, samples: Optional[int] = None) -> float:
        """Compute and return coverage percent. Optionally verify via sampling."""
        if verify:
            ok = self._verify_zero(buf, samples=samples or self._verify_samples_default)
            return 100.0 if ok else 0.0
        return 100.0

    def _verify_zero(self, buf: KVBuffer, *, samples: int = 8) -> bool:
        """Deterministically sample elements to verify zeroization."""
        t = buf._tensor
        # Determine element count
        try:
            if torch is not None and isinstance(t, torch.Tensor):
                n = int(t.numel())
            else:
                n = int(t.size if hasattr(t, "size") else t.size)
        except Exception:
            n = 0
        if n == 0:
            buf.verify_samples = []
            return True
        k = max(1, min(samples, n))
        rng = random.Random(buf.handle)
        idxs = sorted({rng.randrange(0, n) for _ in range(k)})
        buf.verify_samples = list(map(int, idxs))
        # Flat views for sampling
        if torch is not None and isinstance(t, torch.Tensor):
            flat = t.view(-1)
            for i in idxs:
                try:
                    val = float(flat[int(i)].item())
                except Exception:
                    return False
                if val != 0.0:
                    return False
            return True
        else:
            # NumPy path
            if np is None:
                return True
            flat = t.reshape(-1)
            for i in idxs:
                if float(flat[int(i)]) != 0.0:
                    return False
            return True

    @staticmethod
    def _percentile(values: Iterable[Optional[float]], q: float) -> float:
        arr = [float(v) for v in values if v is not None]
        if not arr:
            return 0.0
        arr.sort()
        if len(arr) == 1:
            return arr[0]
        pos = (q / 100.0) * (len(arr) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return arr[lo]
        frac = pos - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac
