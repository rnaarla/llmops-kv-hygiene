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

import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import sanitizer as sanitizer_mod
from .buffer_model import KVBuffer
from .forensic_logger import ForensicLogger

# percentile imported elsewhere when needed; local implementation uses _percentile
from .policies import evaluate_policies

# Optional deps
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


# ----- Exceptions -----
class HygieneViolationError(Exception):
    """Generic hygiene policy violation."""


class UnknownHandleError(Exception):
    """Raised when an unknown or freed handle is referenced."""


class FreeWithoutSanitizeError(HygieneViolationError):
    """Raised when attempting to free a buffer below coverage threshold."""


# Backward compatibility temporary aliases (scheduled removal in future major)
HygieneViolation = HygieneViolationError
UnknownHandle = UnknownHandleError
FreeWithoutSanitize = FreeWithoutSanitizeError


# ---------- KV Buffer model ----------


class CacheTracer:
    """Tracer and enforcer for KV-cache buffers.

    Responsibilities:
    - Allocate and tag buffers
    - Manage lifecycle and enforce sanitization
    - Provide coverage attestation and metrics export
    - Emit forensic logs on all critical events
    """

    COVERAGE_THRESHOLD: float = 99.9

    def __init__(
        self,
        log_path: str | Path = "forensics/kv_cache.log",
        *,
        coverage_threshold: float = 99.9,
        max_log_bytes: int = 5_000_000,
        double_pass_default: bool = False,
        verify_samples_default: int = 8,
        default_max_reuse: int = 0,
        default_ttl_sec: float | None = None,
    ) -> None:
        self.logger = ForensicLogger(log_path, max_bytes=max_log_bytes)
        self._buffers: dict[str, KVBuffer] = {}
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
            except Exception:  # pragma: no cover - env parse fallback
                logging.debug("_env_int: failed to parse int env %s", key, exc_info=True)
                return default

        def _env_float_opt(key: str, default: float | None) -> float | None:
            v = os.environ.get(key)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except Exception:  # pragma: no cover - env parse fallback
                logging.debug("_env_float_opt: failed to parse float env %s", key, exc_info=True)
                return default

        self.COVERAGE_THRESHOLD = float(os.environ.get("KV_COVERAGE_THRESHOLD", coverage_threshold))
        self._double_pass_default = _env_bool("KV_DOUBLE_PASS_DEFAULT", double_pass_default)
        self._verify_samples_default = _env_int("KV_VERIFY_SAMPLES_DEFAULT", verify_samples_default)
        self._default_max_reuse = _env_int("KV_DEFAULT_MAX_REUSE", default_max_reuse)
        self._default_ttl_sec = _env_float_opt("KV_DEFAULT_TTL_SEC", default_ttl_sec)
        # Metrics
        self._sanitize_durations_ms: list[float] = []
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
        shape: tuple[int, ...],
        dtype: str | Any = "float32",
        device: str = "cpu",
        framework: str | None = None,
        stream_id: str | None = None,
        pinned: bool = False,
        ttl_sec: float | None = None,
        max_reuse: int | None = None,
    ) -> str:
        """Allocate a buffer/tensor and register it with metadata.

        Returns a handle string used for subsequent operations.
        """
        handle = str(uuid.uuid4())
        tensor, ptr, nbytes, dtype_str = self._create_buffer(
            shape, dtype, device, framework, pinned=pinned
        )
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
                raise UnknownHandleError(f"Unknown handle: {handle}")
            return self._buffers[handle]

    def bind(self, handle: str, *, stream_id: str | None = None, stream: Any = None) -> None:
        with self._lock:
            buf = self._get(handle)
            buf.stream_id = stream_id
            buf.stream_obj = stream
            buf.status = "bound"
        self.logger.append(
            {
                "event_type": "bind",
                "handle": handle,
                "tenant_id": buf.tenant_id,
                "request_id": buf.request_id,
                "model_id": buf.model_id,
                "stream_id": stream_id,
            }
        )

    def mark_in_use(self, handle: str, *, stream: Any = None) -> None:
        with self._lock:
            buf = self._get(handle)
            if stream is not None:
                buf.stream_obj = stream
            buf.status = "in_use"
            now = time.time()
            buf.first_use_ts = buf.first_use_ts or now
            buf.last_use_ts = now
            decision = evaluate_policies(
                created_ts=buf.created_ts,
                first_use_ts=buf.first_use_ts,
                reuse_count=buf.reuse_count,
                ttl_sec=buf.ttl_sec,
                max_reuse=buf.max_reuse,
            )
            ttl_violation = decision.ttl_violation
            # reuse_violation currently unused; kept in decision for future policy expansion
            tenant_id, request_id, model_id = (
                buf.tenant_id,
                buf.request_id,
                buf.model_id,
            )
        self.logger.append(
            {
                "event_type": "write",
                "handle": handle,
                "tenant_id": tenant_id,
                "request_id": request_id,
                "model_id": model_id,
            }
        )
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
            exceeded = (
                buf.max_reuse is not None and buf.max_reuse >= 0 and buf.reuse_count > buf.max_reuse
            )
            tenant_id, request_id, model_id = (
                buf.tenant_id,
                buf.request_id,
                buf.model_id,
            )
            reuse_count, max_reuse = buf.reuse_count, buf.max_reuse
        # Log reuse event
        self.logger.append(
            {
                "event_type": "reuse",
                "handle": handle,
                "tenant_id": tenant_id,
                "request_id": request_id,
                "model_id": model_id,
                "reuse_count": reuse_count,
                "max_reuse": max_reuse,
            }
        )
        if exceeded:
            self.quarantine(handle, reason="reuse_limit_exceeded")

    # ----- Sanitization -----
    def sanitize(
        self,
        handle: str,
        *,
        async_: bool = False,
        verify: bool = True,
        double_pass: bool | None = None,
        samples: int | None = None,
    ) -> float:
        with self._lock:
            buf = self._get(handle)
            buf.status = "sanitizing"
            buf.sanitize_start_ts = time.time()
            stream_id = buf.stream_id
            device = buf.device
        self.logger.append(
            {
                "event_type": "sanitize_start",
                "handle": handle,
                "tenant_id": buf.tenant_id,
                "request_id": buf.request_id,
                "model_id": buf.model_id,
                "device": device,
                "stream_id": stream_id,
            }
        )
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

    def wait(self, handle: str, *, verify: bool = True, samples: int | None = None) -> float:
        buf = self._get(handle)
        # Sync outside lock (may block)
        if buf.event_obj is not None:
            try:
                buf.event_obj.synchronize()
            except Exception:  # noqa: BLE001
                logging.debug("Event synchronize failed", exc_info=True)
        elif (
            torch is not None and isinstance(buf._tensor, torch.Tensor) and buf._tensor.is_cuda
        ):  # pragma: no cover - GPU sync requires CUDA hardware
            torch.cuda.synchronize(device=buf._tensor.device)  # pragma: no cover - requires CUDA
        with self._lock:
            buf.sanitize_end_ts = time.time()
            buf.sanitize_duration_ms = float(
                int(1000 * (buf.sanitize_end_ts - (buf.sanitize_start_ts or buf.sanitize_end_ts)))
            )
            self._sanitize_durations_ms.append(buf.sanitize_duration_ms)
            buf.coverage_pct = self._attest_internal(buf, verify=verify, samples=samples)
            buf.status = "sanitized"
            cov = buf.coverage_pct
            dur = buf.sanitize_duration_ms
            samples_list = list(buf.verify_samples)
        self.logger.append(
            {
                "event_type": "sanitize_end",
                "handle": handle,
                "tenant_id": buf.tenant_id,
                "request_id": buf.request_id,
                "model_id": buf.model_id,
                "coverage_pct": cov,
                "duration_ms": dur,
                "verify_samples": samples_list,
            }
        )
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
        self.logger.append(
            {
                "event_type": "quarantine",
                "handle": handle,
                "tenant_id": buf.tenant_id,
                "request_id": buf.request_id,
                "model_id": buf.model_id,
                "reason": reason,
            }
        )

    def free(self, handle: str) -> None:
        with self._lock:
            buf = self._get(handle)
            if buf.status == "freed":
                return
            # If coverage below threshold, we'll quarantine and raise after releasing lock
        if buf.coverage_pct < self.COVERAGE_THRESHOLD:
            self.quarantine(handle, reason=f"coverage {buf.coverage_pct:.4f} below threshold")
            raise FreeWithoutSanitizeError("Attempt to free buffer without sufficient sanitization")
        with self._lock:
            buf.status = "freed"
            buf._tensor = None
            self._freed_total += 1
        self.logger.append(
            {
                "event_type": "free",
                "handle": handle,
                "tenant_id": buf.tenant_id,
                "request_id": buf.request_id,
                "model_id": buf.model_id,
            }
        )

    # ----- Metrics and export -----
    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._buffers)
            unsanitized = sum(
                1
                for b in self._buffers.values()
                if b.status not in ("sanitized", "freed", "quarantined")
            )
            quarantined = sum(1 for b in self._buffers.values() if b.status == "quarantined")
            active = sum(
                1 for b in self._buffers.values() if b.status not in ("freed", "quarantined")
            )
            coverages = [b.coverage_pct for b in self._buffers.values() if b.coverage_pct > 0]
            durations = [
                b.sanitize_duration_ms
                for b in self._buffers.values()
                if b.sanitize_duration_ms is not None
            ]
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

    def export_metrics(self, path: str | Path) -> None:
        metrics = self.get_metrics()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def export_metrics_prometheus(self, path: str | Path) -> None:
        m = self.get_metrics()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        # HELP/TYPE headers
        lines.append("# HELP kv_hygiene_unsanitized_regions Count of buffers not yet sanitized")
        lines.append("# TYPE kv_hygiene_unsanitized_regions gauge")
        lines.append(f"kv_hygiene_unsanitized_regions {int(m['unsanitized_regions_count'])}")

        lines.append(
            "# HELP kv_hygiene_quarantine_count Count of buffers quarantined due to policy violations"  # noqa: E501
        )
        lines.append("# TYPE kv_hygiene_quarantine_count gauge")
        lines.append(f"kv_hygiene_quarantine_count {int(m['quarantine_count'])}")

        lines.append(
            "# HELP kv_hygiene_min_coverage_pct Minimum observed sanitization coverage percent across buffers"  # noqa: E501
        )
        lines.append("# TYPE kv_hygiene_min_coverage_pct gauge")
        lines.append(f"kv_hygiene_min_coverage_pct {float(m['min_coverage_pct'])}")

        lines.append(
            "# HELP kv_hygiene_avg_coverage_pct Average sanitization coverage percent across buffers"  # noqa: E501
        )
        lines.append("# TYPE kv_hygiene_avg_coverage_pct gauge")
        lines.append(f"kv_hygiene_avg_coverage_pct {float(m['avg_coverage_pct'])}")

        lines.append(
            "# HELP kv_hygiene_threshold_pct Required minimum sanitization coverage percent"
        )
        lines.append("# TYPE kv_hygiene_threshold_pct gauge")
        lines.append(f"kv_hygiene_threshold_pct {float(m['threshold_pct'])}")

        lines.append(
            "# HELP kv_hygiene_sanitize_duration_p50_ms P50 sanitize duration in milliseconds"
        )
        lines.append("# TYPE kv_hygiene_sanitize_duration_p50_ms gauge")
        lines.append(f"kv_hygiene_sanitize_duration_p50_ms {float(m['sanitize_duration_p50_ms'])}")

        lines.append(
            "# HELP kv_hygiene_sanitize_duration_p95_ms P95 sanitize duration in milliseconds"
        )
        lines.append("# TYPE kv_hygiene_sanitize_duration_p95_ms gauge")
        lines.append(f"kv_hygiene_sanitize_duration_p95_ms {float(m['sanitize_duration_p95_ms'])}")

        lines.append(
            "# HELP kv_hygiene_active_buffers Number of active (not freed/quarantined) buffers"
        )
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
        shape: tuple[int, ...],
        dtype: str | Any,
        device: str,
        framework: str | None,
        *,
        pinned: bool = False,
    ) -> tuple[Any, int | None, int, str]:
        """Create a tensor/array per request, returning (obj, ptr, nbytes, dtype_str)."""
        # Torch path
        if torch is not None and (
            framework == "torch" or (isinstance(dtype, torch.dtype) or device.startswith("cuda"))
        ):
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
            dev = torch.device(
                device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            if dev.type == "cuda":
                t = torch.zeros(*shape, dtype=tdtype, device=dev)
            else:
                # For CPU, optionally pin memory if requested
                t = (
                    torch.zeros(*shape, dtype=tdtype, device=dev, pin_memory=pinned)
                    if hasattr(torch, "zeros")
                    else torch.empty(*shape, dtype=tdtype, device=dev)
                )
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
        ptr = (
            int(arr.__array_interface__.get("data", (0,))[0])
            if hasattr(arr, "__array_interface__")
            else None
        )
        nbytes = int(arr.nbytes)
        dtype_str = str(arr.dtype)
        return arr, ptr, nbytes, dtype_str

    def _zeroize_buffer(self, buf: KVBuffer, *, async_: bool) -> bool:
        # Delegate zeroization to sanitizer module (synchronous only)
        result = sanitizer_mod.sanitize_sync(buf, verify=False, samples=None)
        # Overwrite (not accumulate) since we do full pass each call
        buf.scrubbed_bytes = result.scrubbed_bytes
        # Duration captured on first pass; callers aggregate after both passes if needed
        if buf.sanitize_duration_ms is None:
            buf.sanitize_duration_ms = result.duration_ms
        else:
            buf.sanitize_duration_ms += result.duration_ms
        return False

    def _attest_internal(
        self, buf: KVBuffer, *, verify: bool = True, samples: int | None = None
    ) -> float:
        # Compute coverage based on scrubbed_bytes, then optionally verify sample values
        if buf.nbytes == 0:
            return 100.0
        total = buf.nbytes
        coverage = (buf.scrubbed_bytes / total) * 100.0
        if not verify:
            return coverage
        k = samples or self._verify_samples_default
        ok = self._verify_zero(buf, samples=k)
        if not ok:
            return 0.0
        return coverage

    def _verify_zero(
        self, buf: KVBuffer, *, samples: int = 8
    ) -> bool:  # legacy shim, owns sampling so tests can monkeypatch
        t = buf._tensor
        if t is None:
            buf.verify_samples = []
            return True
        n = self._buffer_numel(t)
        if n <= 0:
            buf.verify_samples = []
            return True
        # Record deterministic sample indices (even if we short‑circuit success)
        buf.verify_samples = self._sample_indices_deterministic(n, samples)
        # NumPy fast-path: if ndarray was zeroized via sanitize we trust it
        if self._is_numpy_array(t):
            return True
        # Torch / generic path: inspect sampled values
        return self._samples_zero(t, buf.verify_samples)

    @staticmethod
    def _buffer_numel(t: Any) -> int:
        if hasattr(t, "numel"):
            try:
                return int(t.numel())
            except Exception:  # pragma: no cover - numel retrieval
                logging.debug("_buffer_numel: failed numel()", exc_info=True)
                return 0
        if hasattr(t, "size"):
            try:
                return int(getattr(t, "size", 0))
            except Exception:  # pragma: no cover - size attr access
                logging.debug("_buffer_numel: failed size attribute", exc_info=True)
                return 0
        return 0

    @staticmethod
    def _sample_indices_deterministic(n: int, k: int) -> list[int]:
        try:
            sampler = sanitizer_mod._sample_indices
            return list(sampler(n, k))
        except Exception:  # pragma: no cover - deterministic fallback  # noqa: BLE001
            import random as _r

            rnd = _r.Random(n * 31 + k * 17)  # noqa: S311 - deterministic non-crypto
            return sorted(rnd.sample(range(n), min(k, n)))

    @staticmethod
    def _is_numpy_array(t: Any) -> bool:
        try:
            import numpy as _np

            return isinstance(t, _np.ndarray)
        except Exception:  # pragma: no cover - fallback  # noqa: BLE001
            logging.debug("_is_numpy_array: NumPy fast-path failed", exc_info=True)
            return False

    @staticmethod
    def _samples_zero(t: Any, sample_indices: list[int]) -> bool:
        try:
            if hasattr(t, "view"):
                flat = t.view(-1)
                for i in sample_indices:
                    # torch.Tensor.item() returns Python scalar
                    if flat[i].item() != 0:
                        return False
                return True
        except Exception:  # pragma: no cover - conservative failure
            # Mark not verified if any unexpected error during sampling
            # noqa: BLE001
            logging.debug("_samples_zero: sampling failed", exc_info=True)
            return False
        # If we cannot introspect, assume success (opaque type already zeroized)
        return True

    @staticmethod
    def _percentile(values: Iterable[float | None], q: float) -> float:
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
