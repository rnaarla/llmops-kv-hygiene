from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVBuffer:
    handle: str
    tenant_id: str
    request_id: str
    model_id: str
    device: str  # e.g., 'cpu', 'cuda:0'
    shape: tuple[int, ...]
    dtype: str
    ptr: int | None
    nbytes: int
    created_ts: float = field(default_factory=time.time)
    stream_id: str | None = None
    status: str = "allocated"  # allocated|bound|in_use|sanitizing|sanitized|freed|quarantined
    scrubbed_bytes: int = 0
    sanitize_start_ts: float | None = None
    sanitize_end_ts: float | None = None
    coverage_pct: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)
    _tensor: Any = None  # torch.Tensor or np.ndarray
    pinned: bool = False
    reuse_count: int = 0
    first_use_ts: float | None = None
    last_use_ts: float | None = None
    ttl_sec: float | None = None
    max_reuse: int | None = None
    stream_obj: Any = None
    event_obj: Any = None
    verify_samples: list[int] = field(default_factory=list)
    sanitize_duration_ms: float | None = None
