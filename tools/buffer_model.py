from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


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
    status: str = (
        "allocated"  # allocated|bound|in_use|sanitizing|sanitized|freed|quarantined
    )
    scrubbed_bytes: int = 0
    sanitize_start_ts: Optional[float] = None
    sanitize_end_ts: Optional[float] = None
    coverage_pct: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)
    _tensor: Any = None  # torch.Tensor or np.ndarray
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
