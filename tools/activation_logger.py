"""
Activation Logger: hooks into inference to log rare or unexpected activations.

Provides simple anomaly detection by tracking activation statistics per layer and
flagging outliers. Intended for runtime monitoring during tests/fuzzing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
from pathlib import Path
import math
import time
import threading

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class ActivationStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences from the current mean

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))


class ActivationLogger:
    """Runtime activation monitor with basic z-score anomaly detection."""

    def __init__(self, out_path: str = "forensics/activations.jsonl", z_threshold: float = 6.0) -> None:
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.z_threshold = z_threshold
        self._stats: Dict[str, ActivationStats] = {}
        self._lock = threading.Lock()

    def _record(self, layer: str, values: Tuple[float, float, float, int], *, anomalous: bool, tenant_id: Optional[str] = None, request_id: Optional[str] = None) -> None:
        ts = time.time()
        rec = {
            "schema": 1,
            "ts": ts,
            "layer": layer,
            "mean": values[0],
            "std": values[1],
            "max": values[2],
            "numel": values[3],
            "anomalous": anomalous,
        }
        if tenant_id:
            rec["tenant_id"] = tenant_id
        if request_id:
            rec["request_id"] = request_id
        line = json.dumps(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def observe(self, layer: str, tensor: Any, *, tenant_id: Optional[str] = None, request_id: Optional[str] = None, rate_limit_hz: Optional[float] = 10.0) -> bool:
        """Observe an activation tensor and log anomalies with optional rate limiting.

        Returns True if anomalous, False otherwise.
        """
        if torch is not None and isinstance(tensor, torch.Tensor):
            t = tensor.detach()
            mean = float(t.mean().item()) if t.numel() > 0 else 0.0
            std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
            max_val = float(t.abs().max().item()) if t.numel() > 0 else 0.0
            numel = int(t.numel())
        else:
            try:  # type: ignore[index]
                mean, std, max_val, numel = float(tensor[0]), float(tensor[1]), float(tensor[2]), int(tensor[3])
            except Exception as e:  # pragma: no cover - defensive unreachable with current tests
                raise TypeError("Unsupported tensor type for ActivationLogger.observe") from e

        key = layer
        with self._lock:
            stats = self._stats.setdefault(key, ActivationStats())
            stats.update(mean)
            baseline_mean, baseline_std = stats.mean, stats.std
            # Simple rate limit per key
            now = time.time()
            rl_attr = getattr(stats, "_last_log_ts", 0.0)
            allow = True
            if rate_limit_hz is not None and rate_limit_hz > 0:
                min_dt = 1.0 / rate_limit_hz
                allow = (now - rl_attr) >= min_dt
            if allow:
                setattr(stats, "_last_log_ts", now)

        z = 0.0 if baseline_std == 0.0 else abs(mean - baseline_mean) / baseline_std
        anomalous = z > self.z_threshold or max_val > (baseline_mean + 10 * (baseline_std or 1.0))
        if allow:
            self._record(layer, (mean, std, max_val, numel), anomalous=anomalous, tenant_id=tenant_id, request_id=request_id)
        return anomalous


if __name__ == "__main__":  # pragma: no cover - manual smoke path
    if torch is not None:  # pragma: no cover - exercised only manually
        logger = ActivationLogger()
        for i in range(100):
            x = torch.randn(1024)
            logger.observe("layer1", x)
        x = torch.randn(1024) * 100.0
        flagged = logger.observe("layer1", x)
        print("Anomalous?", flagged)
    else:  # pragma: no cover
        logger = ActivationLogger()
        for i in range(5):
            logger.observe("layer1", (0.0, 1.0, 3.0, 1024))
        print("Logged without torch")
