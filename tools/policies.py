from __future__ import annotations
from typing import Optional
import time

class PolicyDecision:
    __slots__ = ("ttl_violation", "reuse_violation")
    def __init__(self, ttl_violation: bool, reuse_violation: bool):
        self.ttl_violation = ttl_violation
        self.reuse_violation = reuse_violation
    def any(self) -> bool:
        return self.ttl_violation or self.reuse_violation


def check_ttl(created_ts: float, first_use_ts: float, ttl_sec: Optional[float]) -> bool:
    if ttl_sec is None:
        return False
    if ttl_sec <= 0:
        return True
    now = time.time()
    # If time since creation > ttl before first use, violation
    if first_use_ts is None:
        return (now - created_ts) > ttl_sec
    return (now - first_use_ts) > ttl_sec


def check_reuse(reuse_count: int, max_reuse: Optional[int]) -> bool:
    if max_reuse is None:
        return False
    if max_reuse < 0:
        return False
    return reuse_count > max_reuse


def evaluate_policies(created_ts: float, first_use_ts: Optional[float], reuse_count: int, ttl_sec: Optional[float], max_reuse: Optional[int]) -> PolicyDecision:
    return PolicyDecision(
        ttl_violation=check_ttl(created_ts, first_use_ts, ttl_sec),
        reuse_violation=check_reuse(reuse_count, max_reuse),
    )
