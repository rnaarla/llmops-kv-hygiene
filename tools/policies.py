from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class PolicyDecision:
    ttl_violation: bool
    reuse_violation: bool

    def any(self) -> bool:  # noqa: D401 - simple predicate
        return self.ttl_violation or self.reuse_violation


def check_ttl(created_ts: float, first_use_ts: float | None, ttl_sec: float | None) -> bool:
    if ttl_sec is None:
        return False
    if ttl_sec <= 0:
        return True
    now = time.time()
    if first_use_ts is None:
        return (now - created_ts) > ttl_sec
    return (now - first_use_ts) > ttl_sec


def check_reuse(reuse_count: int, max_reuse: int | None) -> bool:
    if max_reuse is None:
        return False
    if max_reuse < 0:
        return False
    return reuse_count > max_reuse


def evaluate_policies(
    created_ts: float,
    first_use_ts: float | None,
    reuse_count: int,
    ttl_sec: float | None,
    max_reuse: int | None,
) -> PolicyDecision:
    return PolicyDecision(
        ttl_violation=check_ttl(created_ts, first_use_ts, ttl_sec),
        reuse_violation=check_reuse(reuse_count, max_reuse),
    )
