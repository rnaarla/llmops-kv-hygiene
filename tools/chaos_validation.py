"""Chaos validation harness for CI.

Runs a deterministic series of cache tracer operations while injecting
randomized scheduling to ensure the hygiene routines survive disorderly
execution patterns. Designed to be fast and side-effect free so it can run
as part of the pipeline health gate.
"""

from __future__ import annotations

import random
from pathlib import Path

from tools.cache_tracer import CacheTracer, ForensicLogger


def run_chaos_trials(trials: int = 8) -> None:
    random.seed(133742)  # noqa: S311 - deterministic chaos harness seed
    tracer = CacheTracer(log_path="chaos/kv_cache.log")
    Path("chaos").mkdir(exist_ok=True)

    handles = []
    for idx in range(trials):
        h = tracer.allocate(
            tenant_id="chaos",
            request_id=f"chaos-{idx}",
            model_id="m",
            shape=(32,),
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(h)
        handles.append(h)

    for h in handles:
        async_mode = random.choice([True, False])  # noqa: S311 - deterministic test harness
        tracer.sanitize(h, async_=async_mode, verify=True)
        if async_mode:
            tracer.wait(h, verify=True)
        if random.random() < 0.2:  # noqa: S311 - deterministic test harness
            tracer.quarantine(h, reason="chaos")
        tracer.free(h)

    tracer.export_metrics("chaos/metrics.json")
    result = ForensicLogger.verify_all("chaos/kv_cache.log")
    if not result.get("ok", False):
        raise SystemExit("Forensic chain verification failed during chaos validation.")


def main() -> None:
    run_chaos_trials()
    print("✅ Chaos validation completed successfully.")


if __name__ == "__main__":
    main()
