"""Emit `forensics/coverage.json` for CI hygiene gates (eviction_checker / composite action)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tools.cache_tracer import CacheTracer

    out = Path(os.environ.get("KV_HYGIENE_METRICS_OUT", "forensics/coverage.json"))
    log = Path(os.environ.get("KV_HYGIENE_LOG", "forensics/kv_cache.log"))
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    tracer = CacheTracer(log_path=log)
    handle = tracer.allocate(
        tenant_id="ci",
        request_id="ci-gate",
        model_id="ci",
        shape=(256,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(handle)
    tracer.sanitize(handle, async_=False, verify=True)
    tracer.free(handle)
    tracer.export_metrics(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
