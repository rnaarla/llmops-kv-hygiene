from __future__ import annotations

from tools.adapters.reference_hooks import TracerHygieneHooks
from tools.cache_tracer import CacheTracer


def test_tracer_hooks_allocate_release_numpy(tmp_path) -> None:
    tracer = CacheTracer(log_path=tmp_path / "kv.log")
    hooks = TracerHygieneHooks(tracer)
    h = hooks.hygiene_allocate(
        tenant_id="a",
        request_id="b",
        model_id="c",
        shape=(32,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    hooks.hygiene_release(h, async_sanitize=False)
    m = tracer.get_metrics()
    assert m["quarantine_count"] == 0
    assert m["unsanitized_regions_count"] == 0
