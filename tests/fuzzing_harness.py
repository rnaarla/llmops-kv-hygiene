import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from tools.cache_tracer import CacheTracer


def _write_sentinel(tracer: CacheTracer, handle: str, value: float = 1.0) -> None:
    # Directly access underlying tensor/array for test harness only
    buf = tracer._get(handle)
    t = buf._tensor
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if torch is not None and isinstance(t, torch.Tensor):
        t.fill_(value)
    elif np is not None:
        t.fill(value)


def _assert_zero(tracer: CacheTracer, handle: str) -> None:
    buf = tracer._get(handle)
    t = buf._tensor
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if torch is not None and isinstance(t, torch.Tensor):
        if t.numel() > 0:
            assert int(torch.count_nonzero(t).item()) == 0
    elif np is not None:
        if t.size > 0:
            assert int((t != 0).sum()) == 0


@pytest.mark.timeout(30)
def test_fuzz_kv_cross_session_no_leak(tmp_path):
    rnd = random.Random(1337)
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")

    def one_iter(i: int) -> None:
        tenant = f"tenant-{i % 4}"
        req = f"req-{i}-{rnd.randint(1, 9999)}"
        size = rnd.randint(16, 4096)
        shape = (size,)
        handle = tracer.allocate(
            tenant_id=tenant,
            request_id=req,
            model_id="m",
            shape=shape,
            dtype="float32",
            device="cpu",
            framework="numpy",
        )
        tracer.mark_in_use(handle)
        _write_sentinel(tracer, handle, value=1.0)
        cov = tracer.sanitize(handle, async_=False, verify=True)
        assert cov >= tracer.COVERAGE_THRESHOLD
        _assert_zero(tracer, handle)
        tracer.free(handle)

    iters = 64
    # Run part single-threaded, part multi-threaded
    for i in range(iters // 2):
        one_iter(i)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(one_iter, i) for i in range(iters // 2, iters)]
        for f in as_completed(futures):
            f.result()

    metrics = tracer.get_metrics()
    assert metrics["quarantine_count"] == 0
    assert metrics["unsanitized_regions_count"] == 0
