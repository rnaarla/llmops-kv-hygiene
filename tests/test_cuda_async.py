import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from tools.cache_tracer import CacheTracer


def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_cuda_async_sanitize_and_wait(tmp_path):
    tracer = CacheTracer(log_path=tmp_path / "kv_cache.log")
    device = f"cuda:{torch.cuda.current_device()}"
    stream = torch.cuda.Stream()

    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(1024,),
        dtype="float32",
        device=device,
        framework="torch",
    )
    tracer.bind(h, stream_id="s1", stream=stream)
    tracer.mark_in_use(h, stream=stream)

    # Schedule async sanitize on stream
    cov0 = tracer.sanitize(h, async_=True, verify=True)
    assert cov0 == 0.0  # scheduled, not yet complete

    # Now wait for completion and attest
    cov = tracer.wait(h, verify=True)
    assert cov >= tracer.COVERAGE_THRESHOLD
    # Ensure we can free after sanitize
    tracer.free(h)
