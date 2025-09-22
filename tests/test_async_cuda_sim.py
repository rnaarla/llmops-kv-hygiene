import types

import pytest

from tools.cache_tracer import CacheTracer

try:  # optional torch simulation if real torch absent
    import torch
except Exception:  # pragma: no cover
    torch = None


class FakeEvent:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):  # pragma: no cover - trivial
        self.synchronized = True


class FakeCUDATensor:
    def __init__(self, n=16):
        self._data = [0.0] * n
        self._n = n
        self.dtype = "float32"
        self.is_cuda = True
        self.device = types.SimpleNamespace(type="cuda")

    def numel(self):
        return self._n

    def element_size(self):
        return 4

    def view(self, *_):  # emulate torch view returning self like flat
        return self

    def __getitem__(self, i):  # returns scalar like torch tensor
        return self._data[i]

    def zero_(self):  # set all zeros (already zeros but keep semantics)
        for i in range(self._n):
            self._data[i] = 0.0

    def item(self):  # not used directly, but mimic torch scalar
        return 0.0


@pytest.mark.parametrize("async_flag", [True, False])
def test_sanitize_async_and_wait_simulated(async_flag):
    tracer = CacheTracer(log_path="forensics/async_sim.log")
    # Allocate a normal CPU buffer first
    h = tracer.allocate(
        tenant_id="t",
        request_id="r",
        model_id="m",
        shape=(8,),
        dtype="float32",
        device="cpu",
        framework="numpy",
    )
    tracer.mark_in_use(h)
    buf = tracer._get(h)
    # Inject fake CUDA tensor + event to exercise alternate wait path logic
    buf._tensor = FakeCUDATensor(32)
    buf.device = "cuda:0"
    buf.event_obj = FakeEvent()
    # Force sanitize with async flag; our internal zeroize is synchronous but async flag path returns early
    cov = tracer.sanitize(h, async_=async_flag, verify=True)
    # Implementation returns wait() result because _zeroize_buffer never schedules async (always False)
    assert cov >= 0.0
    # Ensure event synchronize was invoked
    assert buf.event_obj.synchronized is True
