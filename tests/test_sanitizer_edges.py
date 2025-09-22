from types import SimpleNamespace

from tools import sanitizer


class DummyBuf:
    def __init__(self, tensor, device: str, nbytes: int):
        self._tensor = tensor
        self.device = device
        self.nbytes = nbytes


class FailingTensor:
    def __init__(self, n):
        self._n = n
        self._data = [1] * n
        self.device = SimpleNamespace(type="cpu")

    def numel(self):  # used for size
        return self._n

    def element_size(self):
        raise RuntimeError("size boom")  # trigger defensive branch

    def zero_(self):  # still let zeroization attempt
        for i in range(self._n):
            self._data[i] = 0

    def view(self, *_, **__):  # fallback path not used here
        return self


def test_zeroize_cpu_numpy(monkeypatch):
    import numpy as np

    arr = np.ones((4,), dtype="int32")
    buf = DummyBuf(arr, "cpu", arr.nbytes)
    scrubbed = sanitizer.zeroize_cpu(buf)
    assert scrubbed == arr.nbytes
    assert arr.sum() == 0


def test_zeroize_cpu_torch(monkeypatch):
    torch = sanitizer.torch
    if torch is None:  # optional dependency absent
        return
    t = torch.ones(8, dtype=torch.float32)
    buf = DummyBuf(t, "cpu", t.numel() * t.element_size())
    scrubbed = sanitizer.zeroize_cpu(buf)
    assert scrubbed == t.numel() * t.element_size()
    assert float(t.sum().item()) == 0.0


def test_zeroize_cpu_unsupported():
    buf = DummyBuf(object(), "cpu", 0)
    assert sanitizer.zeroize_cpu(buf) == 0


def test_zeroize_cpu_failure_paths(monkeypatch):
    # Trigger exception in element_size and still handle gracefully
    ft = FailingTensor(4)
    buf = DummyBuf(ft, "cpu", ft.numel() * 4)
    scrubbed = sanitizer.zeroize_cpu(buf)
    # Failure in element_size sets nbytes=0 then writes zeros; treated as 0 coverage
    assert scrubbed in (0, ft.numel() * 4)


def test_verify_zero_numpy_samples():
    import numpy as np

    arr = np.zeros((10,), dtype="int32")
    buf = DummyBuf(arr, "cpu", arr.nbytes)
    res = sanitizer.verify_zero(buf, samples=5)
    assert res is True


def test_sanitize_sync_cpu(monkeypatch):
    import numpy as np

    arr = np.ones((16,), dtype="float32")
    buf = DummyBuf(arr, "cpu", arr.nbytes)
    result = sanitizer.sanitize_sync(buf, verify=True, samples=4)
    assert result.coverage_pct == 100.0
    assert arr.sum() == 0
