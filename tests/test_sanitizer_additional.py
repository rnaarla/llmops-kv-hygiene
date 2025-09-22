from types import SimpleNamespace

from tools import sanitizer


class DummyBuf:
    def __init__(self, tensor, device: str, nbytes: int):
        self._tensor = tensor
        self.device = device
        self.nbytes = nbytes


class TorchLikeTensor:
    def __init__(self, n: int):
        self._data = [1] * n
        self._n = n
        self.device = SimpleNamespace(type="cpu")

    def numel(self):
        return self._n

    def element_size(self):
        return 4

    def zero_(self):
        for i in range(self._n):
            self._data[i] = 0

    def sum(self):  # mimic torch semantics returning object with item()
        class _S:
            def __init__(self, v):
                self.v = v

            def item(self):
                return self.v

        return _S(sum(self._data))

    def view(self, *_, **__):
        return self  # flattened view representation

    class _Elem:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    def __getitem__(self, idx):
        return self._Elem(self._data[idx])


class FailingZeroTensor(TorchLikeTensor):
    def zero_(self):  # force failure branch in zeroize_cpu
        raise RuntimeError("fail zero")


class BadElementSizeTensor(TorchLikeTensor):
    def element_size(self):  # trigger size calculation exception
        raise RuntimeError("boom size")


def test_verify_zero_failure_sets_coverage_zero(monkeypatch):
    import numpy as np

    arr = np.ones((8,), dtype="int32")
    buf = DummyBuf(arr, "cpu", arr.nbytes)
    # Force verify_zero to report failure irrespective of actual zeros to exercise coverage reset logic.
    monkeypatch.setattr(sanitizer, "verify_zero", lambda *_args, **_kw: False)
    result = sanitizer.sanitize_sync(buf, verify=True, samples=4)
    assert result.coverage_pct == 0.0


def test_torch_like_tensor_path_when_torch_missing(monkeypatch):
    monkeypatch.setattr(sanitizer, "torch", object())
    t = TorchLikeTensor(5)
    buf = DummyBuf(t, "cpu", t.numel() * t.element_size())
    scrubbed = sanitizer.zeroize_cpu(buf)
    assert isinstance(scrubbed, int)


def test_zeroize_cpu_zero_failure(monkeypatch):
    t = FailingZeroTensor(4)
    buf = DummyBuf(t, "cpu", t.numel() * t.element_size())
    val = sanitizer.zeroize_cpu(buf)
    assert val in (0, t.numel() * t.element_size())


def test_zeroize_cpu_bad_element_size(monkeypatch):
    t = BadElementSizeTensor(3)
    buf = DummyBuf(t, "cpu", t.numel() * 4)
    val = sanitizer.zeroize_cpu(buf)
    assert isinstance(val, int)


def test_sample_indices_all_cases():
    f = sanitizer._sample_indices
    assert f(10, 0) == []
    assert f(5, 10) == [0, 1, 2, 3, 4]
    s = f(10, 3)
    assert len(s) == 3 and s == sorted(s) and len(set(s)) == 3


def test_verify_zero_none_samples():
    buf = DummyBuf(None, "cpu", 0)
    assert sanitizer.verify_zero(buf, None) is True


def test_verify_zero_torch_like(monkeypatch):
    class FakeTorch:
        def tensor(self, x):  # pragma: no cover - not used directly
            return x

    monkeypatch.setattr(sanitizer, "torch", FakeTorch())
    t = TorchLikeTensor(6)
    buf = DummyBuf(t, "cpu", t.numel() * t.element_size())
    # zeroize then verify
    sanitizer.zeroize_cpu(buf)
    ok = sanitizer.verify_zero(buf, samples=3)
    assert ok is True
