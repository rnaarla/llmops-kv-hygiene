"""Central helpers for optional test dependencies.

Provides lightweight accessors so tests avoid repeating guarded imports and
do not assign None to module symbols (which confuses mypy under
``warn_unused_ignores``).

Usage::

    from test_support.optional_deps import opt
    np = opt.numpy()
    if np is None:
        pytest.skip("NumPy not available")

Torch note: Tests must check the returned module is not ``None`` before CUDA
specific logic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol


class _OptionalAccessor(Protocol):  # pragma: no cover - structural
    def numpy(self) -> Any | None: ...  # noqa: D401,E701
    def torch(self) -> Any | None: ...  # noqa: D401,E701


@lru_cache(maxsize=1)
def _import_numpy():
    try:  # pragma: no cover - import success path covered indirectly
        import numpy as _np

        return _np
    except Exception:  # pragma: no cover - absence branch
        return None


@lru_cache(maxsize=1)
def _import_torch():
    try:  # pragma: no cover
        import torch as _torch

        return _torch
    except Exception:  # pragma: no cover
        return None


class _Opt:
    def numpy(self):  # noqa: D401 - simple forwarder
        return _import_numpy()

    def torch(self):  # noqa: D401 - simple forwarder
        return _import_torch()


opt: _OptionalAccessor = _Opt()

__all__ = ["opt"]
