from __future__ import annotations

from tools.adapters.protocol import HygieneRuntimeHooks


def test_protocol_is_runtime_checkable_import() -> None:
    assert HygieneRuntimeHooks.__name__ == "HygieneRuntimeHooks"
