"""Typed integration boundary for KV hygiene hooks at the allocator / pool layer."""

from __future__ import annotations

from typing import Any, Protocol


class HygieneRuntimeHooks(Protocol):
    """Minimal contract for wiring `CacheTracer` into a serving stack.

    Implementations allocate through the tracer, call `mark_in_use` / `sanitize` /
    `wait` / `free` at pool boundaries, and never reuse memory across tenants without
    a successful sanitize + coverage attestation.
    """

    def hygiene_allocate(
        self,
        *,
        tenant_id: str,
        request_id: str,
        model_id: str,
        shape: tuple[int, ...],
        dtype: str | Any,
        device: str,
        framework: str | None,
    ) -> str:
        """Return a tracer handle for a new KV region."""
        ...

    def hygiene_release(self, handle: str, *, async_sanitize: bool = False) -> None:
        """Sanitize, wait, and free the handle (fail-secure on violation)."""
        ...
