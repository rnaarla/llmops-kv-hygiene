"""Runnable `HygieneRuntimeHooks` implementation backed by `CacheTracer` (CPU / NumPy reference)."""

from __future__ import annotations

from typing import Any

from tools.cache_tracer import CacheTracer


class TracerHygieneHooks:
    """Reference pool adapter: allocate/sanitize/free through `CacheTracer`.

    Serving stacks (vLLM, TensorRT-LLM, custom pools) can mirror this sequencing at
    their own allocator boundaries while swapping buffer construction for framework tensors.
    """

    def __init__(self, tracer: CacheTracer, *, framework: str | None = "numpy") -> None:
        self._tracer = tracer
        self._framework = framework

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
        fw = framework if framework is not None else self._framework
        return self._tracer.allocate(
            tenant_id=tenant_id,
            request_id=request_id,
            model_id=model_id,
            shape=shape,
            dtype=dtype,
            device=device,
            framework=fw,
        )

    def hygiene_release(self, handle: str, *, async_sanitize: bool = False) -> None:
        self._tracer.mark_in_use(handle)
        self._tracer.sanitize(handle, async_=async_sanitize, verify=True)
        if async_sanitize:
            self._tracer.wait(handle)
        self._tracer.free(handle)
