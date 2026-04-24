"""TensorRT-LLM–style integration notes (no `tensorrt_llm` dependency).

Mount `TracerHygieneHooks` where the engine recycles KV pages: completion of a request,
page manager release, or PagedAttention block return. Use ``framework="torch"`` with CUDA
device strings that match TRT-LLM worker visibility; call ``hygiene_release(...,
async_sanitize=True)`` only when your stream ordering matches ``CacheTracer.wait`` semantics.
"""

from __future__ import annotations

from .reference_hooks import TracerHygieneHooks

__all__ = ["TracerHygieneHooks"]
