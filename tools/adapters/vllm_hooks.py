"""vLLM-shaped integration notes and helpers (no `vllm` package dependency).

Hook `TracerHygieneHooks` at KV block / slot **free** or **evict** boundaries in the worker
so every reuse path runs sanitize + verify before memory is handed to another sequence.

Typical placement (version-dependent names): block manager eviction, prefix-cache swap,
or the allocator callback that returns a physical block to the free list. Pass
``framework="torch"`` and ``device="cuda"`` when wiring real GPU KV tensors.
"""

from __future__ import annotations

from .reference_hooks import TracerHygieneHooks

__all__ = ["TracerHygieneHooks"]
