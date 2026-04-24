"""Integration adapters for serving runtimes (vLLM, TensorRT-LLM, custom pools).

Concrete adapters are organization-specific; this package holds the **contract**
implementations should follow. See `protocol.py` and `docs/PRODUCT_SPECIFICATION.md`.
"""

from .protocol import HygieneRuntimeHooks

__all__ = ["HygieneRuntimeHooks"]
