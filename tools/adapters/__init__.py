"""Integration adapters for serving runtimes (vLLM, TensorRT-LLM, custom pools).

See `protocol.py` for the contract, `reference_hooks.py` for a runnable CPU adapter,
and `vllm_hooks` / `tensorrt_llm_hooks` for framework-specific placement notes.
"""

from .protocol import HygieneRuntimeHooks
from .reference_hooks import TracerHygieneHooks

__all__ = ["HygieneRuntimeHooks", "TracerHygieneHooks"]
