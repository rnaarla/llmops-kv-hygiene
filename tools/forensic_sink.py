"""Optional append-only sinks for forensic JSONL replication (secondary path / SIEM hook).

Primary logging remains in `ForensicLogger`. Mirrors receive the exact same line bytes the
primary file receives, so downstream WORM or object-store upload jobs can tail a second path
without changing hash semantics on the primary file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path


def mirror_append_sink(path: str | Path) -> Callable[[str], None]:
    """Return a callable that appends one JSONL line (with trailing newline) to *path*."""

    dest = Path(path)

    def _emit(line: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(line)
        try:
            import os

            os.chmod(dest, 0o600)
        except Exception:  # pragma: no cover - best-effort hardening
            logging.debug("mirror_append_sink: chmod failed for %s", dest, exc_info=True)

    return _emit


def build_mirror_sinks(paths: Sequence[str | Path] | None) -> tuple[Callable[[str], None], ...]:
    if not paths:
        return ()
    return tuple(mirror_append_sink(p) for p in paths)


def emit_to_sinks(sinks: Sequence[Callable[[str], None]], line: str, *, context: str) -> None:
    """Invoke each sink; failures are logged and do not abort the primary writer."""
    if not sinks:
        return
    payload = line if line.endswith("\n") else line + "\n"
    for sink in sinks:
        try:
            sink(payload)
        except Exception:
            logging.error("Forensic mirror sink failed (%s)", context, exc_info=True)


__all__ = ["build_mirror_sinks", "emit_to_sinks", "mirror_append_sink"]
