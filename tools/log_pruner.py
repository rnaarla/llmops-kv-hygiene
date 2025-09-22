from __future__ import annotations

import time
from pathlib import Path


def prune_rotated_logs(
    log_path: Path,
    *,
    retention_days: int | None = None,
    max_rotated: int | None = None,
    archive_dir: Path | None = None,
) -> list[str]:
    """Prune rotated forensic log files by age and/or count.

    Rotated files follow naming pattern '<stem>-<ts>.log'. The *active* file '<stem>.log'
    is never deleted. When both retention_days and max_rotated are supplied the age
    pruning executes first, then count-based pruning runs on the remainder.

    Returns list of removed file names (not full paths) in order of removal.
    """
    removed: list[str] = []
    base = log_path
    stem = base.stem
    directory = base.parent
    rotated = sorted(directory.glob(f"{stem}-*.log"), key=lambda p: p.name)

    # Age-based pruning
    if retention_days is not None and retention_days >= 0:
        cutoff = time.time() - (retention_days * 86400)
        for p in list(rotated):  # iterate over copy because we mutate rotated
            try:
                ts_part = p.stem.split("-")[-1]
                ts = int(ts_part)
            except Exception:  # pragma: no cover - malformed rotated filename path
                import logging

                logging.debug("prune_rotated_logs: filename timestamp parse failed", exc_info=True)
                ts = int(p.stat().st_mtime)
            if ts < cutoff:
                if archive_dir:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    dest = archive_dir / p.name
                    p.replace(dest)
                else:
                    p.unlink(missing_ok=True)
                removed.append(p.name)
                rotated.remove(p)

    # Count-based pruning (keep newest N)
    if max_rotated is not None and max_rotated >= 0 and len(rotated) > max_rotated:
        to_remove = rotated[:-max_rotated]
        for p in to_remove:
            if archive_dir:
                archive_dir.mkdir(parents=True, exist_ok=True)
                dest = archive_dir / p.name
                p.replace(dest)
            else:
                p.unlink(missing_ok=True)
            removed.append(p.name)

    return removed


__all__ = ["prune_rotated_logs"]
