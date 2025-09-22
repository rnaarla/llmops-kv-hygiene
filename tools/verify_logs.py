from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, List

from .forensic_logger import ForensicLogger


def verify_all_and_write(log_path: Path, out_path: Optional[Path] = None) -> dict:
    res = ForensicLogger.verify_all(str(log_path))
    payload = {
        "schema": 1,
        "ts": time.time(),
        "log": str(log_path),
        "result": res,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))
    return res


def prune_rotated(log_path: Path, *, retention_days: Optional[int] = None, max_rotated: Optional[int] = None, archive_dir: Optional[Path] = None) -> List[str]:
    """Prune rotated logs based on age or count. Returns list of removed file names.

    Rotated files are named '<stem>-<ts>.log'. Active file '<stem>.log' is never removed.
    """
    removed: List[str] = []
    base = log_path
    stem = base.stem
    directory = base.parent
    rotated = sorted(directory.glob(f"{stem}-*.log"), key=lambda p: p.name)

    # By age
    if retention_days is not None and retention_days >= 0:
        cutoff = time.time() - (retention_days * 86400)
        for p in list(rotated):
            try:
                ts_part = p.stem.split("-")[-1]
                ts = int(ts_part)
            except Exception:
                # If name doesn't parse as timestamp, fall back to mtime
                ts = int(p.stat().st_mtime)
            if ts < cutoff:
                if archive_dir:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    dest = archive_dir / p.name
                    p.replace(dest)
                else:
                    p.unlink(missing_ok=True)  # type: ignore[arg-type]
                removed.append(p.name)
                rotated.remove(p)

    # By count (keep newest N)
    if max_rotated is not None and max_rotated >= 0 and len(rotated) > max_rotated:
        to_remove = rotated[:-max_rotated]
        for p in to_remove:
            if archive_dir:
                archive_dir.mkdir(parents=True, exist_ok=True)
                dest = archive_dir / p.name
                p.replace(dest)
            else:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            removed.append(p.name)
        # Not strictly needed to update list further

    return removed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify forensic logs integrity and enforce retention")
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "/var/forensics"))
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE", "kv_cache.log"))
    parser.add_argument("--out", default=os.environ.get("VERDICT_OUT", "verification.json"))
    parser.add_argument("--retention-days", type=int, default=lambda: int(os.environ.get("RETENTION_DAYS", "0")) if os.environ.get("RETENTION_DAYS") else None)
    parser.add_argument("--max-rotated", type=int, default=lambda: int(os.environ.get("MAX_ROTATED", "0")) if os.environ.get("MAX_ROTATED") else None)
    parser.add_argument("--archive-dir", default=os.environ.get("ARCHIVE_DIR"))
    args = parser.parse_args(argv)

    # Resolve defaults for optional ints
    retention_days = None
    if callable(args.retention_days):
        retention_days = args.retention_days()
    else:
        retention_days = args.retention_days
    max_rotated = None
    if callable(args.max_rotated):
        max_rotated = args.max_rotated()
    else:
        max_rotated = args.max_rotated

    base = Path(args.log_dir) / args.log_file
    verdict_path = Path(args.out)
    archive_dir = Path(args.archive_dir) if args.archive_dir else None

    # Verify before pruning
    res_before = verify_all_and_write(base, verdict_path)
    ok_before = bool(res_before.get("ok", False))

    # Enforce retention if configured
    removed = prune_rotated(base, retention_days=retention_days, max_rotated=max_rotated, archive_dir=archive_dir)
    if removed:
        print(json.dumps({"schema": 1, "ts": time.time(), "removed": removed}))

    # Verify again after pruning (chain may now only cover remaining files)
    res_after = verify_all_and_write(base, verdict_path)
    ok_after = bool(res_after.get("ok", False))

    return 0 if (ok_before and ok_after) else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
