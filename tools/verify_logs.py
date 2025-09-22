from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .forensic_logger import ForensicLogger
from .log_pruner import prune_rotated_logs


def verify_all_and_write(log_path: Path, out_path: Path | None = None) -> dict:
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


def prune_rotated(*args, **kwargs):  # backward compatibility wrapper
    """Deprecated wrapper for compatibility; use prune_rotated_logs instead."""
    return prune_rotated_logs(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify forensic logs integrity and enforce retention"
    )
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "/var/forensics"))
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE", "kv_cache.log"))
    parser.add_argument("--out", default=os.environ.get("VERDICT_OUT", "verification.json"))
    parser.add_argument(
        "--retention-days",
        type=int,
        default=lambda: (
            int(os.environ.get("RETENTION_DAYS", "0")) if os.environ.get("RETENTION_DAYS") else None
        ),
    )
    parser.add_argument(
        "--max-rotated",
        type=int,
        default=lambda: (
            int(os.environ.get("MAX_ROTATED", "0")) if os.environ.get("MAX_ROTATED") else None
        ),
    )
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
    removed = prune_rotated_logs(
        base,
        retention_days=retention_days,
        max_rotated=max_rotated,
        archive_dir=archive_dir,
    )
    if removed:
        print(json.dumps({"schema": 1, "ts": time.time(), "removed": removed}))

    # Verify again after pruning (chain may now only cover remaining files)
    res_after = verify_all_and_write(base, verdict_path)
    ok_after = bool(res_after.get("ok", False))

    return 0 if (ok_before and ok_after) else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
