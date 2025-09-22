from pathlib import Path

from tools import verify_logs
from tools.forensic_logger import ForensicLogger


def _make_base_log(tmp_path: Path) -> Path:
    base = tmp_path / "kv_cache.log"
    logger = ForensicLogger(base, max_bytes=10_000)
    # A couple of records (no rotation in this helper)
    logger.append({"event_type": "allocate", "handle": "h1"})
    logger.append({"event_type": "free", "handle": "h1"})
    return base


def test_verify_logs_main_lambda_defaults(tmp_path, monkeypatch, capsys):
    """Covers the callable() branches for retention/max_rotated defaults (lines 68-76).

    We ensure no CLI overrides are supplied so the parser stores the lambda objects as defaults,
    exercising the callable() paths that were previously uncovered.
    """
    base = _make_base_log(tmp_path)

    # Ensure env vars absent so lambda returns None
    for key in ["RETENTION_DAYS", "MAX_ROTATED", "ARCHIVE_DIR"]:
        monkeypatch.delenv(key, raising=False)

    rc = verify_logs.main(
        [
            "--log-dir",
            str(tmp_path),
            "--log-file",
            base.name,
            "--out",
            str(tmp_path / "verdict.json"),
        ]
    )
    captured = capsys.readouterr()
    # Two JSON payload prints (before & after, identical since no pruning)
    assert rc == 0
    assert captured.out.count("schema") >= 2
    assert (tmp_path / "verdict.json").exists()


def test_verify_logs_main_explicit_int_args_no_rotation(tmp_path):
    """Covers non-callable integer branches for retention/max_rotated without needing rotated logs.

    Passing explicit --retention-days and --max-rotated produces int objects, exercising else branches
    in the resolution logic (lines 69-75 in verify_logs.py). No rotation means verification stays OK.
    """
    base = tmp_path / "kv_cache.log"
    logger = ForensicLogger(base)
    logger.append({"event_type": "allocate", "handle": "x"})
    logger.append({"event_type": "free", "handle": "x"})
    rc = verify_logs.main(
        [
            "--log-dir",
            str(tmp_path),
            "--log-file",
            base.name,
            "--out",
            str(tmp_path / "verdict_int.json"),
            "--retention-days",
            "5",
            "--max-rotated",
            "3",
        ]
    )
    assert rc == 0
    assert (tmp_path / "verdict_int.json").exists()


def test_prune_rotated_age_only(tmp_path):
    """Directly exercise age-based pruning in prune_rotated without involving verify_all chain semantics."""
    base = tmp_path / "kv_cache.log"
    base.write_text("{}\n")
    import time as _time
    now = int(_time.time())
    old_ts = [now - 400, now - 300]
    for ts in old_ts:
        p = tmp_path / f"kv_cache-{ts}.log"
        p.write_text("{}\n")
    removed = verify_logs.prune_rotated(base, retention_days=0, max_rotated=None, archive_dir=None)
    assert len(removed) == len(old_ts)
    assert not list(tmp_path.glob("kv_cache-*.log"))


def test_prune_rotated_count_only(tmp_path):
    """Exercise count-based pruning path keeping newest N rotated logs."""
    base = tmp_path / "kv_cache.log"
    base.write_text("{}\n")
    import time as _time
    now = int(_time.time())
    ts_values = [now - 400, now - 300, now - 200, now - 100]
    for ts in ts_values:
        p = tmp_path / f"kv_cache-{ts}.log"
        p.write_text("{}\n")
    # Keep only newest 1 rotated file
    removed = verify_logs.prune_rotated(base, retention_days=None, max_rotated=1, archive_dir=None)
    remaining = sorted(tmp_path.glob("kv_cache-*.log"))
    assert len(remaining) == 1
    assert len(removed) == 3
