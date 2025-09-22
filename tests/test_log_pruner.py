from __future__ import annotations

import time
from pathlib import Path

from tools.log_pruner import prune_rotated_logs


def _touch(path: Path, ts: int | None = None):
    path.write_text("dummy")
    if ts is not None:
        # Set mtime for fallback path coverage if a malformed name is ever used
        import os

        os.utime(path, (ts, ts))


def test_retention_mid_range(tmp_path: Path):
    base = tmp_path / "kv_cache.log"
    base.write_text("")
    now = int(time.time())
    old_ts = now - 2 * 86400  # 2 days ago
    recent_ts = now - 3600  # 1 hour ago
    old_file = tmp_path / f"kv_cache-{old_ts}.log"
    recent_file = tmp_path / f"kv_cache-{recent_ts}.log"
    _touch(old_file, old_ts)
    _touch(recent_file, recent_ts)

    removed = prune_rotated_logs(base, retention_days=1, max_rotated=None)
    assert old_file.name in removed
    assert recent_file.name not in removed
    assert recent_file.exists()


def test_max_rotated_boundary(tmp_path: Path):
    base = tmp_path / "kv_cache.log"
    base.write_text("")
    now = int(time.time())
    # Create 4 rotated files with increasing timestamps
    files = []
    for i in range(4):
        ts = now - (1000 - i)  # ensure sorted order by name ~ timestamp ascending
        f = tmp_path / f"kv_cache-{ts}.log"
        _touch(f, ts)
        files.append(f)

    # Keep newest 3
    removed = prune_rotated_logs(base, max_rotated=3)
    # Expect exactly one removal: the oldest
    assert len(removed) == 1
    oldest = sorted(files, key=lambda p: p.name)[0]
    assert removed[0] == oldest.name
    assert not oldest.exists()
    # Remaining three still exist
    remaining = {p.name for p in files[1:]}
    existing = {p.name for p in tmp_path.glob("kv_cache-*.log")}
    assert remaining == existing


def test_combined_retention_then_count(tmp_path: Path):
    base = tmp_path / "kv_cache.log"
    base.write_text("")
    now = int(time.time())
    # Create 5 rotated; two old (2 days), three recent
    old_files = []
    recent_files = []
    for i in range(2):
        ts = now - 2 * 86400 - i  # slightly different to keep ordering
        f = tmp_path / f"kv_cache-{ts}.log"
        _touch(f, ts)
        old_files.append(f)
    for i in range(3):
        ts = now - (i + 10)  # recent
        f = tmp_path / f"kv_cache-{ts}.log"
        _touch(f, ts)
        recent_files.append(f)

    removed = prune_rotated_logs(base, retention_days=1, max_rotated=2)
    # First phase should delete the two old files
    old_names = {f.name for f in old_files}
    assert old_names.issubset(set(removed))
    # After age pruning, we have 3 recent left; count pruning keeps 2 newest => removes 1 oldest of the recent set
    recent_sorted = sorted(recent_files, key=lambda p: p.name)
    expected_second_phase = recent_sorted[0].name  # oldest among remaining
    assert expected_second_phase in removed
    # Ensure only 3 deletions total (2 old + 1 recent)
    assert len(removed) == 3
