from __future__ import annotations

from pathlib import Path

from tools.forensic_logger import ForensicLogger


def test_mirror_receives_same_lines_as_primary(tmp_path: Path) -> None:
    primary = tmp_path / "kv.log"
    mirror = tmp_path / "kv.mirror.log"
    flog = ForensicLogger(primary, max_bytes=50_000, mirror_paths=[mirror])
    flog.append({"event_type": "test", "n": 1})
    flog.append({"event_type": "test", "n": 2})
    p_lines = primary.read_text(encoding="utf-8").strip().splitlines()
    m_lines = mirror.read_text(encoding="utf-8").strip().splitlines()
    assert len(p_lines) == 2
    assert p_lines == m_lines


def test_mirror_survives_broken_secondary(tmp_path: Path) -> None:
    primary = tmp_path / "kv.log"

    def boom(_line: str) -> None:
        raise OSError("sink unavailable")

    flog = ForensicLogger(primary, max_bytes=50_000, mirror_sinks=[boom])
    flog.append({"event_type": "ok", "n": 3})
    assert "ok" in primary.read_text(encoding="utf-8")
