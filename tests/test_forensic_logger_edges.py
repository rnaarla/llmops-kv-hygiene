import json

import pytest

from tools.forensic_logger import ForensicLogger


def test_verify_chain_empty_file(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("", encoding="utf-8")
    res = ForensicLogger.verify_chain(p)
    assert res["ok"] is True and res["lines"] == 0


def test_verify_chain_tampered_curr_hash(tmp_path):
    p = tmp_path / "x.log"
    flog = ForensicLogger(p, max_bytes=500)
    flog.append({"event_type": "a"})
    flog.append({"event_type": "b"})
    lines = p.read_text().strip().splitlines()
    obj = json.loads(lines[-1])
    # Corrupt curr_hash only
    obj["curr_hash"] = "deadbeef" * 4
    lines[-1] = json.dumps(obj)
    p.write_text("\n".join(lines) + "\n")
    res = ForensicLogger.verify_chain(p)
    # first_bad_line is zero-based index of offending line; corrupting last line yields index len(lines)-1
    assert res["ok"] is False and res["first_bad_line"] == len(lines) - 1


def test_verify_all_missing_file(tmp_path):
    # For a missing file the current implementation raises FileNotFoundError when verifying chain
    missing = tmp_path / "no_such.log"
    with pytest.raises(FileNotFoundError):
        ForensicLogger.verify_all(str(missing))
