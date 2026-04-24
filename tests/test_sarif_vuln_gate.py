import json
from pathlib import Path

import pytest

from tools import sarif_vuln_gate


def test_gate_passes_empty_results(tmp_path: Path) -> None:
    doc = {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "trivy"}}, "results": []}]}
    p = tmp_path / "empty.sarif"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert sarif_vuln_gate.main([str(p)]) == 0


def test_gate_fails_on_error_level(tmp_path: Path) -> None:
    doc = {
        "runs": [
            {
                "results": [
                    {"ruleId": "CVE-1", "level": "error"},
                ]
            }
        ]
    }
    p = tmp_path / "bad.sarif"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert sarif_vuln_gate.main([str(p)]) == 1


def test_main_missing_file() -> None:
    assert sarif_vuln_gate.main(["/nonexistent/path/file.sarif"]) == 1


@pytest.mark.parametrize("level", ["warning", "WARNING"])
def test_gate_fails_on_warning(tmp_path: Path, level: str) -> None:
    doc = {"runs": [{"results": [{"ruleId": "X", "level": level}]}]}
    p = tmp_path / "warn.sarif"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert sarif_vuln_gate.main([str(p)]) == 1
