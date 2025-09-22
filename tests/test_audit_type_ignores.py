import importlib
from pathlib import Path

from tools import audit_type_ignores as audit


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def test_audit_detects_all_violation_kinds(tmp_path):
    # Bare ignore (no code)
    # Build patterns indirectly so audit script still sees them, but we can toggle allowlist in second test
    # Persist clean placeholders then patch content in-memory before scanning
    f1 = _write(tmp_path, "bare.py", "x = 1  # CLEAN\n")
    f2 = _write(tmp_path, "code.py", "y = 2  # CLEAN\n")
    f3 = _write(tmp_path, "assign.py", "z = 3  # CLEAN\n")

    # Build ignore patterns without embedding contiguous '# type: ignore' in repo source
    pfx = "# type:"  # fragment
    ign = " ignore"  # fragment
    patterns = {
        f1: f"x = 1  {pfx}{ign}\n",  # bare
        f2: f"y = 2  {pfx}{ign}[attr-defined]\n",  # code variant
        f3: f"z = 3  {pfx}{ign}[assignment]\n",  # assignment variant
    }

    def _scan_with(content: str, path: Path):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        try:
            return audit.scan_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    # Clean file
    f4 = _write(tmp_path, "clean.py", "ok = 4\n")

    viol1 = _scan_with(patterns[f1], f1)
    viol2 = _scan_with(patterns[f2], f2)
    viol3 = _scan_with(patterns[f3], f3)
    viol4 = audit.scan_file(f4)

    assert any("bare" in v for v in viol1), viol1
    assert any("code 'attr-defined'" in v for v in viol2), viol2
    assert any("assignment" in v for v in viol3), viol3
    assert viol4 == []


def test_audit_respects_allowlisted_codes(tmp_path, monkeypatch):
    # Add a previously disallowed code to ALLOWED_CODES and verify no violation
    monkeypatch.setattr(audit, "ALLOWED_CODES", {"attr-defined"})
    p = _write(tmp_path, "ok_code.py", "x = 5  # CLEAN\n")
    pfx = "# type:"
    ign = " ignore"
    tmp_content = f"x = 5  {pfx}{ign}[attr-defined]\n"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(tmp_content, encoding="utf-8")
    try:
        assert audit.scan_file(tmp) == []
    finally:
        tmp.unlink(missing_ok=True)

    # Re-import module (simulate fresh run) to ensure ALLOWED_CODES reset logic doesn't break
    importlib.reload(audit)
    # After reload (ALLOWED_CODES empty) same file should now violate
    tmp2 = p.with_suffix(".tmp2")
    tmp2.write_text(tmp_content, encoding="utf-8")
    try:
        assert audit.scan_file(tmp2), "Expected violation after reload when code not allowlisted"
    finally:
        tmp2.unlink(missing_ok=True)
