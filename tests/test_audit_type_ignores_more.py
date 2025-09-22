import io
import sys
from pathlib import Path

from tools import audit_type_ignores as audit


def _mk(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


def test_audit_main_no_violations(monkeypatch, tmp_path):
    # Scenario: only allowlisted code -> exit 0, no header printed
    pfx = "# type:"
    ign = " ignore"
    f = _mk(tmp_path, "ok.py", f"x=1  {pfx}{ign}[attr-defined]\n")
    monkeypatch.setattr(audit, "ALLOWED_CODES", {"attr-defined"})

    # Restrict file iteration to our single file
    monkeypatch.setattr(audit, "iter_python_files", lambda root: [f])

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    rc = audit.main()
    out = buf.getvalue()
    assert rc == 0
    assert "violations" not in out.lower()


def test_audit_main_with_multiple_violations(monkeypatch, tmp_path):
    # Mix of bare, unknown code, assignment, and decoy that should produce only specific violations.
    pfx = "# type:"
    ign = " ignore"
    bare = _mk(tmp_path, "bare.py", f"a=1  {pfx}{ign}\n")
    unknown = _mk(tmp_path, "unknown.py", f"b=2  {pfx}{ign}[attr-defined]\n")
    assignment = _mk(tmp_path, "assign.py", f"c=3  {pfx}{ign}[assignment]\n")
    # Decoy: contains the sequence but not matching any regex (no space before fragment continuation)
    decoy = _mk(tmp_path, "decoy.py", f"d=4  {pfx}{ign}XYZ\n")

    # Ensure allowlist empty
    monkeypatch.setattr(audit, "ALLOWED_CODES", set())
    monkeypatch.setattr(audit, "iter_python_files", lambda root: [bare, unknown, assignment, decoy])

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    rc = audit.main()
    out = buf.getvalue().splitlines()

    assert rc == 1
    header = next((line for line in out if "violations" in line.lower()), None)
    assert header is not None
    joined = "\n".join(out)
    # Expect bare, unknown code, assignment; decoy should NOT appear as violation
    assert "bare" in joined
    assert "attr-defined" in joined
    assert "assignment" in joined
    assert "decoy" not in joined
