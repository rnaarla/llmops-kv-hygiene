"""Additional tests to boost audit_type_ignores coverage above 90%."""

from tools import audit_type_ignores as audit


def test_iter_python_files_skips_venv_build_dist(tmp_path):
    """Test that iter_python_files correctly skips .venv, build, and dist directories."""
    # Create directory structure with files that should be included
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("# valid", encoding="utf-8")

    # Create directories that should be skipped
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("# skip", encoding="utf-8")

    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("# skip", encoding="utf-8")

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "package.py").write_text("# skip", encoding="utf-8")

    # Collect files
    files = list(audit.iter_python_files(tmp_path))
    file_names = [f.name for f in files]

    # Should include only module.py, not files in .venv, build, or dist
    assert "module.py" in file_names
    assert "lib.py" not in file_names
    assert "generated.py" not in file_names
    assert "package.py" not in file_names


def test_iter_python_files_nested_excluded_dirs(tmp_path):
    """Test iter_python_files with nested excluded directories."""
    # Create nested structure: src/.venv/something.py should be skipped
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / ".venv").mkdir()
    (tmp_path / "src" / ".venv" / "nested.py").write_text("# skip", encoding="utf-8")

    # Create valid file at same level
    (tmp_path / "src" / "valid.py").write_text("# include", encoding="utf-8")

    files = list(audit.iter_python_files(tmp_path))
    file_names = [f.name for f in files]

    assert "valid.py" in file_names
    assert "nested.py" not in file_names


def test_scan_file_continued_line_with_ignore(tmp_path):
    """Test scan_file with type ignore on a continued line."""
    # Create a file with type: ignore that triggers the continue path
    pfx = "# type:"
    ign = " ignore"
    content = f"""
def func():
    x = 1  {pfx}{ign}[attr-defined]
    return x
"""
    f = tmp_path / "continued.py"
    f.write_text(content, encoding="utf-8")

    # With empty ALLOWED_CODES, this should generate violation
    violations = audit.scan_file(f)
    assert len(violations) == 1
    assert "attr-defined" in violations[0]


def test_iter_python_files_empty_directory(tmp_path):
    """Test iter_python_files with empty directory."""
    files = list(audit.iter_python_files(tmp_path))
    assert files == []


def test_iter_python_files_only_excluded_dirs(tmp_path):
    """Test iter_python_files when only excluded directories exist."""
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("# skip", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("# skip", encoding="utf-8")

    files = list(audit.iter_python_files(tmp_path))
    assert files == []
