"""Tests for meta self-test."""

import json

import pytest

from tools import meta_self_test


def test_aggregate_coverage_empty_dir(tmp_path):
    """Test aggregation with no coverage files."""
    result = meta_self_test.aggregate_coverage(tmp_path)
    assert result == 0.0


def test_aggregate_coverage_single_file(tmp_path):
    """Test aggregation with single coverage file."""
    cov_file = tmp_path / "coverage.json"
    cov_data = {"totals": {"covered_lines": 80, "num_statements": 100}}
    cov_file.write_text(json.dumps(cov_data))

    result = meta_self_test.aggregate_coverage(tmp_path)
    assert result == 80.0


def test_aggregate_coverage_multiple_files(tmp_path):
    """Test aggregation with multiple coverage files."""
    # Create nested structure
    (tmp_path / "job1").mkdir()
    (tmp_path / "job2").mkdir()

    cov1 = tmp_path / "job1" / "coverage.json"
    cov1.write_text(json.dumps({"totals": {"covered_lines": 80, "num_statements": 100}}))

    cov2 = tmp_path / "job2" / "coverage.json"
    cov2.write_text(json.dumps({"totals": {"covered_lines": 90, "num_statements": 100}}))

    result = meta_self_test.aggregate_coverage(tmp_path)
    # Total: 170/200 = 85%
    assert result == 85.0


def test_aggregate_coverage_invalid_json(tmp_path):
    """Test aggregation skips invalid JSON files."""
    # Create subdirectory to match rglob pattern
    subdir = tmp_path / "job1"
    subdir.mkdir()

    valid = subdir / "coverage.json"
    valid.write_text(json.dumps({"totals": {"covered_lines": 80, "num_statements": 100}}))

    invalid = subdir / "invalid.json"
    invalid.write_text("not json{{{")

    result = meta_self_test.aggregate_coverage(tmp_path)
    assert result == 80.0  # Only valid file counted


def test_aggregate_coverage_missing_totals(tmp_path):
    """Test aggregation handles missing totals gracefully."""
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps({"meta": "data"}))

    result = meta_self_test.aggregate_coverage(tmp_path)
    assert result == 0.0


def test_main_success(tmp_path, monkeypatch, capsys):
    """Test main function with matching coverage."""
    # Create aggregate file
    aggregate_file = tmp_path / "aggregate.json"
    aggregate_file.write_text(json.dumps({"coverage": 85.0}))

    # Create artifacts directory with matching coverage
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    cov_file = artifacts_dir / "coverage.json"
    cov_file.write_text(json.dumps({"totals": {"covered_lines": 85, "num_statements": 100}}))

    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_self_test",
            "--aggregate",
            str(aggregate_file),
            "--artifacts",
            str(artifacts_dir),
            "--tolerance",
            "0.05",
        ],
    )

    meta_self_test.main()

    captured = capsys.readouterr()
    assert "stored=85.00%" in captured.out
    assert "recomputed=85.00%" in captured.out


def test_main_tolerance_exceeded(tmp_path, monkeypatch):
    """Test main function fails when tolerance exceeded."""
    # Create aggregate file with one value
    aggregate_file = tmp_path / "aggregate.json"
    aggregate_file.write_text(json.dumps({"coverage": 85.0}))

    # Create artifacts with different coverage
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    cov_file = artifacts_dir / "coverage.json"
    cov_file.write_text(json.dumps({"totals": {"covered_lines": 90, "num_statements": 100}}))

    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_self_test",
            "--aggregate",
            str(aggregate_file),
            "--artifacts",
            str(artifacts_dir),
            "--tolerance",
            "1.0",  # 5% delta is within 1.0% tolerance
        ],
    )

    with pytest.raises(SystemExit, match="Coverage aggregation not idempotent"):
        meta_self_test.main()


def test_main_missing_aggregate(tmp_path, monkeypatch, capsys):
    """Test main skips when aggregate file missing."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_self_test",
            "--aggregate",
            str(tmp_path / "missing.json"),
            "--artifacts",
            str(artifacts_dir),
        ],
    )

    meta_self_test.main()

    captured = capsys.readouterr()
    assert "coverage aggregate artifact not available" in captured.out


def test_main_missing_coverage_field(tmp_path, monkeypatch, capsys):
    """Test main skips when coverage field missing."""
    aggregate_file = tmp_path / "aggregate.json"
    aggregate_file.write_text(json.dumps({"other": "data"}))

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        ["meta_self_test", "--aggregate", str(aggregate_file), "--artifacts", str(artifacts_dir)],
    )

    meta_self_test.main()

    captured = capsys.readouterr()
    assert "missing numeric coverage" in captured.out
