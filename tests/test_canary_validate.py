"""Tests for canary validation."""

import json

import pytest

from tools import canary_validate


def test_load_metrics_success(tmp_path):
    """Test loading valid metrics file."""
    metrics_file = tmp_path / "metrics.json"
    metrics_data = {"metrics": {"coverage_pct": 92.5, "flake_recovery_score": 0.9}}
    metrics_file.write_text(json.dumps(metrics_data))

    result = canary_validate.load_metrics(metrics_file)
    assert result == metrics_data


def test_load_metrics_file_not_found(tmp_path):
    """Test loading non-existent metrics file."""
    metrics_file = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError, match="metrics file not found"):
        canary_validate.load_metrics(metrics_file)


def test_assert_canary_health_pass():
    """Test canary health check with good metrics."""
    metrics = {"metrics": {"coverage_pct": 95.0, "flake_recovery_score": 0.85}}

    # Should not raise
    canary_validate.assert_canary_health(metrics, min_coverage=90.0)


def test_assert_canary_health_fail_coverage():
    """Test canary health check fails on low coverage."""
    metrics = {"metrics": {"coverage_pct": 85.0, "flake_recovery_score": 0.9}}

    with pytest.raises(AssertionError, match="coverage 85.00% < 90.0%"):
        canary_validate.assert_canary_health(metrics, min_coverage=90.0)


def test_assert_canary_health_fail_flake_score():
    """Test canary health check fails on low flake score."""
    metrics = {"metrics": {"coverage_pct": 95.0, "flake_recovery_score": 0.5}}

    with pytest.raises(AssertionError, match="flake recovery score 0.50 < 0.8"):
        canary_validate.assert_canary_health(metrics, min_coverage=90.0)


def test_assert_canary_health_missing_metrics():
    """Test canary health check with missing metrics defaults to 0."""
    metrics = {"metrics": {}}

    with pytest.raises(AssertionError, match="coverage 0.00% <"):
        canary_validate.assert_canary_health(metrics, min_coverage=90.0)


def test_main_success(tmp_path, monkeypatch, capsys):
    """Test main function with successful validation."""
    metrics_file = tmp_path / "metrics.json"
    metrics_data = {"metrics": {"coverage_pct": 92.5, "flake_recovery_score": 0.9}}
    metrics_file.write_text(json.dumps(metrics_data))

    monkeypatch.setattr(
        "sys.argv", ["canary_validate", "--metrics", str(metrics_file), "--min-coverage", "90.0"]
    )

    canary_validate.main()

    captured = capsys.readouterr()
    assert "✅ Canary validation succeeded" in captured.out
    assert "coverage=92.50%" in captured.out
    assert "flake_score=0.90" in captured.out


def test_main_failure(tmp_path, monkeypatch):
    """Test main function with failing validation."""
    metrics_file = tmp_path / "metrics.json"
    metrics_data = {"metrics": {"coverage_pct": 80.0, "flake_recovery_score": 0.9}}
    metrics_file.write_text(json.dumps(metrics_data))

    monkeypatch.setattr("sys.argv", ["canary_validate", "--metrics", str(metrics_file)])

    with pytest.raises(AssertionError, match="coverage 80.00% < 90.0%"):
        canary_validate.main()
