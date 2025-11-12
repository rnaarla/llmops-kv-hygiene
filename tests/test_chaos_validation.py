"""Tests for chaos validation."""

import pytest

from tools import chaos_validation


def test_run_chaos_trials(tmp_path, monkeypatch):
    """Test chaos validation runs successfully."""
    # Change to tmp directory to avoid polluting workspace
    monkeypatch.chdir(tmp_path)

    # Run chaos trials
    chaos_validation.run_chaos_trials(trials=4)

    # Verify outputs were created
    assert (tmp_path / "chaos" / "kv_cache.log").exists()
    assert (tmp_path / "chaos" / "metrics.json").exists()


def test_run_chaos_trials_deterministic(tmp_path, monkeypatch):
    """Test chaos validation runs with different trial counts."""
    monkeypatch.chdir(tmp_path)

    # Run with different trial counts
    chaos_validation.run_chaos_trials(trials=2)
    first_log = (tmp_path / "chaos" / "kv_cache.log").read_text()

    # Clean up
    import shutil

    shutil.rmtree(tmp_path / "chaos")

    # Run with more trials
    chaos_validation.run_chaos_trials(trials=8)
    second_log = (tmp_path / "chaos" / "kv_cache.log").read_text()

    # More trials should produce longer log
    assert len(second_log) > len(first_log)


def test_main(tmp_path, monkeypatch, capsys):
    """Test main function."""
    monkeypatch.chdir(tmp_path)

    chaos_validation.main()

    captured = capsys.readouterr()
    assert "✅ Chaos validation completed successfully" in captured.out


def test_run_chaos_trials_forensic_failure(tmp_path, monkeypatch):
    """Test chaos trials with forensic verification failure."""
    from tools.forensic_logger import ForensicLogger

    monkeypatch.chdir(tmp_path)

    # Mock verify_all to return failure
    def mock_verify_fail(path):
        return {"ok": False, "error": "test failure"}

    monkeypatch.setattr(ForensicLogger, "verify_all", mock_verify_fail)

    with pytest.raises(SystemExit, match="Forensic chain verification failed"):
        chaos_validation.run_chaos_trials(trials=2)
