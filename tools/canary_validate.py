"""Synthetic canary validation based on CI telemetry."""

from __future__ import annotations

import argparse
import json
import pathlib


def load_metrics(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    return json.loads(path.read_text())


def assert_canary_health(metrics: dict, min_coverage: float = 90.0) -> None:
    # Handle both nested "metrics" key and direct structure
    metrics_data = metrics.get("metrics", metrics)

    # Try to get coverage from different possible locations
    coverage = 0.0
    if "coverage_pct" in metrics_data:
        coverage = float(metrics_data.get("coverage_pct", 0.0))
    elif "coverage" in metrics_data:
        # Handle quality_metrics.json structure with coverage.current
        cov_data = metrics_data.get("coverage", {})
        if isinstance(cov_data, dict):
            coverage = float(cov_data.get("current", 0.0))
        else:
            coverage = float(cov_data)

    flake_score = float(metrics_data.get("flake_recovery_score", 1.0))

    if coverage < min_coverage:
        raise AssertionError(
            f"Canary health gate failed: coverage {coverage:.2f}% < {min_coverage}%"
        )
    if flake_score < 0.8:
        raise AssertionError(
            f"Canary health gate failed: flake recovery score {flake_score:.2f} < 0.8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to run_metrics.json")
    parser.add_argument("--min-coverage", type=float, default=90.0)
    args = parser.parse_args()

    metrics = load_metrics(pathlib.Path(args.metrics))
    assert_canary_health(metrics, args.min_coverage)

    # Extract coverage for display (handle both structures)
    metrics_data = metrics.get("metrics", metrics)
    coverage_pct = 0.0
    if "coverage_pct" in metrics_data:
        coverage_pct = float(metrics_data.get("coverage_pct", 0.0))
    elif "coverage" in metrics_data:
        cov_data = metrics_data.get("coverage", {})
        if isinstance(cov_data, dict):
            coverage_pct = float(cov_data.get("current", 0.0))
        else:
            coverage_pct = float(cov_data)

    flake_score = float(metrics_data.get("flake_recovery_score", 1.0))

    print(
        "✅ Canary validation succeeded:"
        f" coverage={coverage_pct:.2f}%"
        f" flake_score={flake_score:.2f}"
    )


if __name__ == "__main__":
    main()
