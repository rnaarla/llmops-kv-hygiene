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
    coverage = float(metrics["metrics"].get("coverage_pct", 0.0))
    flake_score = float(metrics["metrics"].get("flake_recovery_score", 1.0))
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
    coverage_pct = float(metrics["metrics"].get("coverage_pct", 0.0))
    flake_score = float(metrics["metrics"].get("flake_recovery_score", 1.0))
    print(
        "✅ Canary validation succeeded:"
        f" coverage={coverage_pct:.2f}%"
        f" flake_score={flake_score:.2f}"
    )


if __name__ == "__main__":
    main()
