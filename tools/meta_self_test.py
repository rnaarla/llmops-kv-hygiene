"""Meta self test for coverage aggregation determinism."""

from __future__ import annotations

import argparse
import json
import pathlib


def aggregate_coverage(artifacts_dir: pathlib.Path) -> float:
    total_covered = 0
    total_statements = 0
    for cov_file in artifacts_dir.rglob("coverage.json"):
        try:
            data = json.loads(cov_file.read_text())
        except json.JSONDecodeError:
            continue
        totals = data.get("totals") or {}
        total_covered += int(totals.get("covered_lines", 0))
        total_statements += int(totals.get("num_statements", 0))
    if not total_statements:
        return 0.0
    return (total_covered / total_statements) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True, help="Path to coverage_aggregate.json")
    parser.add_argument("--artifacts", required=True, help="Directory containing coverage shards")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Allowed pct drift")
    args = parser.parse_args()

    aggregate_path = pathlib.Path(args.aggregate)
    artifacts_dir = pathlib.Path(args.artifacts)

    if not aggregate_path.exists():
        print("⚠️  coverage aggregate artifact not available; skipping meta validation.")
        return

    target = json.loads(aggregate_path.read_text()).get("coverage")
    if target is None:
        print("⚠️  coverage aggregate missing numeric coverage; skipping meta validation.")
        return

    recomputed = aggregate_coverage(artifacts_dir)
    delta = abs(recomputed - float(target))
    print(f"Meta self test: stored={target:.2f}% recomputed={recomputed:.2f}% Δ={delta:.4f}%")
    if delta > args.tolerance:
        msg = (
            "Coverage aggregation not idempotent. "
            f"Delta {delta:.4f}% exceeds tolerance {args.tolerance}%"
        )
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
