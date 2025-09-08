"""
Eviction Checker: validates hygiene and eviction thresholds for CI/CD gates.

Consumes metrics exported by CacheTracer (coverage, unsanitized regions, quarantine count)
and produces a machine-readable verdict. Intended to be run in CI to block merges/deploys
if thresholds are not met.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
from pathlib import Path
import sys
import time


@dataclass
class Thresholds:
    coverage_pct_min: float = 99.9
    unsanitized_regions_max: int = 0
    quarantine_count_max: int = 0
    reuse_rate_max: Optional[float] = None  # optional policy bound
    sanitize_p95_ms_max: Optional[float] = None


def check_metrics(metrics: Dict[str, Any], thresholds: Optional[Thresholds] = None) -> Dict[str, Any]:
    th = thresholds or Thresholds()
    coverage = float(metrics.get("min_coverage_pct", 0.0))
    unsanitized = int(metrics.get("unsanitized_regions_count", 0))
    quarantine = int(metrics.get("quarantine_count", 0))
    reuse_rate = float(metrics.get("reuse_rate", 0.0))
    p95 = float(metrics.get("sanitize_duration_p95_ms", 0.0))

    failures = []
    if coverage < th.coverage_pct_min:
        failures.append({"metric": "coverage_pct", "actual": coverage, "expected_min": th.coverage_pct_min})
    if unsanitized > th.unsanitized_regions_max:
        failures.append({"metric": "unsanitized_regions_count", "actual": unsanitized, "expected_max": th.unsanitized_regions_max})
    if quarantine > th.quarantine_count_max:
        failures.append({"metric": "quarantine_count", "actual": quarantine, "expected_max": th.quarantine_count_max})
    if th.reuse_rate_max is not None and reuse_rate > th.reuse_rate_max:
        failures.append({"metric": "reuse_rate", "actual": reuse_rate, "expected_max": th.reuse_rate_max})
    if th.sanitize_p95_ms_max is not None and p95 > th.sanitize_p95_ms_max:
        failures.append({"metric": "sanitize_duration_p95_ms", "actual": p95, "expected_max": th.sanitize_p95_ms_max})

    verdict = {
        "pass": len(failures) == 0,
        "failures": failures,
        "metrics": metrics,
        "thresholds": {k: v for k, v in th.__dict__.items()},
        "ts": time.time(),
        "schema": 1,
    }
    return verdict


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Eviction / Hygiene Threshold Checker")
    parser.add_argument("metrics", type=str, help="Path to coverage metrics JSON exported by CacheTracer")
    parser.add_argument("--coverage-min", type=float, default=99.9)
    parser.add_argument("--unsanitized-max", type=int, default=0)
    parser.add_argument("--quarantine-max", type=int, default=0)
    parser.add_argument("--reuse-rate-max", type=float, default=None)
    parser.add_argument("--sanitize-p95-ms-max", type=float, default=None)
    parser.add_argument("--out", type=str, default="forensics/verdict.json")
    args = parser.parse_args(argv)

    with Path(args.metrics).open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    verdict = check_metrics(
        metrics,
        Thresholds(
            coverage_pct_min=args.coverage_min,
            unsanitized_regions_max=args.unsanitized_max,
            quarantine_count_max=args.quarantine_max,
            reuse_rate_max=args.reuse_rate_max,
            sanitize_p95_ms_max=args.sanitize_p95_ms_max,
        ),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print(json.dumps(verdict))
    return 0 if verdict["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
