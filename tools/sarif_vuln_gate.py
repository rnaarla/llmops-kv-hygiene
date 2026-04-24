"""Fail CI when a Trivy (or compatible) SARIF report contains blocking findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def count_blocking_results(doc: dict[str, Any]) -> tuple[int, list[str]]:
    """Count SARIF results treated as merge-blocking (error / warning levels)."""
    blocking: list[str] = []
    for run in doc.get("runs") or []:
        for idx, result in enumerate(run.get("results") or []):
            level = (result.get("level") or "").lower()
            rule_id = result.get("ruleId", f"result[{idx}]")
            if level in ("error", "warning"):
                blocking.append(f"{rule_id} ({level or 'unspecified'})")
            elif not level:
                blocking.append(f"{rule_id} (no level)")
    return len(blocking), blocking


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Exit non-zero if SARIF contains blocking results")
    p.add_argument("sarif", type=str, help="Path to SARIF JSON file")
    args = p.parse_args(argv)

    path = Path(args.sarif)
    if not path.is_file():
        print(f"sarif_vuln_gate: missing file {path}", file=sys.stderr)
        return 1

    doc = json.loads(path.read_text(encoding="utf-8"))
    n, items = count_blocking_results(doc)
    if n:
        print(f"sarif_vuln_gate: {n} blocking finding(s) in {path}:", file=sys.stderr)
        for line in items[:50]:
            print(f"  - {line}", file=sys.stderr)
        if n > 50:
            print(f"  ... and {n - 50} more", file=sys.stderr)
        return 1
    print(f"sarif_vuln_gate: no blocking results in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
