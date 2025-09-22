"""Audit disciplined use of ``# type: ignore``.

Policy (allowed patterns only):
  * Targeted error codes explicitly allowlisted via ``ALLOWED_CODES``.
  * No bare directives without a code.
  * Disallow general ``assignment`` ignores (legacy optional dependency pattern).

Run: ``python -m tools.audit_type_ignores``. Returns exit code 1 on violations.
Implementation uses inexpensive line scanning (not full mypy parsing).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

RE_BARE = re.compile(r"#\s*type:\s*ignore(\s|$)(?!\[)")
RE_ASSIGNMENT = re.compile(r"#\s*type:\s*ignore\[assignment\]")

# Allowed mypy ignore codes (strings) explicitly justified.
ALLOWED_CODES: set[str] = set()
# Extend as genuinely needed, e.g.: ALLOWED_CODES.update({'attr-defined'})

RE_ALLOWED = re.compile(r"#\s*type:\s*ignore\[([a-z0-9_-]+)\]")


def iter_python_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Skip virtual envs or build dirs if any appear later
        if any(part in {".venv", "build", "dist"} for part in p.parts):
            continue
        yield p


def scan_file(file: Path) -> list[str]:
    violations: list[str] = []
    text = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    for lineno, line in enumerate(text, 1):
        if "# type: ignore" not in line:
            continue
        if RE_BARE.search(line):
            violations.append(f"{file}:{lineno}: bare '# type: ignore' not allowed")
            continue
        m = RE_ALLOWED.search(line)
        if m:
            code = m.group(1)
            if code not in ALLOWED_CODES:
                violations.append(
                    f"{file}:{lineno}: code '{code}' not in ALLOWED_CODES "
                    "(explicitly justify or remove)"
                )
            continue
        if RE_ASSIGNMENT.search(line):
            violations.append(
                f"{file}:{lineno}: 'type: ignore[assignment]' discouraged; "
                "refactor optional dependency handling"
            )
    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    all_violations: list[str] = []
    for file in iter_python_files(root):
        all_violations.extend(scan_file(file))
    if all_violations:
        print("Type ignore policy violations detected:")
        for v in all_violations:
            print("  -", v)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - trivial CLI wrapper
    sys.exit(main())
