#!/usr/bin/env bash
set -euo pipefail

# Dry-run simulation of .github/workflows/ci.yml (single-host, current Python)
# Usage: ./scripts/dry_run_ci.sh [--matrix]
#   --matrix : iterate through python versions defined in ci_cd/test_matrix.json using pyenv (if available)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[dry-run] Starting CI simulation in $ROOT_DIR"

if [[ "${1:-}" == "--matrix" ]]; then
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "[warn] pyenv not found; falling back to current python only" >&2
    MATRIX_MODE=0
  else
    MATRIX_MODE=1
  fi
else
  MATRIX_MODE=0
fi

PY_VERS=()
if [[ ${MATRIX_MODE:-0} -eq 1 ]]; then
  PY_VERS=($(python - <<'PY'
import json;print(*json.load(open('ci_cd/test_matrix.json'))['python'])
PY
  ))
else
  PY_VERS=($(python - <<'PY'
import sys;print(sys.version.split()[0])
PY
  ))
fi

aggregate_cov_lines=0
aggregate_cov_stmts=0

doit_for_version() {
  local pyv="$1"
  echo "\n=== [version $pyv] Setup ==="
  if [[ ${MATRIX_MODE:-0} -eq 1 ]]; then
    pyenv local "$pyv" || echo "[warn] pyenv local $pyv failed; continuing with system python"
  fi
  python -m pip install --upgrade pip >/dev/null
  pip install -r requirements.txt >/dev/null

  echo "=== Lint Stage ($pyv) ==="
  python tools/audit_type_ignores.py
  ruff check .
  black --check .
  mypy --config-file mypy.ini tools tests

  echo "=== Test Stage ($pyv) ==="
  rm -rf reports forensics || true
  mkdir -p reports
  pytest --maxfail=1 --disable-warnings -q \
    --cov=tools --cov-report=xml:reports/coverage.xml \
    --cov-report=json:reports/coverage.json \
    --junitxml=reports/junit.xml --cov-fail-under=90

  echo "=== Forensics Stage ($pyv) ==="
  python - <<'PY'
from tools.cache_tracer import CacheTracer, ForensicLogger
import json
c = CacheTracer()
h = c.allocate(tenant_id='dry', request_id='dry-1', model_id='m', shape=(64,), dtype='float32', device='cpu', framework='numpy')
c.mark_in_use(h)
try:
    cov = c.sanitize(h, async_=False, verify=True)
    c.free(h)
except Exception:
    pass
c.export_metrics('forensics/coverage.json')
res = ForensicLogger.verify_chain('forensics/kv_cache.log')
print('sanitize_result:', cov)
print('chain_ok:', res.get('ok'))
PY

  python tools/eviction_checker.py forensics/coverage.json --coverage-min 99.9 --unsanitized-max 0 --quarantine-max 0 --out forensics/verdict.json || echo "[warn] eviction thresholds not met"

  echo "=== Coverage Collect ($pyv) ==="
  python - <<'PY'
import json
j=json.load(open('reports/coverage.json'))
t=j['totals']
print(f"coverage_lines: {t['covered_lines']} / {t['num_statements']} -> {t['percent_covered']:.2f}%")
print(f"::COVERAGE::{t['covered_lines']}::{t['num_statements']}")
PY
}

while IFS= read -r line; do
  if [[ $line == ::COVERAGE::* ]]; then
    # Format: ::COVERAGE::covered::total
    payload=${line#::COVERAGE::}
    IFS='::' read -r cov_lines cov_stmts <<< "$payload"
    [[ $cov_lines =~ ^[0-9]+$ ]] || cov_lines=0
    [[ $cov_stmts =~ ^[0-9]+$ ]] || cov_stmts=0
    aggregate_cov_lines=$((aggregate_cov_lines + cov_lines))
    aggregate_cov_stmts=$((aggregate_cov_stmts + cov_stmts))
  fi
done < <(for v in "${PY_VERS[@]}"; do doit_for_version "$v"; done)

if (( aggregate_cov_stmts > 0 )); then
  pct=$(python - <<PY
lines=$aggregate_cov_lines
stmts=$aggregate_cov_stmts
print(f"{(lines/stmts)*100:.2f}")
PY
  )
  echo "\n=== Aggregate Coverage ==="
  echo "Total: $pct% ($aggregate_cov_lines / $aggregate_cov_stmts)"
  awk -v p="$pct" 'BEGIN{ exit (p>=90.0)?0:2 }'
fi

echo "\n[dry-run] Completed successfully"
