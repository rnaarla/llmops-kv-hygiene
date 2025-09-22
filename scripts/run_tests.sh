#!/usr/bin/env bash
set -euo pipefail

# Colors
Y="\033[33m"; G="\033[32m"; R="\033[31m"; B="\033[36m"; X="\033[0m"

info(){ echo -e "${B}[INFO]${X} $*"; }
warn(){ echo -e "${Y}[WARN]${X} $*"; }
err(){ echo -e "${R}[ERR ]${X} $*"; }
success(){ echo -e "${G}[ OK ]${X} $*"; }

# Determine python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  err "Python interpreter not found. Install Python 3.11+ first."
  exit 127
fi

REQ_FILE="requirements-dev.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  warn "Missing $REQ_FILE; creating minimal one."
  echo -e "numpy\npytest\npytest-cov" > "$REQ_FILE"
fi

# Create venv
if [[ ! -d .venv ]]; then
  info "Creating virtual environment (.venv)"
  $PY -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Upgrading pip"
python -m pip install --upgrade pip >/dev/null

info "Installing dev requirements"
pip install -r "$REQ_FILE" >/dev/null

# Optional torch (best-effort) if not present; do not fail build
if ! python -c "import torch" 2>/dev/null; then
  warn "PyTorch not installed; GPU/torch branches will be skipped. (Attempting CPU install)"
  if ! pip install torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1; then
    warn "PyTorch CPU wheel install failed; continuing without it."
  fi
fi

info "Running tests with coverage (target 90%)"
set +e
# Chosen 87%: empirical plateau after excluding low-value defensive/GPU branches.
# Future refactor can raise this again once tracer is decomposed.
pytest -q --cov=tools --cov-report=term-missing --cov-fail-under=90
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  err "Tests or coverage gate failed (exit code $RC)."
  info "Re-run verbosely: source .venv/bin/activate && pytest -vv --cov=tools --cov-report=term-missing"
  exit $RC
fi

success "All tests passed with required coverage."

# Provide HTML report optionally
if [[ -d htmlcov ]]; then
  info "Open HTML coverage report: open htmlcov/index.html"
fi
