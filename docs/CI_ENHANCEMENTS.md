# CI/CD Enhancements (Current)

This document describes **what the repository CI actually enforces** today. The canonical workflow is `.github/workflows/ci.yml`.

## Pipeline stages

1. **Generate matrix** — Loads `ci_cd/test_matrix.json` for OS and Python versions.
2. **Lint** — `ruff check`, `black --check`, `mypy` on `tools` and `tests`.
3. **Policy guard** — `conftest test` on `.github/workflows/ci.yml`, `codeql.yml`, and `dependency-review.yml` against `policy/` (OPA/Rego: top-level `permissions.contents: read`; only `update-metrics` may use `contents: write` for metrics commits).
4. **Test** — `pytest` with coverage reports under `coverage-reports/` (artifacts).
5. **Coverage aggregate** — Merges coverage JSON and enforces `COVERAGE_THRESHOLD` (default 90%).
6. **Meta self-test** — Validates coverage aggregation behavior.
7. **Chaos validation** — Runs `tools/chaos_validation.py`.
8. **Hygiene gate** — `tools/ci_export_hygiene_metrics.py` plus composite `.github/actions/kv-hygiene-gate` (runs `eviction_checker.py` on exported metrics).
9. **SBOM** — Syft SPDX JSON artifact (`sbom` job).
10. **Security scan** — Trivy filesystem scan (`scanners: vuln,secret`), SARIF upload; optional **`VULN_GATING=true`** secret runs `tools/sarif_vuln_gate.py` on the FS SARIF.
11. **Docker** (main only) — Build/push CPU image to GHCR; Trivy image scan (waits on hygiene gate + SBOM + coverage + security scan).
12. **Docker scan PR** — Build CPU image without push; optional SARIF upload.
13. **Update metrics** (main only) — Updates `ci_metrics/quality_metrics.json` and commits when coverage changes.
14. **Canary validation** (main only) — Pulls image and runs `tools/canary_validate.py`.

Additional workflows: **CodeQL** (`.github/workflows/codeql.yml`), **Dependency Review** (`.github/workflows/dependency-review.yml` on PRs). Python dependencies in CI and the CPU image install from **`requirements.lock`**.

## Hygiene and evidence gates (application-level)

- The default pipeline runs `tools/ci_export_hygiene_metrics.py` and the composite hygiene gate (same thresholds as README: 99.9% min coverage, zero unsanitized/quarantine).
- Forensic log integrity: `ForensicLogger.verify_chain` / `verify_all` and `python -m tools.verify_logs` (see README).

## Vulnerability gating

Repository secret **`VULN_GATING=true`** enables a merge-blocking step on the **filesystem** Trivy SARIF (`tools/sarif_vuln_gate.py`). Image scans remain visibility-oriented unless you add a similar gate.

## Related documentation

- [Product specification](./PRODUCT_SPECIFICATION.md) — scope, interfaces, and quality bar.
- [Implementation status](./IMPLEMENTATION_STATUS.md) — phased delivery progress against the enterprise roadmap.
