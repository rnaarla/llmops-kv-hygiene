# CI/CD Enhancements (Current)

This document describes **what the repository CI actually enforces** today. The canonical workflow is `.github/workflows/ci.yml`.

## Pipeline stages

1. **Generate matrix** — Loads `ci_cd/test_matrix.json` for OS and Python versions.
2. **Lint** — `ruff check`, `black --check`, `mypy` on `tools` and `tests`.
3. **Policy guard** — `conftest test .github/workflows/ci.yml -p policy` (OPA/Rego: top-level `permissions.contents: read`; only `update-metrics` may use `contents: write` for metrics commits).
4. **Test** — `pytest` with coverage reports under `coverage-reports/` (artifacts).
5. **Coverage aggregate** — Merges coverage JSON and enforces `COVERAGE_THRESHOLD` (default 90%).
6. **Meta self-test** — Validates coverage aggregation behavior.
7. **Chaos validation** — Runs `tools/chaos_validation.py`.
8. **Security scan** — Trivy filesystem SARIF upload.
9. **Docker** (main only) — Build/push CPU image to GHCR; Trivy image scan.
10. **Docker scan PR** — Build CPU image without push; optional SARIF upload.
11. **Update metrics** (main only) — Updates `ci_metrics/quality_metrics.json` and commits when coverage changes.
12. **Canary validation** (main only) — Pulls image and runs `tools/canary_validate.py`.

## Hygiene and evidence gates (application-level)

- Export metrics with `CacheTracer.export_metrics()` and gate with `tools/eviction_checker.py` in your own deploy pipeline or additional workflow steps if required.
- Forensic log integrity: `ForensicLogger.verify_chain` / `verify_all` and `python -m tools.verify_logs` (see README).

## Vulnerability gating

Optional image gating via repository secret `VULN_GATING` is described in the main README feature matrix. Trivy SARIF is produced for visibility; gating policy is organization-specific.

## Related documentation

- [Product specification](./PRODUCT_SPECIFICATION.md) — scope, interfaces, and quality bar.
- [Implementation status](./IMPLEMENTATION_STATUS.md) — phased delivery progress against the enterprise roadmap.
