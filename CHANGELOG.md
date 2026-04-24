# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `ForensicLogger`: strict HMAC verification when a verifier supplies an HMAC key — every log line must include a valid `hmac` field (`verify_chain` / `verify_all` with key configured).
- `docs/PRODUCT_SPECIFICATION.md`, `docs/IMPLEMENTATION_STATUS.md`, `docs/POST_IMPLEMENTATION_REVIEW.md` for enterprise delivery traceability.

### Changed

- `CacheTracer._verify_zero`: NumPy buffers now use `sanitizer.verify_zero` sampling instead of an unconditional trust path.
- **CI/CD**: default GitHub Actions `permissions.contents: read`; grant `contents: write` only on the `update-metrics` job; remove AI triage and autonomous-fix jobs; simplify matrix job (remove unused test-selection outputs); Conftest policy updated for OPA v1.
- **Documentation**: README accuracy pass; `docs/CI_ENHANCEMENTS.md` rewritten to match the real workflow; removed obsolete AI-only guides.

### Removed

- Duplicate root `CI_ENHANCEMENTS.md` and obsolete `docs/AI_*.md` / `docs/NEW_WORKFLOW_SUMMARY.md` (content superseded by accurate CI and product docs).

### Fixed

- `docker-compose.yaml`: removed references to a non-existent `cache_tracer.py --demo` entrypoint.
