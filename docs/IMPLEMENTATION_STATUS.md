# Implementation status (90-day enterprise plan)

This file tracks **delivered** work against the phased roadmap. It is updated when a phase slice ships to `main`.

## Completed in this delivery slice

### Week 1–2 (Phase 0–1 foundations)

- **Forensic strict HMAC**: `ForensicLogger.verify_chain` requires an `hmac` field on every line when verification is performed with an HMAC key (fail closed on missing or wrong MAC).
- **NumPy verification**: `CacheTracer._verify_zero` uses `sanitizer.verify_zero` for NumPy arrays so attestation is not skipped on CPU.
- **CI economics and security**: Removed OpenAI-based triage/auto-fix jobs (no recurring token spend, no outbound secret-bearing CI calls). Removed unused “test selection” matrix plumbing that produced no consumer.
- **Least privilege**: Workflow default `permissions.contents: read`; `update-metrics` alone uses `contents: write` for metrics commits.
- **Policy as code**: Rewrote `policy/ci.rego` for OPA/Conftest v1 (`package main`, `deny contains … if`) so `policy-guard` is meaningful.
- **Repository hygiene**: Removed duplicate and obsolete doc files; aligned `docker-compose` commands with reality; expanded `.gitignore` for CI/local artifacts.

### Week 3+ (not yet implemented here)

- Runtime adapters (vLLM, TensorRT-LLM), immutable WORM log sink, GPU CI lane, SLSA provenance, and full async CUDA scrub remain **future** milestones. Track them in your program plan; they are out of scope for the current merge slice.

## How to update this document

After each merged milestone, append a dated subsection under the appropriate week band with: **scope**, **PR / commit**, **tests run**, and **known limitations**.
