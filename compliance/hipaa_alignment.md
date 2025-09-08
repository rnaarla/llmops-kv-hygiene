# HIPAA Alignment: KV-Cache Hygiene and PHI Protection

Scope

- PHI risk surface: transient KV-cache memory used during inference (tensors/arrays), and operational logs/metrics.
- Objective: prevent residual data exposure across sessions/tenants and provide auditable evidence of sanitation.

Safeguards Mapping

1. Administrative Safeguards

- Policies/Procedures: eviction thresholds and quarantine policies encoded in `eviction_checker.py` and CI gates in `ci_cd/github_actions.yml`.
- Workforce training implication: runbooks reference metrics and forensic verification outputs (`ForensicLogger.verify_all`).

Evidence

- CI verdict artifacts (`forensics/verdict.json`), coverage JSON, Prometheus text files.

1. Physical Safeguards

- Out of project scope; assumed provided by hosting/DC (facility access, hardware disposal). Containers run as non-root per Dockerfiles.

1. Technical Safeguards

- Access control: forensic logs created with 0600 permissions; HMAC signing optional via `FORENSIC_HMAC_SECRET` to detect tampering.
- Integrity: append-only JSONL with SHA-256 hash chaining; rotation markers link files; `verify_all()` checks across rotations.
- Transmission security: metrics and logs can be exposed internally; recommend TLS termination at ingress/service mesh.
- Person/entity authentication: provided by platform (K8s RBAC/Secrets, CI secrets).
- Audit controls: detailed forensic events (allocate, bind, write, sanitize_start/end, reuse, quarantine, free) with chain of custody.
- Data minimization: no payload contents or PHI are logged; only metadata IDs and counters.

PHI Protection in KV Cache

- Zeroization: explicit scrubbing of buffers upon sanitize with deterministic verification sampling; optional double pass.
- Isolation: TTL and max-reuse policies; TTL<=0 immediately quarantines to avoid stale retention.
- Enforcement: freeing requires coverage >= threshold; otherwise quarantine and exception.
- GPU semantics: CUDA stream/event fidelity ensures scrubbing aligns with execution order (prevents races/leaks).

Evidence

- Code: `tools/cache_tracer.py` (`sanitize`, `_verify_zero`, quarantine, hash-chained logs).
- Tests: `tests/fuzzing_harness.py` (no cross-session leak), `tests/test_kv_sanitization.py` and `tests/test_kv_isolation.py`.
- Ops: K8s CronJob for periodic eviction checks; metrics exporter for monitoring.

Retention and Disposal

- In-memory TTL and reuse bounds reduce retention risk.
- Forensic logs: size-based rotation with hash-linked continuity. Recommend environment-specific retention period and secure archival; deletion via scheduled CronJob.

Assumptions and Gaps

- At-rest encryption and node disk protection are platform responsibilities.
- TLS for metrics endpoints and secure secret management (HMAC key) must be configured by ops.
- Business Associate Agreement (BAA) and breach notification workflows are organizational responsibilities.
