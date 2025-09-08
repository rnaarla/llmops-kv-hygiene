# SOC 2 Alignment: KV-Cache Hygiene

Scope

- Components: `tools/cache_tracer.py`, `tools/activation_logger.py`, `tools/eviction_checker.py`, `tools/metrics_exporter.py`, tests under `tests/`, CI/CD under `ci_cd/`, container/K8s manifests.
- Data in scope: transient KV-cache tensors/arrays and associated metadata (tenant_id, request_id, model_id). No payload contents are logged.

Trust Services Criteria (TSC) Mapping

1. Security (Common Criteria)

- Access restriction: forensic logs created with 0600 perms; containers run as non-root (Dockerfiles). HMAC signing of logs via `FORENSIC_HMAC_SECRET` (optional) for tamper evidence.
- Change/alteration prevention: append-only JSONL with SHA-256 hash chaining; rotation emits linkage marker; `ForensicLogger.verify_all()` validates active and rotated files.
- Enforcement: buffers must meet coverage threshold (default 99.9%) before free; violations trigger quarantine with forensic events.
- Secure defaults: TTL and max reuse policies; TTL<=0 triggers immediate quarantine; deterministic verification sampling; optional double-pass scrubbing.
- Hardening/testing: unit/integration tests cover CPU and conditional CUDA paths; CI enforces coverage gates, chain verification, vuln scans.

Evidence

- Implementation: `tools/cache_tracer.py` (ForensicLogger, TTL/reuse/quarantine, sanitization), `tools/activation_logger.py` (anomaly detection), `tools/eviction_checker.py` (gates).
- Integrity tests: `tests/test_kv_isolation.py::test_forensic_rotation_and_chain`, `tests/test_kv_sanitization.py::test_prometheus_export_contains_metrics`.
- CI: `ci_cd/github_actions.yml` (chain verify, coverage >=95%, image scans).

2. Availability

- Observability: metrics via JSON and Prometheus textfile; p50/p95 sanitize duration tracked.
- Health/ops: `metrics_exporter.py` HTTP exporter; K8s `ServiceMonitor` (manifests) for scraping; CronJobs for nightly checks.
- Performance guardrails: eviction checker supports p95 duration threshold.

Evidence

- Metrics: `CacheTracer.export_metrics*`, Prom text lines (HELP/TYPE). Tests validate presence.
- Ops: K8s manifests (Deployment/Service/ServiceMonitor, CronJob) under `k8s/`.

3. Processing Integrity

- Deterministic verification of scrubbing via sampling; optional double pass.
- Strict free gate: freeing without sufficient coverage quarantines and raises.

Evidence

- Tests: `tests/fuzzing_harness.py` (no cross-session leak), `tests/test_kv_sanitization.py::test_free_without_sanitize_quarantines_and_raises`.

4. Confidentiality

- Inter-session isolation via zeroization; TTL/reuse policies reduce residual risk.
- Logs exclude payload data; only IDs and states recorded.
- HMAC and hash chaining protect log integrity/confidentiality (via secrecy of HMAC key).

Evidence

- Code references: `CacheTracer.sanitize()/wait()/_verify_zero()`, `ForensicLogger.append()`.

5. Privacy

- Data minimization: logs avoid PHI/PII payloads by design; only metadata and counters.
- Retention: size-based rotation with linkage; recommended archival/retention policies defined in FedRAMP doc; deletion jobs can be scheduled via K8s CronJob.

Assumptions and Gaps

- Platform controls (disk encryption at rest, NTP, RBAC, TLS) are provided by the hosting environment.
- Optional hardening to enable: readOnlyRootFilesystem, drop Linux caps, mypy/type gates, SBOM attestation, TLS for metrics endpoint.
