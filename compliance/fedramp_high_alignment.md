# FedRAMP High Alignment: Log Retention and Auditability

Scope

- Artifacts: KV-cache forensic logs, coverage metrics JSON, Prometheus text files, CI verdicts.
- Objectives: tamper-evident audit trail, controlled retention/archival, and regular verification.

Audit and Accountability (AU)

1. AU-2 Event Logging

- Events logged: allocate, bind, write, reuse, sanitize_start/end, quarantine, free, rotate.
- Content: timestamp, handle, tenant/request/model IDs, device/shape/dtype, prev/curr hash, optional HMAC.
- No payload data captured.

1. AU-3 Content of Audit Records

- Canonicalized JSON with schema version; prev_hash and curr_hash provide linkage.
- Optional HMAC via `FORENSIC_HMAC_SECRET` strengthens integrity assurances.

1. AU-6 Audit Review, Analysis, and Reporting

- `ForensicLogger.verify_chain` for per-file validation; `verify_all` for rotated chain including linkage checks.
- CI job includes automated verification and publishes artifacts; operators can run periodic reviews via CronJobs.

1. AU-9 Protection of Audit Information

- File permissions set to 0600 at creation; containers run as non-root.
- Append-only write path; rotation implemented by rename + explicit rotate marker, preventing in-place modifications.

1. AU-11 Audit Record Retention

- Size-based rotation out of the box; recommend environment policy for time-based retention and archival (e.g., S3 with bucket lifecycle).
- Provide deletion/archival via scheduled K8s CronJob; maintain chain-of-custody by preserving rotate linkage metadata.

System and Communications Protection (SC)

1. SC-13 Cryptographic Protection

- HMAC signing option for audit lines; key managed via K8s Secret/CI secret. Rotate keys via secret rotation process.

1. SC-28 Protection of Information at Rest

- Project assumes platform disk encryption (node/volume); forensic logs are metadata-only and permission-restricted.

Configuration and Operations

- Time sync: rely on platform NTP; timestamps are UNIX epoch seconds.
- Metrics: Prometheus textfile export; ServiceMonitor for scraping; p50/p95 latency for performance monitoring.
- Jobs: `k8s/cronjob-eviction-checker.yaml` runs regular metrics checks; add a `verify-logs` CronJob to run `ForensicLogger.verify_all` and alert on failures.

Evidence

- Source: `tools/cache_tracer.py` (ForensicLogger, rotation), `tools/eviction_checker.py`, `tools/metrics_exporter.py`.
- Tests: rotation and chain verification; metrics presence; TTL/reuse policies.
- CI: `ci_cd/github_actions.yml` includes verification step and vulnerability scanning.

Assumptions and Gaps

- Centralized log management (e.g., CloudWatch, Splunk, SIEM) integration is environment-specific.
- Access control, KMS, and encryption at rest/in transit are provided by the platform.
- Consider adding WORM storage or object lock for immutable archives where required.
