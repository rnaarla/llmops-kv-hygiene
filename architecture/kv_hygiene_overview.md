# KV-Cache Hygiene: Isolation, Sanitization, and Forensic Logging

This document describes an end-to-end approach to prevent stale context injection and cross-tenant leakage in Large Language Model (LLM) serving by enforcing strict KV-cache hygiene. It covers the threat model, objectives, architecture, isolation model, sanitization algorithms, forensic logging, performance expectations, and integration guidance.

## 1) Problem and Threat Model

- Context: Decoder-only models maintain per-token keys/values (K/V) in GPU memory (HBM) and sometimes in CPU pinned memory. High-throughput serving reuses buffers across requests and tenants to save latency/cost.
- Primary risk: Stale context injection — a later request observes K/V residues from an earlier request, enabling prompt or activation leakage across users/tenants. Variants include:
  - Cross-tenant KV reuse (allocator pooling without strict fences)
  - Partial sanitization (holes during fragmentation or strided layouts)
  - Async race (memset scheduled but not awaited before next allocation)
  - CPU pinned-memory spill not scrubbed when tensors are sharded/paged
- Impact: Confidentiality breach (PHI, PII, trade secrets), integrity (model behaviors influenced by foreign context), compliance violations (SOC2, HIPAA, FedRAMP), and auditability gaps.

## 2) Objectives and SLAs

- Isolation: Per-request unique allocation; no reuse across tenants. Within-tenant reuse only when sanitizer attests 99.9%+ coverage.
- Sanitization: Deterministic, verifiable scrub before reuse/free. Coverage objective: ≥ 99.9% across all tracked regions, per batch and per stream.
- Forensic logging: Append-only, tamper-evident chain with retention aligned to FedRAMP High. Zero data loss objective for lifecycle events.
- Performance: End-to-end overhead budget ≤ 2–5% latency p50/p95 with amortized scrub; sublinear throughput impact via stream-overlapped memset.
- Safety: Fail-secure defaults — on uncertainty, quarantine and disable reuse.

## 3) Architecture (components in this repo)

- cache_tracer (tools/cache_tracer.py)
  - Tags and traces KV allocations with {tenant_id, request_id, model_id, device, ptr/range, shape, dtype, stream_id, ts}.
  - Emits lifecycle events: allocate → tag → bind → scrub_start → scrub_end → free.
  - Aggregates coverage metrics and exposes a gauge for CI/CD “hygiene” checks.
- eviction_checker (tools/eviction_checker.py)
  - Validates policy thresholds (e.g., ≥ 99.9% sanitization coverage, bounded reuse TTL, max live un-sanitized regions = 0).
  - Produces a machine-readable verdict consumed by CI/CD gates.
- activation_logger (tools/activation_logger.py)
  - Hooks into inference to log rare or unexpected activations indicative of stale context.
  - Supports tenant-aware sampling and anomaly flags.
- Tests (tests/*.py)
  - Isolation: unique per-request allocation and enforced flush.
  - Sanitization: GPU/CPU scrubbing, async ordering, fragmentation edges (tests simulate cudaMemsetAsync semantics in PyTorch).
  - Fuzzing harness: multi-tenant inputs probing leakage and logging anomalies.
- CI/CD (ci_cd/*)
  - GitHub Actions + Jenkins pipelines run unit + fuzz + coverage, then enforce hygiene thresholds and policy-as-code before deploy.

## 4) Isolation Model

- Namespacing and fencing
  - No buffer is shared across tenants. Tag each allocation with tenant_id and request_id.
  - Reuse within the same request only; reuse across requests requires sanitizer attest and policy approval.
  - Enforce hard fences between tenants with allocator pools or virtualized contexts; deny admission if an isolated pool cannot be guaranteed.
- Allocation policy
  - allocate(tag) → exclusive handle; bind(handle, request_id) → in-use; flush(handle) → scrub; free(handle) → return to pool.
  - Optional quarantine ring for buffers that fail scrub or validation.
- Ordering and concurrency
  - All scrubs are enqueued on the same CUDA stream as the last writer or synchronized via events; next reader must await scrub_end event.
  - CPU pinned buffers scrubbed synchronously before reuse; no speculative mapping without scrub attest.

## 5) Sanitization Algorithms

- GPU memory zeroization
  - Zero fill K/V regions using memset-like kernels (e.g., cudaMemsetAsync) or tensor ops (tensor.zero_()).
  - Verify coverage by sampling canary pages and by byte-count accounting from the tracer. Optionally double-pass (0x00 then 0xFF→0x00) for high-assurance.
- CPU pinned memory
  - Overwrite pages used for KV staging; use mlock/munlock aware scrubs when applicable.
- Coverage attest
  - The tracer records total planned bytes vs. scrubbed bytes; coverage = scrubbed/total.
  - Require coverage ≥ 99.9% and no gaps (no un-scrubbed spans). Gaps trigger quarantine and incident alerting.
- Idempotence and determinism
  - Scrubs are idempotent; repeating scrub cannot reduce coverage. Scrub failures escalate to fail-secure path.

## 6) Forensic Logging and Audit

- Log channel: Append-only file(s) or stream with WORM semantics; tamper-evident via hash chaining (prev_hash → curr_hash over canonicalized record).
- Record schema (example)
  - trace_id, ts_open, ts_close, tenant_id, request_id, model_id
  - device, stream_id, ptr_start, nbytes, shape, dtype
  - event_type ∈ {allocate, bind, sanitize_start, sanitize_end, free, quarantine}
  - coverage_pct, duration_ms, verdict ∈ {pass, fail}
  - prev_hash, curr_hash
- Retention and access control
  - Retention per FedRAMP High. Restricted operators; read-only replication to audit sink.
- Uses
  - Post-incident forensics, compliance audits, and real-time SLO dashboards.

## 7) Performance and Safety

- Overlap scrubs with I/O and compute when safe (stream semantics). Batch scrubbing to minimize launches.
- Maintain budgets and report p50/p95 overhead deltas. Never trade coverage for latency.
- On failure to meet thresholds, automatically:
  - Quarantine affected buffers and disable KV reuse for impacted pools
  - Raise alert, annotate requests, and degrade gracefully (e.g., recompute without cache)

## 8) Integration Guide

- Integration points for popular stacks (to be wired in examples/tests):
  - PyTorch: Hook tensor lifecycles; use stream-aware zero_() and CUDA events; avoid relying on empty_cache for sanitization (it does not zero memory).
  - TensorRT-LLM / vLLM: Wrap KV allocator abstraction; ensure per-tenant pools and scrub hooks on release; propagate tags across stages.
- Minimal contract for a serving system:
  - Call tracer.allocate(tagged_meta)
  - Before any reuse/free: tracer.sanitize(handle) and wait for completion
  - On completion: tracer.attest_coverage(handle) must return ≥ 99.9%
  - Always emit forensic records (success or failure)

## 9) Metrics and Alerts

- Gauges: coverage_pct, unsanitized_regions_count, quarantine_count, sanitizer_duration_ms, allocator_reuse_rate, anomaly_rate.
- Alerts: coverage < 99.9%, unsanitized_regions_count > 0, cross-tenant tag mismatch, sanitizer overdue, hash chain break.

## 10) References and Diagrams

- See Mermaid diagrams under architecture/diagrams:
  - kv-cache-flow.mmd — request → cache → sanitization → forensic log → audit
  - pipeline-gates.mmd — CI/CD gates enforcing hygiene before deployment
