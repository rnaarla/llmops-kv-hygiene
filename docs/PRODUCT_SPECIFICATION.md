# Product specification — KV hygiene toolkit

## Purpose

`llmops-kv-hygiene` is a **Python library and operational toolkit** for multi-tenant LLM serving platforms. It enforces and evidences **KV-cache sanitization** (zeroization), **coverage attestation**, **policy gates** (TTL, reuse limits), and **tamper-evident forensic logging** suitable for enterprise audit and SRE workflows.

## Functional requirements

### FR-1: Lifecycle tracing

- Every tracked buffer has a unique handle, tenant/request/model metadata, device, shape, and lifecycle state transitions logged to append-only JSONL.

### FR-2: Sanitization and attestation

- CPU buffers (NumPy) and Torch tensors are zeroized through `sanitizer.py`.
- When verification is enabled, **sampled checks apply to NumPy and Torch** (no silent trust path on CPU ndarray).
- Coverage is computed as scrubbed bytes versus total bytes; freeing below the configured threshold quarantines and raises.

### FR-3: Forensic integrity

- Each log line includes a SHA-256 hash chain (`prev_hash`, `curr_hash`).
- When `FORENSIC_HMAC_SECRET` is set, new records include an `hmac` field.
- **Strict verification**: if verification is run with an HMAC key (environment or `verify_chain(..., hmac_secret=...)`), **every** non-empty line must include a valid `hmac`. Lines written without HMAC fail closed under that verifier configuration.
- **Optional replication**: `ForensicLogger` accepts `mirror_paths` or `mirror_sinks` to append the
  same JSONL line to secondary files or custom sinks (for WORM staging, SIEM upload, or object
  storage pipelines). Mirror failures are logged; the primary log still appends.

### FR-4: Metrics and gates

- JSON and Prometheus textfile exports summarize hygiene KPIs.
- `eviction_checker.py` produces a machine-readable verdict for CI/CD.

### FR-5: Operations

- `verify_logs` verifies chains, optionally prunes rotated files, and writes a JSON verdict.
- Kubernetes examples under `k8s/` illustrate scheduled verification and retention.

## Non-goals (current release)

- **Not** a full inference server: no bundled vLLM/TensorRT-LLM integration; use
  `tools.adapters.protocol.HygieneRuntimeHooks` as the integration contract in your serving repo.
- **Not** a turnkey WORM object-store writer: replication hooks ship in-process; upload to S3/GCS
  with Object Lock remains an external job consuming mirror files or sinks.
- **No** third-party LLM API calls in CI (removed for cost, determinism, and attack surface per
  Engineering Codex economic-efficiency rules).

## CUDA behavior

- When `buf.stream_obj` is a CUDA stream and `sanitize(..., async_=True)` is used, `zeroize_cuda`
  submits `zero_()` on that stream and returns `True` so callers can `wait()` on ordering.
- Without a stream, CUDA zeroization runs synchronously on the default stream context.

## Interfaces (stable contracts)

- **Python**: `CacheTracer`, `ForensicLogger`, `eviction_checker`, `verify_logs` CLI — public entry
  points documented in `README.md`.
- **Adapters**: `tools.adapters.protocol.HygieneRuntimeHooks` — integration boundary for serving
  runtimes (implement in your deployment repo).
- **Metrics identifiers**: Prometheus names prefixed with `kv_hygiene_*` (do not rename without deprecation, per telemetry stability expectations).

## Quality bar

- Unit and integration tests in `tests/` are required for behavioral changes.
- GitHub Actions: lint, typecheck, Conftest policy, coverage threshold, security scan, Docker publish path on `main`.
- OPA policy (`policy/ci.rego`) enforces least-privilege default `permissions.contents: read` on the workflow file.
