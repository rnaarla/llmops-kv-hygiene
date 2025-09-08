# llmops-kv-hygiene

![CI](https://github.com/${GITHUB_REPOSITORY:-owner/repo}/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Production-grade KV‑cache hygiene toolkit for LLM serving.

Why this matters

- KV caches can leak cross‑request/tenant data if not scrubbed. This project enforces zeroization, proves it with tamper‑evident logs, and exposes gates for CI/CD and ops.
- Alignment materials for SOC 2, HIPAA, and FedRAMP High are included under `compliance/` as the evidence layer.

What’s inside

- Tamper‑evident forensic logging: append‑only JSONL with SHA‑256 hash chain and optional HMAC (`FORENSIC_HMAC_SECRET`). Rotation markers link files; `verify_all()` validates chains across rotations. See `tools/cache_tracer.py` (ForensicLogger).
- KV‑cache tracer and enforcer: allocation tagging, sanitization (CPU/NumPy and optional CUDA/PyTorch), coverage attestation with deterministic sampling, policy enforcement (TTL, max reuse), quarantine on violations, and metrics export (JSON + Prometheus). See `tools/cache_tracer.py`.
- Threshold gates for CI/CD: `tools/eviction_checker.py` consumes exported metrics and fails builds if coverage or hygiene thresholds aren’t met.
- Metrics: JSON via `export_metrics`, Prometheus textfile via `export_metrics_prometheus`. Optional simple HTTP exporter `tools/metrics_exporter.py`.
- Periodic verification and retention: `tools/verify_logs.py` + `k8s/cronjob-verify-logs.yaml` to schedule `verify_all` and prune rotated logs by age/count.
- Tests: unit/integration and a fuzz harness under `tests/`.
- Architecture and principles: `architecture/` with overview and diagrams.

How it works (high level)

- Allocate and tag buffers per request/tenant → mark in‑use → zeroize on sanitize (CPU sync or CUDA stream/event‑aware) → verify via deterministic sampling → free only if coverage ≥ threshold (default 99.9%).
- Every lifecycle event is written to a hash‑chained forensic log with optional HMAC. Log rotation emits a linkage record that references the previous file and its last hash.
- Metrics summarize hygiene (unsanitized/quarantine counts, min/avg coverage, p50/p95 sanitize duration, reuse stats) for dashboards and gates.

Setup

- Python 3.11+
- Install dependencies (NumPy required; PyTorch optional for CUDA):
  - pip install numpy pytest
  - Optional: pip install torch torchvision --index-url <https://download.pytorch.org/whl/cu121>  # or CPU wheels
- Run tests: `pytest -q`
- Optional: enable pre-commit hooks:
  - pip install pre-commit
  - pre-commit install

Python usage

```python
from tools.cache_tracer import CacheTracer, ForensicLogger

tracer = CacheTracer(log_path="forensics/kv_cache.log")
h = tracer.allocate(tenant_id="t1", request_id="r1", model_id="m1", shape=(256,), dtype="float32", device="cpu", framework="numpy")
tracer.mark_in_use(h)
tracer.sanitize(h, async_=False, verify=True)
tracer.free(h)

# Export metrics
tracer.export_metrics("forensics/coverage.json")
tracer.export_metrics_prometheus("forensics/metrics.prom")

# Verify log integrity (active file)
print(ForensicLogger.verify_chain("forensics/kv_cache.log"))
# Verify across rotations
print(ForensicLogger.verify_all("forensics/kv_cache.log"))
```

CLI tools

- Eviction checker (CI gate):
  - `python tools/eviction_checker.py forensics/coverage.json --coverage-min 99.9 --unsanitized-max 0 --quarantine-max 0`
- Verify logs + retention (ops):
  - `python -m tools.verify_logs --log-dir forensics --log-file kv_cache.log --out forensics/verification.json --retention-days 30 --max-rotated 20 --archive-dir forensics/archive`
- Metrics HTTP exporter (optional):
  - `METRICS_FILE=forensics/metrics.prom METRICS_PORT=9101 python tools/metrics_exporter.py`

Kubernetes (optional)

- Scheduled verification and retention:
  - `kubectl apply -f k8s/cronjob-verify-logs.yaml`
  - Ensure a PVC with forensic logs is mounted by the CronJob and set the image to your build.

Configuration (env vars)

- FORENSIC_HMAC_SECRET: optional HMAC key for forensic lines.
- KV_COVERAGE_THRESHOLD: minimum coverage percent to allow free (default 99.9).
- KV_DOUBLE_PASS_DEFAULT: if set true, runs a second zeroization pass by default.
- KV_VERIFY_SAMPLES_DEFAULT: number of elements to sample for verification.
- KV_DEFAULT_MAX_REUSE: default max reuse before quarantine (0 → any reuse quarantines).
- KV_DEFAULT_TTL_SEC: default TTL; <=0 causes immediate TTL violation on first use.

Tests overview (selected)

- tests/test_kv_sanitization.py
  - CPU sanitize + free happy path
  - Free without sanitize quarantines and raises
  - Verification failure causes quarantine
  - Prometheus export contains expected metrics
  - TTL expiry triggers quarantine
  - Prometheus line prefixes sanity
- tests/test_kv_isolation.py
  - Unique handles and UnknownHandle
  - Forensic chain validity; HMAC chain validity
  - Eviction checker CLI return codes
  - Activation logger anomaly + rate limit
- tests/fuzzing_harness.py
  - Multi‑threaded, randomized allocate/sanitize/free with no cross‑session leakage

Compliance docs

- SOC 2 mapping: `compliance/soc2_alignment.md`
- HIPAA alignment (PHI protection): `compliance/hipaa_alignment.md`
- FedRAMP High (AU controls, retention): `compliance/fedramp_high_alignment.md`

Notes and usage hints

- CUDA: GPU paths require PyTorch with CUDA and a compatible driver; CUDA operations are async and synchronized via events/streams in `sanitize()/wait()`.
- File perms: forensic logs are created with 0600 perms when possible; keep the log directory restricted.
- Secrets: set `FORENSIC_HMAC_SECRET` via a secret manager (K8s Secret, CI secrets). Rotating the key will cause subsequent lines to use the new HMAC; verification accepts a single key at a time.
- Retention: `verify_logs.py` prunes only rotated files; active `<stem>.log` remains. Consider offloading archives to object storage with lifecycle policies.
- Tuning: for very large buffers, increase samples or enable double‑pass to harden verification.

Troubleshooting

- Chain verification fails after manual edits: forensic logs are append‑only; editing any line breaks the chain. Recreate logs or keep unmodified.
- Coverage below threshold: ensure sanitize is called and verification passes; adjust `KV_COVERAGE_THRESHOLD` only if policy allows.
- CUDA hangs: check stream usage and ensure `wait()` is called to synchronize events; verify driver/toolkit compatibility.
