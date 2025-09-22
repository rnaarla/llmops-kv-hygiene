# llmops-kv-hygiene

![CI](https://github.com/${GITHUB_REPOSITORY:-owner/repo}/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Production‑grade KV‑cache hygiene for LLM serving: enforce zeroization, prove it, and gate it in CI/ops.

Who is this for

- Inference platform engineers running multi‑tenant LLMs
- MLOps/SREs needing objective hygiene gates and dashboards
- Compliance teams (SOC 2, HIPAA, FedRAMP) needing evidence trails

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

How it looks

```mermaid
flowchart LR
  A[Allocate/Tag] --> B[Mark In‑Use]
  B --> C[Sanitize (Zeroize)]
  C --> D[Verify (Sample/Double‑Pass)]
  D --> E{Coverage ≥ Threshold?}
  E -- Yes --> F[Free]
  E -- No --> G[Quarantine]
  A -.-> H[Forensic Log]
  B -.-> H
  C -.-> H
  D -.-> H
  F -.-> H
  G -.-> H
```

How it works (high level)

- Allocate and tag buffers per request/tenant → mark in‑use → zeroize on sanitize (CPU sync or CUDA stream/event‑aware) → verify via deterministic sampling → free only if coverage ≥ threshold (default 99.9%).
- Every lifecycle event is written to a hash‑chained forensic log with optional HMAC. Log rotation emits a linkage record that references the previous file and its last hash.
- Metrics summarize hygiene (unsanitized/quarantine counts, min/avg coverage, p50/p95 sanitize duration, reuse stats) for dashboards and gates.

Guarantees and limits

- Tamper‑evident logs: hash‑chained JSONL with optional HMAC; rotation linkage verified end‑to‑end.
- Coverage gate: free only when attested coverage ≥ threshold (default 99.9%).
- Verification: deterministic sampling; optional double‑pass for higher assurance.
- Limits: cannot scrub memory not owned/managed by your runtime; configure TTL/reuse to match your serving model.

Quickstart

- Install (Python 3.11+):
  - pip install -r requirements.txt
- Run tests and generate metrics/prom textfile in forensics/:
  - pytest -q
  - python - <<'PY'\nfrom tools.cache_tracer import CacheTracer\nt=CacheTracer()\nh=t.allocate('t1','r1','m1',(256,),"float32",'cpu','numpy')\nt.mark_in_use(h); t.sanitize(h,async_=False,verify=True); t.free(h)\nt.export_metrics('forensics/coverage.json'); t.export_metrics_prometheus('forensics/metrics.prom')\nPY
- Optional: enable pre‑commit hooks:
  - pip install pre-commit && pre-commit install

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

CacheTracer (CPU/GPU)

- Call order: `allocate` → `mark_in_use` → `sanitize` (zeroize) → optional `wait` → `free`.
- CPU mode uses NumPy; operations are synchronous.
- CUDA mode (optional PyTorch): `sanitize(async_=True)` records an event; call `wait()` before asserting coverage or freeing. Streams/events are honored.
- Verification controls: set `verify=True` on `sanitize`; adjust samples via env `KV_VERIFY_SAMPLES_DEFAULT`; enable double‑pass via `KV_DOUBLE_PASS_DEFAULT=true` for higher assurance.
- Policies: TTL (`KV_DEFAULT_TTL_SEC`) and max reuse (`KV_DEFAULT_MAX_REUSE`). TTL ≤ 0 flags immediate violation on first use. Any violation quarantines the handle.
- Coverage gate: free only when coverage ≥ `KV_COVERAGE_THRESHOLD` (default 99.9). Failing the gate raises and quarantines.
- Handles are unique and thread‑safe; unknown or already‑freed handles raise.

ForensicLogger (tamper‑evident logs)

- Log is append‑only JSONL with a SHA‑256 hash chain; optional HMAC if `FORENSIC_HMAC_SECRET` is set.
- Rotation emits a linkage marker referencing the previous file and last hash; `verify_all(path)` validates across rotations.
- Treat logs as immutable; any line edit breaks the chain. Keep directory permissions restrictive; files are created with 0600 when possible.
- Use `verify_chain(path)` for the active file, and `verify_all(path)` for full history.

Metrics export

- JSON: `export_metrics(path)` writes hygiene KPIs (coverage, durations, reuse/quarantine counts).
- Prometheus textfile: `export_metrics_prometheus(path)` writes HELP/TYPE and metrics suitable for the node exporter textfile collector.
- If multiple tracers run, write to distinct files and aggregate in your metrics pipeline.

CLI gates and tools

- Eviction checker: `python tools/eviction_checker.py <metrics.json> --coverage-min 99.9 --unsanitized-max 0 --quarantine-max 0`.
  - Exit code 0 = pass, 2 = fail. Use in CI to block merges.
- Verify logs + retention: `python -m tools.verify_logs --log-dir forensics --log-file kv_cache.log --out forensics/verification.json --retention-days 30 --max-rotated 20 [--archive-dir forensics/archive]`.
  - Verifies integrity before and after pruning. Prunes only rotated files; active `<stem>.log` remains.
- Metrics HTTP exporter (optional): `METRICS_FILE=forensics/metrics.prom METRICS_PORT=9101 python tools/metrics_exporter.py`.
  - Serves the textfile over HTTP for simple scraping.

CUDA specifics

- Requires PyTorch with CUDA and a compatible driver. Operations are async; rely on events/streams and use `wait()` when asserting coverage.
- Zeroization is bandwidth‑bound; verification sampling helps bound cost. For strict environments, enable double‑pass.

Kubernetes

- Mount a PVC at the forensic logs path for persistence. Provide `FORENSIC_HMAC_SECRET` via a Secret.
- Schedule `k8s/cronjob-verify-logs.yaml` for periodic integrity checks and retention pruning.
- CPU/GPU deployments are provided; ensure GPU nodes and drivers for CUDA images.

Security and secrets

- Set `FORENSIC_HMAC_SECRET` via a secret manager (K8s Secret/CI). Rotate keys periodically; subsequent lines use the new key. Verification assumes a single active key.
- Restrict file permissions and directory access; avoid placing logs on world‑writable paths.

Tuning and performance

- Increase `KV_VERIFY_SAMPLES_DEFAULT` for stronger verification on large buffers; enable double‑pass when policy requires.
- Keep coverage thresholds high (99.9% default). Lower only with explicit risk acceptance.
- For high‑throughput GPU workloads, batch `wait()` calls to reduce synchronization overhead.

Troubleshooting

- Chain verification fails after manual edits: forensic logs are append‑only; editing any line breaks the chain. Recreate logs or keep unmodified.
- Coverage below threshold: ensure sanitize is called and verification passes; adjust `KV_COVERAGE_THRESHOLD` only if policy allows.
- CUDA hangs: check stream usage and ensure `wait()` is called to synchronize events; verify driver/toolkit compatibility.

Core components

- CacheTracer: in‑process allocation tagging, zeroization, verification, policy enforcement, and metrics.
- ForensicLogger: append‑only, hash‑chained JSONL with optional HMAC and rotation linkage; `verify_chain`/`verify_all`.
- Eviction checker: CLI gate for CI/CD that enforces hygiene thresholds from exported metrics.
- Verify logs + retention: periodic integrity check and rotated log pruning; K8s CronJob provided.

Operational gates and K8s

- Prometheus textfile export for scrape/alerting.
- CronJobs for eviction checks and forensic verification/retention.
- CPU/CUDA images and manifests provided; mount PVC for logs and set `FORENSIC_HMAC_SECRET` via Secret.

Performance considerations

- Zeroization is O(n); verification sampling cost scales with samples chosen.
- CUDA paths are async; use `wait()` to synchronize when asserting coverage.
- Double‑pass increases confidence at the cost of extra write/verify cycles.

Security and compliance

- Secrets: provide `FORENSIC_HMAC_SECRET` via secret manager (K8s Secret/CI). Rotate periodically.
- Evidence: compliance mappings under `compliance/` reference logs, metrics, and CI artifacts as evidence.

Roadmap (short)

- Strict mypy typing gate (enable incrementally)
- SBOM publication badge and automated dependency update policy
- Optional write‑once log sink adapter (e.g., object storage with immutability)

License

MIT © Contributors. See `LICENSE`.

Contributing

See `CONTRIBUTING.md`. For security issues, see `SECURITY.md`.

## Hygiene Workflow & Quality Gates

The project enforces a layered hygiene pipeline so regressions are caught as early (and cheaply) as possible:

1. Local (developer workstation)

- `pre-commit` hooks: `ruff check --fix`, `ruff format`, and a fast `pytest -q -k "sanitization or isolation"` smoke slice.
- Goal: <10s feedback; reject style / obvious hygiene logic regressions before commit.

1. CI Test & Coverage Gate

- Full `pytest` run with `pytest --cov --cov-report=term-missing --cov-fail-under=90` in GitHub Actions and Jenkins.
- Fails the pipeline if aggregate line+branch coverage dips below 90%.

1. Hygiene Metrics Gate (optional separate job/stage)

- Runs `python tools/eviction_checker.py forensics/coverage.json --coverage-min 99.9 --unsanitized-max 0 --quarantine-max 0` against freshly exported metrics.
- Ensures real runtime hygiene (sanitization coverage per buffer) matches policy; lets per‑buffer attestation stay stricter (99.9%) than code coverage (90%).

1. Forensic Integrity Gate (periodic / cron)

- `python -m tools.verify_logs --log-dir forensics --log-file kv_cache.log --retention-days 30 --max-rotated 20`.
- Verifies hash chain + rotation linkage and prunes old rotated logs (active file never pruned) providing auditable integrity evidence.

Escalation path: failures at layers 1–2 block merges; layer 3 blocks deploy promotion; layer 4 raises operational alerts (not a merge blocker) but produces compliance evidence.

## Coverage policy

Current enforced line+branch coverage threshold: **90%**.

### Rationale

- Threshold raised from 87% → 90% after refactoring and targeted tests created stable headroom (observed 90.2–90.4% across multiple runs).
- Remaining uncovered lines are predominantly: (a) CUDA / hardware‑gated branches, (b) rare IO or corruption error handling, (c) defensive guards in `sanitizer` and log verification utilities.
- Further synthetic tests to chase the final ~10% would risk brittle mocks around hardware / filesystem edge cases without increasing real assurance.

### Improvement path (organic, low risk)

1. Isolate CUDA‑specific logic into a dedicated module to allow conditional import and focused tests when GPUs are present.
2. Add structured property / fuzz tests for sanitizer verification sampling to exercise additional statistical paths naturally.
3. Incrementally enable `mypy --strict` (module allow‑list expansion) to surface type drift early.
4. Consider raise to 92% only after: (a) CUDA module split, (b) zero flakiness over 30 consecutive CI runs, (c) coverage floor >92.3% p95.

### Current decomposition status

- Extracted modules: `buffer_model.py`, `metrics_utils.py`, `policies.py`, `sanitizer.py`, `forensic_logger.py` (reduces monolith complexity in `cache_tracer`).
- Sanitizer centralizes zeroization logic (NumPy / simulated CUDA) with `_verify_zero` seam for targeted test instrumentation.
- Edge‑case clarifications codified: zero‑length buffers = 100% coverage; negative `max_reuse` => unlimited; TTL ≤ 0 => immediate violation.

### Next planned steps

1. Introduce mypy incremental strict gates (warn → fail) in CI.
2. Optional GPU CI lane to execute real CUDA sanitize/wait paths when hardware is available.
3. Evaluate raising per‑buffer attestation threshold (currently 99.9%) if double‑pass becomes default policy.

### Guidance

- Do not lower coverage below 90% without explicit risk acceptance sign‑off.
- New modules should target ≥95% coverage (≥99% if pure logic, excluding hardware / randomness branches).
- Keep attestation (runtime coverage per buffer) threshold distinct and higher than code coverage to maintain defense in depth.

