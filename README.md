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

1. Install (Python 3.11+):

  ```bash
  pip install -r requirements.txt
  ```

1. Run tests:

  ```bash
  pytest -q
  ```

1. Minimal tracer usage (keyword-only API):

  ```python
  from tools.cache_tracer import CacheTracer

  tracer = CacheTracer()
  handle = tracer.allocate(
     tenant_id="tenantA",
     request_id="req123",
     model_id="modelX",
     shape=(256,),
     dtype="float32",
     device="cpu",
     framework="numpy",
  )
  tracer.mark_in_use(handle)
  tracer.sanitize(handle, async_=False, verify=True)
  tracer.free(handle)
  tracer.export_metrics("forensics/coverage.json")
  tracer.export_metrics_prometheus("forensics/metrics.prom")
  ```

1. Optional: enable pre‑commit hooks:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

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

## CI/CD & Security Scanning Enhancements

Recent pipeline hardening work introduced layered quality and security gates:

1. Test & Coverage Matrix
   - Multi-version Python test matrix (see `ci.yml`) with unified coverage aggregation.
   - Fails if overall line/branch coverage < **90%**.
2. Forensic & Hygiene Gates
   - Metrics export + `eviction_checker.py` invoked to enforce runtime hygiene (sanitization coverage, quarantine counts).
3. Container Build & Scan
   - CPU + (optional) CUDA images built and tagged with short SHA; Trivy scans (table + SARIF) execute post-build.
4. SBOM & Vulnerability Surfaces
   - Syft/Trivy generated artifacts (SBOM and SARIF) uploaded for audit; can be consumed by code-scanning UI.
5. Optional Vulnerability Gating (HIGH/CRITICAL)
   - Controlled by secret `VULN_GATING`; when set to `true`, any HIGH or CRITICAL finding in Trivy SARIF fails the job.
6. Standalone Scheduled Scan
   - `trivy-scan.yml` runs on `main` pushes + weekly cron and supports the same optional gating.

## Feature Matrix

| Category | Feature | Description | Primary File(s) / Workflow | Default Enabled | Gated / Config | Failure Impact |
|----------|---------|-------------|-----------------------------|-----------------|----------------|----------------|
| Core Hygiene | Allocation Tagging | Per-tenant/request/model metadata on buffers | `tools/cache_tracer.py` | Yes | — | N/A |
| Core Hygiene | Zeroization (CPU / Torch) | Full buffer overwrite + coverage computation | `cache_tracer.py`, `sanitizer.py` | Yes | `KV_DOUBLE_PASS_DEFAULT` (2nd pass) | Low (quarantine if coverage inadequate) |
| Core Hygiene | Deterministic Sampling Verify | Sample indices recorded for attestation | `cache_tracer.py` | Yes | `KV_VERIFY_SAMPLES_DEFAULT` | If verify fails → coverage=0 → quarantine |
| Policies | TTL Enforcement | Quarantine on expired lifetime | `policies.py`, `cache_tracer.py` | Yes (if TTL provided) | `KV_DEFAULT_TTL_SEC` | Buffer quarantined |
| Policies | Reuse Limit | Quarantine when reuse > limit | `cache_tracer.py` | Yes | `KV_DEFAULT_MAX_REUSE` | Buffer quarantined |
| Forensics | Hash-Chained Log | Append-only JSONL with prev/curr hash | `forensic_logger.py` | Yes | `FORENSIC_HMAC_SECRET` (adds HMAC) | Integrity evidence degraded if broken |
| Forensics | Rotation & Linkage | Size-based rotation with linkage record | `forensic_logger.py` | Yes | `_max_bytes` ctor param | Linkage mismatch surfaced in verify |
| Forensics | Integrity Verification | Single file & full rotation chain checks | `forensic_logger.py`, `verify_logs.py` | Manual/CI | CLI invocation | Non-zero exit (ops alert) |
| Forensics | Log Pruning & Archive | Age/Count prune with optional archive directory | `log_pruner.py`, `verify_logs.py` | On demand | CLI flags / env | None (older logs removed) |
| Metrics | JSON Metrics Export | Hygiene KPIs (coverage, durations, reuse) | `cache_tracer.py` | Manual | — | N/A |
| Metrics | Prometheus Textfile Export | Textfile collector lines for scraping | `cache_tracer.py` | Manual | — | Observability gap if missing |
| Metrics | HTTP Metrics Server | Lightweight HTTP `/metrics` endpoint | `metrics_exporter.py` | Manual start | `METRICS_FILE`, `METRICS_PORT` | Monitoring gap if absent |
| CI Gate | Hygiene Threshold Checker | Enforces coverage / quarantine limits | `eviction_checker.py` | In CI | CLI flags | Merge/deploy blocked |
| CI Gate | Coverage ≥ 90% | Line/branch coverage enforcement | `ci.yml` | Yes | Coverage flag in workflow | Merge blocked |
| Security | Container Image Build | CPU(+CUDA) image build & tag | `ci.yml` | On non-PR | — | Deploy artifact missing |
| Security | Trivy Scan (Image/FS) | Vulnerability + secret scan (table + SARIF) | `ci.yml`, `trivy-scan.yml` | Yes | `VULN_GATING` secret | If gated & fail → pipeline fails |
| Security | Vulnerability Gating (HIGH/CRITICAL) | Fail on any HIGH/CRITICAL findings | Workflows (conditional step) | Off by default | `VULN_GATING=true` | Build/scan job fails |
| Security | SBOM Generation | Software bill of materials artifact | `ci.yml` | Yes | — | Compliance signal reduced if absent |
| Security | Scheduled Weekly Scan | Cron-based fresh scan of code/image | `trivy-scan.yml` | Yes | `VULN_GATING` secret | Alerts via gating failure |
| Compliance | HMAC Log Strengthening | Adds HMAC to each forensic line | `forensic_logger.py` | Optional | `FORENSIC_HMAC_SECRET` | Lower tamper resistance if unset |
| Observability | Activation Anomaly Logging | z-score + max outlier detection | `activation_logger.py` | Manual use | `z_threshold` arg | Missed anomaly detection |
| Resilience | Double-Pass Sanitization | Second overwrite for assurance | `cache_tracer.py` | Off | `KV_DOUBLE_PASS_DEFAULT=true` | Slight perf cost if enabled |
| Performance | P50/P95 Track | Sanitize duration distribution | `cache_tracer.py` | Yes | — | Missing latency insight |
| Performance | Reuse Rate Metric | reuse_events / allocations | `cache_tracer.py` | Yes | — | Policy tuning insight lost |
| Ops | Verify + Prune CLI | Combined integrity + retention tool | `verify_logs.py` | Manual/cron | CLI flags | Old logs accumulate if unused |
| Future (Planned) | Kubernetes Cron Examples | CronJobs for verify/prune/gates | (planned docs) | No | — | Manual scripting needed |
| Future (Planned) | Real CUDA Async Streams | True async zeroization path | (future implementation) | No | — | Lower overlap efficiency |
| Future (Planned) | Composite Action for Gating | Reusable vuln gate action | (future action) | No | — | Duplication across repos |


### Enabling Vulnerability Gating

Add a repository (or org) secret named `VULN_GATING` with value `true`.

Behavior:

- Secret unset or not exactly `true` (case-sensitive) ⇒ scans are informational only (summaries + artifacts).
- Secret set to `true` ⇒ build or scheduled scan job fails if any HIGH or CRITICAL severity is present.

### Gating Logic (Excerpt from `ci.yml`)

```yaml
    - name: Export gating flag
      env:
        RAW_FLAG: ${{ secrets.VULN_GATING }}
      run: echo "VULN_GATING=${RAW_FLAG:-false}" >> $GITHUB_ENV

    - name: Enforce vulnerability policy (fail on any HIGH/CRITICAL)
      if: ${{ env.VULN_GATING == 'true' }}
      run: |
        python - <<'PY'
import json, pathlib, sys
sarif = pathlib.Path('trivy-image-results.sarif')  # (example path)
if not sarif.exists():
    print('[gate] No SARIF produced; allowing pass.')
    raise SystemExit(0)
data = json.loads(sarif.read_text())
results = data.get('runs', [{}])[0].get('results', [])
sev = {}
for r in results:
    lvl = r.get('level','').upper()
    sev[lvl] = sev.get(lvl,0)+1
crit = sev.get('CRITICAL',0)
high = sev.get('HIGH',0)
print(f'[gate] HIGH={high} CRITICAL={crit}')
if crit or high:
    print('[gate] Policy violation: HIGH/CRITICAL findings present.')
    sys.exit(1)
print('[gate] Pass: no HIGH/CRITICAL findings.')
PY
```

### Standalone Trivy Scan (`trivy-scan.yml`)

Runs a matrix across `image` and `fs` targets:

- Pulls latest built image (best effort) and scans it plus the repository filesystem.
- Generates table output + SARIF (`trivy-<target>-results.*`).
- Optional gating (same `VULN_GATING` secret) uses a lightweight `jq` parser to count HIGH/CRITICALs.

Snippet of gating via `jq` in scheduled scan:

```yaml
    - name: Enforce vulnerability policy (optional)
      if: ${{ env.VULN_GATING == 'true' }}
      run: |
        f=trivy-image-results.sarif
        if [ ! -f "$f" ]; then echo "[gate] No SARIF produced; allowing pass."; exit 0; fi
        crit=$(jq -r '[.runs[0].results[]?.level // empty | ascii_upcase | select(.=="CRITICAL")] | length' "$f")
        high=$(jq -r '[.runs[0].results[]?.level // empty | ascii_upcase | select(.=="HIGH")] | length' "$f")
        echo "[gate] HIGH=$high CRITICAL=$crit"
        if [ "$crit" -gt 0 ] || [ "$high" -gt 0 ]; then
          echo "[gate] Policy violation: HIGH/CRITICAL findings present."; exit 1; fi
        echo "[gate] Pass: no HIGH/CRITICAL findings."
```

### Customizing / Extending Gating

You can introduce thresholds instead of zero tolerance:

```yaml
env:
  MAX_HIGH: 0
  MAX_CRITICAL: 0
...
    - name: Enforce policy (tolerances)
      if: ${{ env.VULN_GATING == 'true' }}
      run: |
        f=trivy-image-results.sarif
        [ -f "$f" ] || { echo '[gate] No SARIF'; exit 0; }
        crit=$(jq -r '[.runs[0].results[]?.level // empty | ascii_upcase | select(.=="CRITICAL")] | length' "$f")
        high=$(jq -r '[.runs[0].results[]?.level // empty | ascii_upcase | select(.=="HIGH")] | length' "$f")
        echo "[gate] HIGH=$high CRITICAL=$crit (limits H=$MAX_HIGH C=$MAX_CRITICAL)"
        if [ "$crit" -gt "$MAX_CRITICAL" ] || [ "$high" -gt "$MAX_HIGH" ]; then
          echo '[gate] Policy violation'; exit 1; fi
        echo '[gate] Pass';
```

### Consuming SARIF Locally

To inspect findings locally without waiting for code scanning UI:

```bash
jq '.runs[0].results[] | {ruleId, level, message: .message.text}' trivy-image-results.sarif | less
```

### Failure Semantics

- Gating steps run after artifacts upload, so even on failure you retain SBOM/SARIF evidence.
- Missing SARIF files default to soft-pass (graceful degradation) to avoid noisy false negatives when an upstream scan is skipped.

### Linter Warnings About `secrets.*`

Local YAML or action linters may surface warnings like “Context access might be invalid: VULN_GATING”. These are static-analysis false positives—the live GitHub Actions runtime resolves undefined secrets to empty strings; gating defaults `false` unless explicitly enabled.

Mitigations (optional):

- Add ignore directives (`# yamllint disable-line`).
- Provide a mock `.secrets-example` file for certain linters.
- Wrap secret expansion in a neutral step that echoes a redacted placeholder when unset.

---
_Updated: Vulnerability gating, scheduled scanning, coverage hardening, and forensic gate documentation added._

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
