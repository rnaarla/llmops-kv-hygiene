# Implementation status (90-day enterprise plan)

This file tracks **delivered** work against the phased roadmap. Each band lists what is **in
this repository** versus what remains **integration work in consuming services**.

## Week 1–2 — Phase 0–1 foundations (shipped)

- Forensic strict HMAC verification when a verifier key is configured.
- NumPy sampled verification (no silent trust path).
- CI least privilege (`contents: read` default; `update-metrics` write exception in OPA policy).
- Removal of unused AI CI jobs and dead matrix outputs.
- Conftest policy (`policy/ci.rego`, OPA v1, `package main`).
- Repository hygiene (`.gitignore`, `.markdownlint.yaml`, duplicate docs removed).

## Week 3–4 — Phase 1–2 evidence and correctness (shipped)

- **Forensic mirror sinks**: `ForensicLogger(mirror_paths=…)` / `mirror_sinks=…` and
  `CacheTracer(forensic_mirror_paths=…)` replicate JSONL lines for WORM staging or upload jobs;
  mirror failures are logged and never block the primary append (`tools/forensic_sink.py`).
- **CUDA stream zeroization**: `zeroize_cuda` uses `torch.cuda.stream(buf.stream_obj)` when
  `async_=True` and a stream is bound; `SanitizationResult.scheduled_async` propagates to
  `CacheTracer` so `sanitize(..., async_=True)` + `wait()` ordering matches GPU tests.
- **Micro-benchmark script**: `scripts/microbench_sanitize.py` (local use; not a CI gate).

## Week 5–6 — Phase 3 CI and supply chain (partially shipped)

- **Policy-as-code**: Conftest workflow checks (see Week 1–2).
- **Supply chain guidance**: `docs/SUPPLY_CHAIN.md` (lockfiles, SLSA, Dependabot, CodeQL — org
  actions to enable).
- **Threat model**: `docs/THREAT_MODEL_KV_HYGIENE.md` (bounded scope for this library).

## Week 7–8 — Phase 4 integration boundary (shipped as contract)

- **`HygieneRuntimeHooks` protocol** under `tools/adapters/` — implement in your serving
  repository; no bundled vLLM/TensorRT-LLM fork (keeps this repo small and license-clean).

## Week 9–10 — Phase 5 performance and reliability (ongoing)

- Local micro-benchmark only; no continuous perf gate in CI (add org-owned load tests if needed).
- Chaos and fuzz tests remain in `tests/`; extend with GPU hardware lanes when available.

## Week 11–13 — Phase 6–7 compliance and ops (documentation)

- Compliance mappings remain under `compliance/`; evidence automation (signed bundles per
  release) is an **org pipeline** step — document requirements in `docs/SUPPLY_CHAIN.md` and
  `docs/POST_IMPLEMENTATION_REVIEW.md`.

## Verification

Last full `pytest` run before tagging this status: run `python3 -m pytest -q` on `main` after
merge (see commit history).

## How to update this document

After each merged milestone, append a dated subsection with **scope**, **commit**, **tests
run**, and **known limitations**.
