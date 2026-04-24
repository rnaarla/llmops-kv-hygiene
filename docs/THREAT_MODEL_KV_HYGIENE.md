# Threat model — KV cache hygiene

## Scope

In-process hygiene for tracked buffers: allocation tagging, sanitization, verification,
forensic logging, and CI gates shipped in this repository.

## Trust boundaries

| Boundary | Trust assumption | Failure impact |
| --- | --- | --- |
| Process | Tracer owns tensor lifecycle | Cross-request leakage if hooks bypassed |
| Tenant | `tenant_id` labels are correct from caller | Mis-labeling breaks isolation guarantees |
| Forensic log | Append-only file or mirror path | Tampering breaks hash chain / HMAC |
| CI | Deterministic tests and policy | Bad deploy if gates removed |

## Threats and mitigations

1. **Stale KV reuse** — Mitigation: sanitize + coverage threshold before `free`; quarantine on failure.
2. **Async GPU race** — Mitigation: `wait()` sync; optional CUDA stream in `zeroize_cuda` when `stream_obj` set.
3. **Log tampering** — Mitigation: hash chain; optional HMAC; strict verify mode when key supplied.
4. **Mirror sink outage** — Mitigation: mirror failures are logged; primary append still succeeds.
5. **Resource exhaustion** — Mitigation: bounded verification samples; configurable log rotation.

## Out of scope

Physical GPU side-channel attacks, compromise of the host kernel, and incorrect upstream
allocator behavior without tracer hooks are out of scope for this library.
