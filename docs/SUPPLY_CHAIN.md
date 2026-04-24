# Supply chain and provenance

## Current state

- **Dependencies**: Declared in `requirements.txt` (CI installs from this manifest).
- **Images**: `Dockerfile.cpu` / `Dockerfile.cuda` build reproducible base images; GitHub Actions
  runs Trivy SARIF upload on pushes to `main`.
- **Policy**: `policy/ci.rego` constrains workflow `permissions` via Conftest.

## Recommended hardening (org-owned)

1. **Lock files** — Add `requirements.lock` or `uv.lock` / `pip-tools` output; point CI at the lock
   file for deterministic installs (Codex §16).
2. **SLSA / attestations** — Enable GitHub artifact attestations or Sigstore cosign for published
   container images (`ghcr.io/...`).
3. **Dependabot / Renovate** — Automated manifest updates with CVE awareness.
4. **CodeQL** — Enable GitHub CodeQL workflow for Python (`security-extended` suite) in addition
   to Trivy.

These items require org-level configuration beyond this repository’s defaults.
