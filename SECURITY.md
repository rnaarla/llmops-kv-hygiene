# Security Policy

Supported versions

- Main branch only. Use latest release for fixes.

Reporting a vulnerability

- Do not open a public issue.
- Email security contact: [security@example.com](mailto:security@example.com)
- Include steps to reproduce, impact, affected versions, and any CVEs if known.
- We aim to acknowledge within 72 hours.

Best practices

- Never commit secrets; use `FORENSIC_HMAC_SECRET` via secrets manager.
- Rotate keys periodically; verify logs with `tools/verify_logs.py`.
- Run CI scanners (Trivy) and keep dependencies updated.
