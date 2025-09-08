# Contributing

Thanks for your interest in contributing!

- Use Python 3.11+.
- Set up a virtualenv and install dev deps from `requirements.txt`.
- Run the test suite locally: `pytest -q`.
- Format with `black` and lint with `ruff` before pushing.
- Include tests for new features and bug fixes.
- Keep docs up to date (README, architecture, compliance mappings if relevant).

Quick start

- Install and enable hooks:
  - pip install pre-commit
  - pre-commit install
  - pre-commit run --all-files  # run on existing code

Workflow

1. Fork and create a feature branch.
2. Make your changes with clear, small commits.
3. Run tests and linters; ensure all pass.
4. Open a Pull Request with a concise description and rationale.

Code style

- Follow PEP 8 where practical; prefer explicit names.
- Avoid breaking public APIs without a deprecation note.
- Keep functions small and testable.

Security

- Never commit secrets. Use environment variables/Secrets.
- Report vulnerabilities privately (see `SECURITY.md`).
