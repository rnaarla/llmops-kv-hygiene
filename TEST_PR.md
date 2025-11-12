# Test PR for Docker Scan Validation

This is a test pull request to verify that the `docker-scan-pr` job runs correctly on PRs.

## What this tests

- Docker image build without pushing
- Trivy security scan on PR images
- SARIF upload to GitHub Security tab
- Disk cleanup before build

## Expected behavior

- ✅ Docker image builds successfully
- ✅ Trivy scan completes
- ✅ No actual image push to registry
- ✅ Results appear in Security tab

This file can be deleted after PR validation.
