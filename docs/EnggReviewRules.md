# Engineering Review Rules

> These rules define the expectations for every code review in the HybridRAG
> repository.  They apply equally to human reviewers and automated review tools.
> Rules are derived from the [Engineering Codex](Engineering-Codex.md)
> and the [CODEX-COMPLIANCE-MATRIX](governance/CODEX-COMPLIANCE-MATRIX.md).

---

## 1. Reviewer Responsibilities

A reviewer who approves a PR accepts joint responsibility for the correctness
and safety of the merged change.  Approval must not be granted unless the
reviewer has:

1. Read the entire diff, not just the summary.
2. Verified that the stated intent of the PR matches the actual change.
3. Confirmed that all mandatory checklist items (§4) are satisfied.
4. Raised or acknowledged every concern found during the review.

Rubber-stamp approvals ("LGTM" without evidence of inspection) are a policy
violation.

---

## 2. Mandatory Pre-Merge Checks

The following automated gates must be green before any PR is merged.  Reviewers
must not approve a PR while gates are still failing.

| Gate | Workflow / Tool |
| --- | --- |
| Unit and integration tests | `ci.yml` |
| Linting and formatting | `ci.yml` |
| CodeQL SAST | `codeql.yml` |
| Secrets detection | `security-scanning.yml` |
| Dependency vulnerability scan | `dependency-check.yml` |
| Documentation integrity | `docs-integrity.yml` |
| Contract tests (if API/SDK touched) | `sdk-contract-tests.yml` |
| Container security scan (if Dockerfile touched) | `container-security.yml` |

A failing gate may be overridden only by a repository maintainer, who must
document the reason in the PR comments and create a tracking issue for
remediation.

---

## 3. Review Scope by Change Type

### 3.1 Feature Changes

- Verify that new functionality is covered by unit, integration, and (where
  applicable) contract tests.
- Check that public interfaces follow versioning rules (Codex §15.1).
- Confirm that observability requirements are met: structured logs, metrics,
  and spans (Codex §11.4).
- Ensure the CHANGELOG has a `### Added` entry.

### 3.2 Bug Fixes

- Confirm that a regression test reproducing the bug is included.
- Check that the fix does not silently change behavior elsewhere (Codex §1).
- Ensure the CHANGELOG has a `### Fixed` entry.

### 3.3 Refactoring / Cleanup

- Verify behavioral invariance: contract tests must pass without modification
  (Codex §§1, 3).
- Reject any refactor that removes, renames, or suppresses tests.
- Confirm that telemetry identifiers are unchanged (Codex §10).

### 3.4 Infrastructure / IaC Changes

- Confirm `iac-security.yml` scan passes.
- Verify that no IAM wildcards, public storage resources, or unencrypted
  resources are introduced.
- Check that Terraform/CloudFormation plan output is included in the PR
  description or attached as a CI artifact.

### 3.5 Dependency Updates

- Confirm that lock files are updated alongside manifest files (Codex §16).
- Verify that no new `CRITICAL` CVEs are introduced.
- Check for unused dependency removal opportunities.

### 3.6 Documentation-Only Changes

- Confirm that referenced file paths, code snippets, and command examples are
  accurate.
- Verify that `docs-integrity.yml` passes (all required files present, links
  valid).

---

## 4. PR Author Checklist

Every PR description must include a completed checklist.  Reviewers must verify
that items are honestly checked, not blindly ticked.

```markdown
## PR Checklist

### Correctness
- [ ] All CI gates are green (see §2 above)
- [ ] New and changed code paths are covered by tests
- [ ] Regression test added for any bug fix

### Behavioral Invariance
- [ ] No externally observable behavior change (or change is intentional and
      described in the PR body)
- [ ] Telemetry identifiers (metric names, log keys, span names) are unchanged
      or deprecated according to policy

### Security
- [ ] No secrets, credentials, or PII in the diff
- [ ] No new `CRITICAL`/`HIGH` vulnerability findings
- [ ] IAM and storage changes reviewed against least-privilege policy

### Documentation and Hygiene
- [ ] CHANGELOG updated (Added / Fixed / Changed / Deprecated / Removed)
- [ ] Relevant documentation updated (API.md, runbooks, ADRs, etc.)
- [ ] No build artifacts or transient files committed
- [ ] If structural change: rationale and preserved-behavior description included
      (Codex §19)
```

---

## 5. Commenting Standards

### 5.1 Comment Types

Use standard prefixes so automated tooling and other reviewers can triage
comments quickly:

| Prefix | Meaning |
| --- | --- |
| `MUST:` | Blocking — must be resolved before merge |
| `SHOULD:` | Strong recommendation — author must address or explicitly decline with reason |
| `NIT:` | Non-blocking style or minor preference |
| `Q:` | Question for understanding — may or may not require action |

### 5.2 Actionable Comments

Every `MUST:` and `SHOULD:` comment must:

- Identify the specific line(s) or section in question.
- Explain the concern clearly (why, not just what).
- Suggest a concrete remediation path where possible.

Vague comments such as "this looks wrong" without explanation must be revised
before a review is considered complete.

### 5.3 Resolving Comments

- Only the **comment author** or a repository maintainer may mark a thread as
  resolved.
- Resolving a `MUST:` thread requires either a code change addressing the
  concern or a written explanation of why the concern does not apply.

---

## 6. Security-Sensitive Review Rules

The following patterns require heightened scrutiny.  The reviewer must
explicitly confirm in the PR comments that each applicable pattern was
inspected.

| Pattern | Required Check |
| --- | --- |
| New external HTTP calls | Validate URL construction; confirm no SSRF surface |
| User-controlled input used in SQL / commands | Confirm parameterization / escaping |
| New IAM permissions | Confirm least privilege; confirm no wildcard actions |
| Deserialization of untrusted data | Confirm safe deserialization library and schema validation |
| Secrets or credentials referenced in code | Confirm all values come from environment or secrets manager |
| Cryptographic operations | Confirm use of approved algorithms (AES-256, RSA-2048+, SHA-256+) |
| New public endpoints (no auth) | Confirm intentional; confirm rate limiting in place |

---

## 7. Merge and Squash Policy

- PRs with a single logical change should be **squash-merged** to keep the
  commit history clean.
- PRs that represent a series of logically distinct steps (e.g., a multi-step
  migration) may be **merge-committed** to preserve the step-by-step history;
  each commit must have a meaningful message.
- **Force pushes to `main` / `master` are forbidden.**
- **Direct commits to `main` / `master` are forbidden** except for automated
  release tooling operating through the CI/CD pipeline.

---

## 8. Escalation and Maintainer Review

The following change categories require at least one repository maintainer
review in addition to the standard peer review:

- Changes to `.github/workflows/`
- Changes to `infrastructure/` (IaC) touching production environments
- Changes to security policies (`security/`, `infrastructure/terraform/policies/`)
- Public API or SDK breaking changes
- Changes to this document (`docs/EnggReviewRules.md`) or the Engineering Codex
  (`docs/Engineering-Codex.md`)

Maintainer review must be documented with an explicit approval comment; implicit
approval through CI pass alone is not sufficient.

---

## 9. Handling Exceptions

When a rule cannot be followed due to exceptional circumstances:

1. The PR author documents the exception in the PR description with a full
   rationale.
2. A tracking issue is created for remediation.
3. A repository maintainer explicitly approves the exception in a PR comment.
4. The exception is noted in the CHANGELOG.

Repeated exceptions (three or more for the same author or rule within a single
quarter) must trigger a retrospective to determine whether the rule or the
workflow needs to change.
