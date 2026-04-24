# Engineering Codex

> The authoritative set of engineering principles governing the HybridRAG platform.
> Every change to production code, infrastructure, documentation, or configuration
> must satisfy the applicable sections below.  The
> [CODEX-COMPLIANCE-MATRIX](governance/CODEX-COMPLIANCE-MATRIX.md) maps each
> section to concrete repository controls and CI/CD enforcement gates.

---

## **0. Economic Efficiency and Complexity Preamble**

All engineering decisions in this repository are governed by the principle that **code, tokens, and dependencies are capital allocations with ongoing cost**.

Correctness and performance are necessary but insufficient. Every change must also satisfy **economic efficiency and systemic simplicity**.

---

### **0.1 Code Justification (Revenue per LOC)**

Every line of code introduces persistent cost:

* Maintenance and debugging overhead
* Test surface expansion
* Security exposure
* CI/CD execution cost

Rules:

* New code must map to a **measurable outcome**: revenue generation, latency reduction, reliability improvement, or risk mitigation.
* Prefer **deletion over addition**. Removing unused or redundant code is a first-class engineering outcome.
* Redundant abstractions, speculative generalization, and AI-generated bulk code without justification are not permitted.
* Refactors must comply with **Behavioral Invariance (§1)** while reducing structural complexity.

PR Requirements:

* Explicit justification for net-new code
* Evidence that any increase in complexity is warranted by measurable benefit

---

### **0.2 Token Cost Accountability (Revenue per Token)**

All AI/LLM usage incurs direct operational cost and must be economically justified.

Rules:

* Every token must be **traceable** to a production feature and user value.
* Token usage must be bounded by:

  * Per-request limits
  * Per-feature budgets
  * Per-tenant quotas
* Non-production or unused inference paths must be eliminated.

Optimization Requirements:

* Prefer the smallest viable model for the task
* Enforce caching (prompt reuse, KV reuse where applicable)
* Eliminate redundant or duplicate calls

Kill Criteria:

* Any workflow with recurring token cost and no production usage must be removed.

---

### **0.3 Complexity Control**

System complexity is a multiplicative risk factor affecting:

* Failure modes and cascading errors
* Latency variability
* Debugging time
* Security surface area

Rules:

* Maintain strict adherence to **Bounded Contexts (§12)** and dependency directionality.
* Enforce **Dependency Governance (§16)** with zero tolerance for unused or unjustified dependencies.
* AI-assisted code must meet the same structural and quality constraints as human-written code.

Metrics:

* Track complexity indicators:

  * Dependency count
  * Module proliferation
  * LOC per feature
* Changes that increase system entropy without proportional benefit must be rejected.

---

### **0.4 Economic Review Requirements**

Every PR must include:

* **Cost Impact**: infrastructure, token usage, CI/CD overhead
* **Performance Impact**: latency, throughput, resource utilization
* **Business Outcome**: revenue, cost reduction, or risk mitigation

Absence of any of the above constitutes an incomplete change and must block review.

---

### **0.5 Subtractive Engineering Bias**

Engineering decisions must default to **minimization**:

* Favor simplification over expansion
* Collapse unnecessary layers and abstractions
* Remove dead code and unused dependencies proactively

“No new code” is a valid and often preferred outcome.

---

### **0.6 Enforcement and Governance**

This preamble is enforced through:

* **Deterministic CI/CD Gates (§11.1)**:

  * Complexity-aware diff checks (LOC, dependency changes)
  * Token usage validation in test environments
* **Observability Requirements (§11.4)**:

  * Cost per request
  * Token consumption per feature
  * Latency per dependency layer
* **Change Rationale (§19)**:

  * All structural or cost-impacting changes must include explicit economic justification

---

### **0.7 Principle**

Code, tokens, and complexity are **non-free resources**.

They are **capital investments** with expected return.

Any component that does not produce measurable value is a liability and must be removed.

---

## 1. Behavioral Invariance

Refactoring and structural changes must not introduce externally observable
behavior differences.  The externally observable surface includes:

* HTTP response shapes, status codes, and headers
* Event and message payloads
* Stored data schemas and query semantics
* SDK method signatures and return types
* Metric, log, and trace identifiers (see §10)

Automated contract tests and a release-diff review step are the enforcement
mechanism.  A PR that modifies internals must demonstrate invariance through
passing contract tests, not only through author assertion.

## 2. Test Pyramid Discipline

Every production code path must be covered at the appropriate layer:

| Layer | Scope | Minimum Coverage Expectation |
| --- | --- | --- |
| Unit | Pure logic, adapters, helpers | 80% line coverage |
| Integration | Service boundaries, adapter contracts | All public endpoints |
| Contract | External API / SDK / event schema | Regression on every change |
| E2E / Smoke | Critical user journeys | Happy path + one failure path |

Tests must be co-located with the code they cover and included in the same PR as
the change.  Tests that are disabled or skipped must carry a dated comment
explaining the reason and a tracking issue reference.

## 3. No Silent Behavior Drift

Changes introduced under the label of "refactoring", "cleanup", or "chore" are
subject to the same behavioral-invariance check as feature changes (see §1).
Reviewers must verify that no observable behavior was altered.  If a behavior
change is intentional and desirable, it must be explicitly described in the PR
body and released under a feature or fix label, not a refactor label.

---

## 10. Telemetry Identity Stability

Metric names, log field keys, and OpenTelemetry span names are part of the
public operational contract of the platform.  Once a telemetry identifier
reaches a production environment it must not be renamed or removed without a
deprecation period of at least one release cycle.

Rules:

* New identifiers must follow the canonical naming scheme documented in
  `docs/operations/TROUBLESHOOTING.md`.
* Renamed or removed identifiers require a deprecation comment, a CHANGELOG
  entry, and an updated alert/dashboard reference.
* Code review must explicitly inspect metric/log/span key changes.

---

## 11. Security and Quality Gates

### 11.1 Deterministic CI/CD Gates

Every merge to `main` / `master` must pass a fully deterministic gate set:

* Unit and integration test suite (no flaky skips)
* Linting and formatting checks
* Dependency resolution against pinned lock files
* Container image build (no cache-only validation)
* Deployment dry-run or plan output for infrastructure changes

Promotions to staging or production require audited, gated pipeline stages
defined in `.github/workflows/ci-cd-pipeline.yml`.  Manual hotfixes bypassing
CI are not permitted; use an expedited pipeline run instead.

### 11.2 Security Scanning

Every PR must complete the following scans without suppressed findings
(exceptions require a dated waiver with a tracking issue):

| Scan | Tooling | Workflow |
| --- | --- | --- |
| SAST | CodeQL | `codeql.yml` |
| Secrets detection | GitHub secret scanning + pre-commit hooks | `security-scanning.yml` |
| Dependency / supply-chain | OWASP Dependency-Check, Dependabot | `dependency-check.yml` |
| Container image | Trivy or equivalent | `container-security.yml` |
| DAST (staging) | OWASP ZAP | `dast-scanning.yml` |

Findings at `CRITICAL` or `HIGH` severity block merge.  `MEDIUM` findings must
have a remediation plan within 14 days.

### 11.3 Infrastructure Security

IaC changes must pass `iac-security.yml` policy checks (OPA/Rego + tfsec or
cfn-nag).  No IAM wildcard policies, no public S3 buckets, no unencrypted
storage resources are permitted without an explicit waiver.

### 11.4 Observability Requirements

All new service entry points must emit:

* A structured log entry (JSON, with `trace_id`, `span_id`, `request_id`)
* A latency metric (histogram preferred)
* A success/failure counter
* An OpenTelemetry span tied to the canonical service name

Runbooks and alert definitions for new SLO-backed paths must be included in the
same PR as the code change.

---

## 12. Bounded Contexts and Dependency Directionality

The repository is partitioned into bounded contexts described in
`docs/architecture/BOUNDED-CONTEXTS.md`.  Dependency rules:

* Outer layers (commercial shell, amplification) may depend on inner layers
  (trust, core product center).
* Inner layers must not import from outer layers.
* Cross-context calls must traverse a defined interface (HTTP, event, or SDK),
  never a direct in-process import across context boundaries.

New top-level modules require a documented context assignment before merge.
Violations are detected by the module-ownership linter referenced in
`docs/architecture/MODULE-OWNERSHIP.md`.

---

## 15. Interface and Persistence Contracts

### 15.1 Public Contract Compatibility

Public APIs (REST, GraphQL, gRPC, SDK), event schemas, and CLI interfaces are
versioned.  Backward-incompatible changes require:

1. A new version segment (`/v2/`, `v2.QueryRequest`, etc.).
2. A deprecation notice in the previous version with a sunset date.
3. Contract test coverage for both the old and new versions during the overlap
   window.
4. A CHANGELOG entry under `### Breaking Changes`.

### 15.2 Safe Persistence Evolution

Database schema and index changes must be:

* Delivered as incremental, reversible migrations.
* Validated in CI against the current schema snapshot.
* Deployed in an expand–contract sequence for zero-downtime changes.
* Documented in the migration file header with rationale and rollback steps.

---

## 16. Dependency Governance

All runtime dependencies must be declared in a canonical manifest
(`requirements.txt` / `pyproject.toml`, `package.json`, Helm `Chart.yaml`) and
pinned in a lock file (`requirements.lock`, `package-lock.json`, etc.).

Rules:

* CI must consume only the canonical manifests and their lock files.
* New dependencies require a security advisory check (GitHub Advisory Database)
  before merge.
* Direct dependencies with known `CRITICAL` CVEs are not permitted.
* Transitive vulnerabilities at `HIGH` or above require a waiver or an upgrade
  within 14 days.
* Unused dependencies must be removed promptly; dependency bloat is a hygiene
  violation.

---

## 18. Repository Hygiene

Only the following artifact categories are versioned:

| Category | Examples |
| --- | --- |
| Production source code | `backend/`, `frontend/`, `sdk/` |
| Tests | `backend/tests/`, `frontend/cypress/` |
| Infrastructure-as-Code | `infrastructure/`, `monitoring/` |
| Build and deploy configuration | `Makefile`, `docker-compose*.yml`, `.github/` |
| Required documentation | `docs/`, `README.md`, `CONTRIBUTING.md` |

The following must not be committed:

* Compiled artifacts, bytecode (`__pycache__/`, `*.pyc`, `dist/`, `build/`)
* Installed dependency trees (`node_modules/`, `.venv/`)
* Local IDE or OS metadata (`.DS_Store`, `.idea/` unless covered by `.gitignore`)
* Temporary experiment files, scratch notebooks, or ephemeral debug outputs

`.gitignore` and `.cursorignore` enforce these rules.  Any artifact that does
not fit the permitted categories requires explicit governance approval before
being added.

---

## 19. Change Rationale and Documentation

Every PR that introduces a structural change (new module, deleted module,
renamed interface, altered data model, changed deployment topology) must include:

* A written rationale explaining why the change is necessary.
* A description of the behavior that is preserved unchanged.
* Updated documentation reflecting the new structure (ADR, architecture diagram,
  module ownership table, or inline docstring as appropriate).

"Rationale" is not optional for structural changes; a PR that lacks it must be
sent back for revision before review can proceed.

---

## 20. Agentic Execution Constraints

When automated agents (coding assistants, bots, autonomous scripts) make changes
to this repository, the following constraints apply:

### 20.1 Small Bounded Cycles

Agent work must be broken into wave-level tasks with explicit checkpoints.  Each
checkpoint produces a verifiable artifact (passing tests, a committed diff, a
passing CI run).  No single agent cycle should span more than one logical change
unit.

### 20.2 Human Approval Before Destructive Operations

Before any of the following, explicit human approval is required:

* Deleting or renaming versioned files
* Modifying CI/CD pipeline definitions
* Changing IAM policies, security groups, or encryption settings
* Promoting to staging or production environments
* Running migrations against a live database

### 20.3 Stop on Failures and Preserve Diagnostics

If an agent encounters a failing test, a linter error, a security scan finding,
or a CI gate failure it must:

1. Stop further changes.
2. Preserve the full diagnostic output (logs, test output, scan report).
3. Surface the failure to a human reviewer before continuing.

Agents must not auto-resolve failures by suppressing checks or deleting failing
tests.

## 22. Concurrency, Parallelism, and Performance Engineering

### 22.1 Concurrency Model Discipline

All services must explicitly define their concurrency model:

Thread-based, async/event-loop, or hybrid
Blocking operations must be isolated or eliminated
Shared state must be minimized; prefer immutability or message passing

Rules:

No unbounded thread creation
No blocking I/O in async execution paths
Synchronization primitives must be justified and minimal

### 22.2 Parallelism and Throughput Optimization

Systems must maximize throughput under bounded resources.

Patterns:

Horizontal fan-out for independent workloads
Batching for compute-bound operations
Partitioning/sharding for high-cardinality workloads

Rules:

Parallelism must be bounded and backpressure-aware
Fan-out operations must define aggregation and failure semantics
No uncontrolled concurrency amplification (e.g., retry storms)

### 22.3 Latency Engineering

Latency must be treated as a first-class SLO.

Requirements:

Define latency budgets per service and per dependency
Eliminate unnecessary hops and serialization layers
Prefer local computation over remote calls when cost-effective

Techniques:

Caching (data, computation, inference)
Precomputation and lazy evaluation
Short-circuiting and early exits

### 22.4 Backpressure and Load Regulation

Systems must degrade gracefully under load.

Rules:

All ingress points must enforce rate limits
Queues must have bounded capacity
Retry mechanisms must use exponential backoff with jitter

Failure to regulate load is a system design violation.

### 22.5 Algorithmic Efficiency and Recursion Discipline

Algorithmic complexity must be explicitly considered.

Rules:

Time and space complexity must be stated for non-trivial logic
Recursive implementations must:
Have explicit termination conditions
Avoid stack overflow (use tail recursion or iterative fallback where required)
Prefer O(n) or better unless justified

### 22.6 Resource Efficiency

All compute paths must optimize:

CPU utilization
Memory footprint
Network bandwidth

Rules:

Avoid redundant allocations and copies
Use streaming over buffering for large payloads
Co-locate compute with data when possible

### 22.7 Performance Validation

All changes impacting execution paths must include:

Benchmark results (before vs after)
Throughput and latency impact
Resource utilization delta

CI/CD must include:

Load test or micro-benchmark validation for critical paths
Regression detection on latency and throughput
