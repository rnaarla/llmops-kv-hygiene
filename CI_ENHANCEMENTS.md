# AI-Augmented CI/CD Pipeline - Enhancement Guide

## Overview

This document describes the AI and agentic enhancements added to the `llmops-kv-hygiene` CI/CD pipeline. These enhancements eliminate friction, leverage AI for intelligent automation, and improve release quality through self-healing mechanisms.

## 🚀 New Features

### 1. AI-Assisted Diagnostics (`ai_triage` job)

**Purpose**: Automatically analyze test failures, lint errors, and code quality issues using LLM-powered triage.

**Capabilities**:
- Parses pytest and linting logs to identify error clusters
- Groups similar failures by root cause
- Detects flaky/non-deterministic tests using frequency analysis
- Generates structured triage reports with confidence scores
- Opens automated fix PRs when confidence > 90%

**Configuration**:
```yaml
env:
  AI_MODEL: 'claude-3-5-sonnet-20241022'  # or 'gpt-4-turbo-preview'
  AI_MAX_TOKENS: '4096'
  AI_CONFIDENCE_THRESHOLD: '0.9'
```

**Secrets Required**:
- `AI_API_KEY`: Claude API key (from Anthropic) or OpenAI API key

**Outputs**:
- `ai_analysis/triage_report.json`: Structured error analysis
- `ai_analysis/error_report.json`: Raw error collection
- Job outputs: `fix_confidence`, `has_flaky_tests`, `triage_summary`

**Usage**: Runs automatically after lint, test, and coverage jobs, even if they fail.

---

### 2. Adaptive Quality Gating

**Purpose**: Replace static thresholds with dynamic, regression-based quality gates that adapt to historical baselines.

**How It Works**:
- Maintains rolling baseline of coverage % and vulnerability counts (last 10 runs)
- Fails builds only when metrics regress beyond configurable deltas
- Updates baseline automatically on successful main branch builds

**Configuration**:
```yaml
env:
  DELTA_COVERAGE: '5.0'        # Max coverage drop allowed: 5%
  DELTA_VULN_HIGH: '3'         # Max 3 new HIGH vulnerabilities
  DELTA_VULN_CRITICAL: '1'     # Max 1 new CRITICAL vulnerability
```

**Baseline Storage**: `ci_metrics/quality_metrics.json`

**Benefits**:
- Prevents false negatives from overly strict static thresholds
- Focuses attention on regressions, not absolute numbers
- Automatically adjusts to project maturity

**Example Logic**:
```
Current Coverage: 88%
Baseline Coverage: 90%
Delta: -2% (within 5% threshold) → ✅ PASS

Current HIGH vulns: 15
Baseline HIGH vulns: 10
Delta: +5 (exceeds 3 threshold) → ❌ FAIL
```

---

### 3. Context-Aware Test Selection (`generate-matrix` + `test` jobs)

**Purpose**: Execute only tests affected by code changes to reduce CI runtime.

**Capabilities**:
- Detects changed files via `git diff`
- Uses `pytest-testmon` for dependency-aware test selection
- Automatically switches to full test suite when:
  - More than `TEST_SELECTION_THRESHOLD` files changed
  - Critical dependency files modified (`.github/`, `requirements.txt`, etc.)

**Configuration**:
```yaml
env:
  TEST_SELECTION_ENABLED: 'true'
  TEST_SELECTION_THRESHOLD: '20'  # Number of changed files
```

**Benefits**:
- Faster feedback on PRs (runs only relevant tests)
- Full coverage maintained when dependency graph changes
- No manual intervention required

**Example**:
```
PR changes: tools/cache_tracer.py
Tests run: tests/test_kv_sanitization.py, tests/test_cache_tracer.py
Tests skipped: tests/test_eviction.py (no dependency)
```

---

### 4. PR Feedback Agent (`pr_feedback` job)

**Purpose**: Generate developer-friendly, AI-powered summaries of CI results and post as PR comments.

**Capabilities**:
- Combines lint, test, coverage, and vulnerability results
- Uses LLM to generate concise, actionable feedback
- Updates existing bot comment on subsequent runs
- Includes emojis, status indicators, and next steps

**Configuration**: Uses same `AI_API_KEY` as triage job

**Example Output**:
```markdown
## 🤖 CI Feedback

✅ **Build Status**: Passed with minor warnings

### Summary
- Coverage: 92.5% (+2.3% from baseline) 🎉
- Tests: All passing
- Vulnerabilities: 2 new HIGH (within threshold)

### Action Items
1. Consider adding tests for `tools/eviction_checker.py:45-67` (low coverage)
2. Review HIGH vulnerabilities in `requirements.txt` dependencies

Great work on improving coverage! 🚀
```

---

### 5. Observability & Metrics Emission (`ci_metrics` job)

**Purpose**: Capture CI performance KPIs and emit to observability platforms.

**Metrics Collected**:
- Build duration and queue time
- Rerun count (flake rate indicator)
- Coverage percentage
- Vulnerability counts
- AI fix confidence scores
- Job statuses

**Supported Platforms**:
- **Datadog**: Real-time metrics via API
- **Prometheus**: Push gateway integration
- **GitHub Artifacts**: JSON metrics for analysis

**Configuration**:
```yaml
secrets:
  DATADOG_API_KEY: ${{ secrets.DATADOG_API_KEY }}
  PROMETHEUS_GATEWAY: ${{ secrets.PROMETHEUS_GATEWAY }}
```

**Daily Insights** (scheduled runs only):
- AI-generated trend analysis
- KPI status reports
- Optimization recommendations
- Developer satisfaction indicators
- Output: `ci_insights.md` artifact

**Metrics Schema**:
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "workflow_run_id": "12345",
  "metrics": {
    "coverage_pct": 92.5,
    "ai_fix_confidence": 0.85,
    "has_flaky_tests": false,
    "rerun_count": 0
  },
  "job_statuses": {
    "lint": "success",
    "test": "success",
    "coverage": "success"
  }
}
```

---

### 6. Self-Healing Agent (`auto_fix` job)

**Purpose**: Automatically generate and submit fix PRs for high-confidence issues.

**Trigger Conditions**:
- AI triage confidence > 90% (configurable via `AI_CONFIDENCE_THRESHOLD`)
- Previous triage job completed successfully

**Capabilities**:
- Requests detailed fix implementation from AI
- Applies code changes to new branch
- Commits with descriptive messages
- Creates PR with:
  - Confidence score
  - Fix explanation
  - Testing recommendations
  - `autofix-bot` and `needs-review` labels

**Safety Features**:
- Only triggers on high-confidence fixes (>0.9)
- All PRs require manual review before merge
- Clear labeling identifies automated changes

**Example Scenario**:
```
1. Test failure detected: ImportError in test_kv_isolation.py
2. AI triage identifies: Missing import statement (confidence: 0.95)
3. Auto-fix job:
   - Creates branch: autofix-20250115-103045
   - Adds: from tools.cache_tracer import CacheTracer
   - Commits: "autofix: Add missing CacheTracer import"
   - Opens PR with explanation and test recommendations
4. Developer reviews and merges (or requests changes)
```

---

### 7. Architectural Dependency Scan (`architecture_scan` job)

**Purpose**: Analyze codebase architecture, detect circular dependencies, and identify complexity hotspots.

**Tools Used**:
- **CodeQL**: Security and quality analysis
- **pydeps**: Python dependency graph generation
- **networkx**: Graph analysis algorithms

**Capabilities**:
- Detects circular import chains
- Identifies tightly coupled modules
- Measures complexity metrics
- Generates AI-powered architecture health reports

**Outputs**:
- CodeQL SARIF results (uploaded to Security tab)
- `architecture_analysis.json`: Dependency graph data
- `architecture_report.md`: AI-generated recommendations

**Example Report Section**:
```markdown
## Architecture Health Score: 8.5/10

### Critical Issues
- Circular dependency: tools/cache_tracer.py ↔ tools/eviction_checker.py
- High coupling: 15 modules depend on tools/activation_logger.py

### Refactoring Priorities
1. Break circular dependency by extracting shared interfaces
2. Consider dependency injection for activation_logger
3. Split cache_tracer.py into smaller, focused modules
```

---

## 🔧 Setup Instructions

### 1. Configure Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

```bash
# Required for AI features
AI_API_KEY=<your-claude-api-key>  # From console.anthropic.com

# Optional for observability
DATADOG_API_KEY=<your-datadog-api-key>
PROMETHEUS_GATEWAY=<your-prometheus-gateway-url>
```

### 2. Initialize Baseline Metrics

The baseline file is pre-created at `ci_metrics/quality_metrics.json`. It will auto-update on main branch builds.

### 3. Enable Vulnerability Gating (Optional)

Set repository/organization environment variable:
```yaml
VULN_GATING=1
```

Adjust thresholds:
```yaml
MAX_HIGH=10           # Absolute limit
MAX_CRITICAL=2        # Absolute limit
DELTA_VULN_HIGH=3     # Regression limit
DELTA_VULN_CRITICAL=1 # Regression limit
```

### 4. Test the Pipeline

Create a test PR to verify:
```bash
git checkout -b test-ai-pipeline
echo "# Test" >> README.md
git add README.md
git commit -m "test: trigger AI pipeline"
git push origin test-ai-pipeline
gh pr create --title "Test AI Pipeline" --body "Testing new features"
```

Expected behavior:
- ✅ Context-aware test selection (selective mode for small change)
- ✅ Coverage adaptive gate checks delta against baseline
- ✅ AI triage analyzes any errors (or reports "no errors")
- ✅ PR feedback comment appears on PR
- ✅ Metrics collected and uploaded as artifact

---

## 📊 Measurable KPIs

| KPI | Measurement | Target |
|-----|-------------|--------|
| **Build Time** | Workflow duration (start to finish) | <15 min for PRs, <30 min for main |
| **Flaky Test Rate** | Reruns per successful build | <5% |
| **Vulnerability MTTR** | Time from detection to fix merge | <7 days for HIGH, <3 days for CRITICAL |
| **Developer Satisfaction** | Survey + AI analysis of PR feedback sentiment | >4/5 stars |
| **Coverage Trend** | Rolling 30-day average | Increasing or stable >90% |
| **Auto-Fix Accuracy** | % of auto-fix PRs merged without changes | >70% |

---

## 🛠️ Troubleshooting

### AI API Calls Failing

**Symptoms**: Jobs show `⚠️ AI API call failed` or `⚠️ AI_API_KEY not set`

**Solutions**:
1. Verify `AI_API_KEY` secret is set in repository settings
2. Check API key validity at console.anthropic.com
3. Review rate limits (Claude: 50 req/min on paid tier)
4. Check workflow logs for specific error messages

### Adaptive Gates Failing Unexpectedly

**Symptoms**: Jobs fail with "regression exceeds delta threshold" on first run

**Solutions**:
1. Check `ci_metrics/quality_metrics.json` exists and has valid baseline
2. First run may use default baseline (90% coverage, 0 vulns)
3. Wait for 1-2 successful main branch builds to establish accurate baseline
4. Adjust `DELTA_*` thresholds if too strict for your project

### Test Selection Not Working

**Symptoms**: Full test suite runs on every PR

**Solutions**:
1. Verify `TEST_SELECTION_ENABLED='true'` in workflow env
2. Check if changed files include critical patterns (`.github/`, `requirements.txt`)
3. Ensure `pytest-testmon` is in `requirements.txt`
4. Review `generate-matrix` job output for mode selection logic

### Auto-Fix PRs Not Created

**Symptoms**: `auto_fix` job skipped even with test failures

**Solutions**:
1. Check triage confidence: must be >0.9 (see `ai_triage` job output)
2. Verify `GITHUB_TOKEN` has permissions to create PRs
3. Ensure GitHub CLI (`gh`) is available in runner (default on ubuntu-latest)
4. Review `ai_triage` artifacts to see fix suggestions and confidence scores

---

## 🎯 Best Practices

### 1. Review Auto-Fix PRs Promptly
- Automated PRs have `autofix-bot` label
- AI confidence scores are high but not infallible
- Review code changes before merging
- Provide feedback to improve future fixes

### 2. Monitor Baseline Drift
- Check `ci_metrics/quality_metrics.json` weekly
- Investigate unexpected baseline changes
- Commit baseline updates to version control

### 3. Tune Delta Thresholds
- Start conservative (current defaults)
- Adjust based on project volatility
- Stricter for mature/stable codebases
- More lenient for rapid development phases

### 4. Leverage Daily Insights
- Review `ci_insights.md` from scheduled runs
- Identify trends before they become problems
- Share with team during sprint planning

### 5. Observability Integration
- Set up Datadog/Prometheus dashboards
- Alert on high flake rates or coverage drops
- Track MTTR for vulnerabilities

---

## 🔄 Migration from Original Pipeline

All original features are **fully retained**:

✅ Matrix testing and code coverage (90% threshold)
✅ Python linting, type checking, caching
✅ Trivy/Grype scanning and SBOM attestation
✅ Docker multi-arch builds (CPU + CUDA)
✅ Cosign signing and provenance
✅ Cache forensics and coverage aggregation
✅ Scheduled daily and PR triggers

**New jobs are additive** and designed to fail gracefully if:
- AI API keys not configured (warnings only)
- Observability platforms not available
- Baseline metrics not yet established

**No breaking changes** to existing workflows.

---

## 📚 Additional Resources

- [Claude API Documentation](https://docs.anthropic.com/claude/reference)
- [GitHub Actions Contexts](https://docs.github.com/en/actions/learn-github-actions/contexts)
- [pytest-testmon](https://github.com/tarpas/pytest-testmon)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Datadog CI Visibility](https://docs.datadoghq.com/continuous_integration/)

---

## 🤝 Contributing

To extend AI features:

1. Add new analysis capabilities to `ai_triage` job
2. Enhance prompts for better triage accuracy
3. Add new metrics to `ci_metrics` job
4. Integrate additional observability platforms
5. Improve auto-fix code generation logic

Pull requests welcome! Tag with `enhancement` and `ci-pipeline`.

---

**Version**: 1.0.0
**Last Updated**: 2025-01-15
**Maintainer**: CI/CD Team
