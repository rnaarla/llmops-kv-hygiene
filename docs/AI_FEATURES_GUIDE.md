# AI Features Guide - llmops-kv-hygiene CI/CD

## Overview

This guide explains how to use the AI-powered features in the CI/CD pipeline, including automated error analysis, code repair, and intelligent quality gates.

## Features

### 1. AI-Assisted Error Triage

**What it does:** Automatically analyzes test failures, lint errors, and code quality issues using OpenAI GPT-4.

**How it works:**

- Runs after lint, test, and coverage jobs (even if they fail)
- Collects errors from test results and coverage reports
- Sends errors to OpenAI for analysis
- Groups similar failures by root cause
- Detects flaky test patterns
- Generates confidence scores for potential fixes

**Outputs:**

- `ai_analysis/error_report.json` - Raw error data
- `ai_analysis/triage_report.json` - AI analysis with fix recommendations
- Job outputs: `fix_confidence`, `has_flaky_tests`, `triage_summary`

### 2. Autonomous Code Repair (Auto-Fix)

**What it does:** Automatically generates code fixes and creates pull requests when confidence is high.

**Trigger conditions:**

- AI triage confidence > 90% (configurable)
- Previous triage job completed successfully

**What it creates:**

- New branch: `autofix-YYYYMMDD-HHMMSS`
- Commits with descriptive messages
- Pull request with:
  - Fix explanation
  - Confidence score
  - Labels: `autofix-bot`, `needs-review`

**Safety:**

- All PRs require manual review before merge
- Only triggers on high-confidence fixes
- Clear labeling identifies automated changes

### 3. Context-Aware Test Selection

**What it does:** Runs only tests affected by code changes to reduce CI runtime.

**How it works:**

- Detects changed files via git diff
- Automatically switches to full test suite when:
  - More than 20 files changed (configurable)
  - Critical files modified (`.github/`, `requirements.txt`, etc.)

**Benefits:**

- Faster PR feedback
- Full coverage maintained
- No manual intervention required

## Setup Instructions

### 1. Configure OpenAI API Key

**Get an API key:**

1. Go to <https://platform.openai.com/api-keys>
2. Create a new API key
3. Copy the key (starts with `sk-`)

**Add to GitHub:**

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `AI_API_KEY`
4. Value: Your OpenAI API key
5. Click "Add secret"

### 2. Configure AI Settings (Optional)

Adjust these in `.github/workflows/ci.yml`:

```yaml
env:
  AI_MODEL: 'gpt-4-turbo-preview'      # OpenAI model
  AI_MAX_TOKENS: '4096'                 # Max response length
  AI_CONFIDENCE_THRESHOLD: '0.9'        # Min confidence for auto-fix (0-1)
  TEST_SELECTION_ENABLED: 'true'        # Enable smart test selection
  TEST_SELECTION_THRESHOLD: '20'        # Files changed threshold
```

### 3. Test the Features

Create a test PR to verify:

```bash
# Create test branch
git checkout -b test-ai-features

# Make a small change
echo "# Test" >> README.md

# Commit and push
git add README.md
git commit -m "test: trigger AI pipeline"
git push origin test-ai-features

# Create PR
gh pr create --title "Test AI Features" --body "Testing AI triage"
```

**Expected behavior:**

- ✅ Context-aware test selection (selective mode for small change)
- ✅ AI triage analyzes any errors (or reports "no errors")
- ✅ If high-confidence fix needed, auto-fix PR created

## Using AI Features

### Reviewing Auto-Fix PRs

1. **Check for `autofix-bot` label** in PR list
2. **Review the changes** carefully
3. **Check confidence score** in PR description
4. **Test locally** if needed:

   ```bash
   gh pr checkout <PR-number>
   pytest
   ```

5. **Merge or request changes** based on quality

### Understanding Triage Reports

Download the triage report artifact:

1. Go to Actions → Select workflow run
2. Scroll to "Artifacts"
3. Download `ai-triage-report`
4. Extract and view `triage_report.json`

**Report structure:**

```json
{
  "clusters": [
    {
      "type": "ImportError",
      "count": 3,
      "root_cause": "Missing import statement"
    }
  ],
  "flaky_tests": ["test_concurrent_access"],
  "fixes": [
    {
      "confidence": 0.95,
      "description": "Add missing import",
      "code_change": "from tools.cache_tracer import CacheTracer"
    }
  ],
  "summary": "3 test failures due to missing import"
}
```

### Monitoring Test Selection

Check the `generate-matrix` job output:

- **Selective mode**: Only affected tests run
- **Full mode**: All tests run (critical files changed or threshold exceeded)

## Troubleshooting

### AI API Calls Failing

**Symptoms:** Jobs show `⚠️ AI API call failed` or `⚠️ AI_API_KEY not set`

**Solutions:**

1. Verify `AI_API_KEY` secret is set in repository settings
2. Check API key validity at <https://platform.openai.com/api-keys>
3. Review rate limits (OpenAI: tier-based limits)
4. Check workflow logs for specific error messages

### Auto-Fix PRs Not Created

**Symptoms:** `auto_fix` job skipped even with test failures

**Solutions:**

1. Check triage confidence: must be >0.9 (see `ai_triage` job output)
2. Verify `GITHUB_TOKEN` has permissions to create PRs
3. Review `ai_triage` artifacts to see fix suggestions

### Test Selection Not Working

**Symptoms:** Full test suite runs on every PR

**Solutions:**

1. Verify `TEST_SELECTION_ENABLED='true'` in workflow env
2. Check if changed files include critical patterns
3. Review `generate-matrix` job output for mode selection

## Best Practices

### 1. Review Auto-Fix PRs Promptly

- Automated PRs have `autofix-bot` label
- AI confidence scores are high but not infallible
- Review code changes before merging
- Provide feedback to improve future fixes

### 2. Monitor AI API Usage

- Track costs in OpenAI dashboard
- Set usage alerts
- Adjust `AI_MAX_TOKENS` if needed

### 3. Tune Confidence Threshold

- Start with 0.9 (current default)
- Increase to 0.95 if too many low-quality fixes
- Decrease to 0.85 if missing good fixes
- Review merged auto-fix PRs to calibrate

### 4. Leverage Triage Reports

- Download and review regularly
- Share with team during retrospectives
- Use to identify recurring issues
- Feed insights back into code quality

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MODEL` | `gpt-4-turbo-preview` | OpenAI model to use |
| `AI_MAX_TOKENS` | `4096` | Maximum response length |
| `AI_CONFIDENCE_THRESHOLD` | `0.9` | Min confidence for auto-fix (0-1) |
| `TEST_SELECTION_ENABLED` | `true` | Enable context-aware testing |
| `TEST_SELECTION_THRESHOLD` | `20` | Files changed before full suite |

### Required Secrets

| Secret | Description | Where to get it |
|--------|-------------|-----------------|
| `AI_API_KEY` | OpenAI API key | <https://platform.openai.com/api-keys> |

### Job Dependencies

```text
generate-matrix → test → ai-triage → auto-fix
                         ↓
                      coverage
```

## Cost Estimation

**OpenAI API costs (approximate):**

- Per triage analysis: ~$0.01-0.05 (depends on error count)
- Per auto-fix generation: ~$0.05-0.15 (depends on complexity)
- Typical PR: ~$0.10 total (if errors present)
- Daily scheduled run: ~$0.05 (minimal errors)

**Tips to reduce costs:**

1. Only run on PRs that modify code (not docs)
2. Use smaller context windows (`AI_MAX_TOKENS`)
3. Skip auto-fix for low-confidence issues
4. Monitor usage in OpenAI dashboard

## Examples

### Example 1: Simple Auto-Fix

**Scenario:** Missing import statement

**AI Triage Output:**

```json
{
  "fixes": [{
    "confidence": 0.95,
    "description": "Add missing CacheTracer import",
    "code_change": "from tools.cache_tracer import CacheTracer"
  }]
}
```

**Auto-Fix PR Created:**

- Branch: `autofix-20251112-103045`
- Commit: "autofix: Add missing CacheTracer import"
- Changes: Added import to `tests/test_kv_isolation.py`
- Status: ✅ All tests passing

### Example 2: Flaky Test Detection

**Scenario:** Test fails intermittently

**AI Triage Output:**

```json
{
  "flaky_tests": ["test_concurrent_access"],
  "summary": "Race condition in concurrent access test"
}
```

**Result:**

- No auto-fix created (confidence too low)
- Manual investigation required
- Report available in artifacts

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [GitHub Actions Contexts](https://docs.github.com/en/actions/learn-github-actions/contexts)
- [CI/CD Pipeline Overview](./NEW_WORKFLOW_SUMMARY.md)
- [Enhancement Details](./CI_ENHANCEMENTS.md)

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review workflow logs in GitHub Actions
3. Download and inspect `ai-triage-report` artifacts
4. Open an issue with `ci-pipeline` label

---

**Version:** 1.0.0
**Last Updated:** November 12, 2025
**Maintainer:** CI/CD Team
