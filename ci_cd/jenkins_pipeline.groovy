pipeline {
  agent any

  options {
    timestamps()
    ansiColor('xterm')
    disableConcurrentBuilds()
  }

  environment {
    PYTHON = 'python3'
    PIP_CACHE_DIR = "${WORKSPACE}/.pip-cache"
    FORENSIC_HMAC_SECRET = 'ci-secret'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        sh '''
          set -e
          ${PYTHON} -m venv .venv
          . .venv/bin/activate
          python -m pip install --upgrade pip
          pip config set global.cache-dir "$PIP_CACHE_DIR" || true
          pip install -r requirements.txt
          pip install ruff black
        '''
      }
    }

    stage('Lint') {
      steps {
        sh '''
          . .venv/bin/activate
          ruff check . || true
          black --check . || true
        '''
      }
    }

    stage('Tests with Coverage') {
      steps {
        sh '''
          set -e
          . .venv/bin/activate
          mkdir -p reports forensics
          pytest --maxfail=1 --disable-warnings -q \
            --cov=tools --cov-report=term-missing \
            --cov-report=xml:reports/coverage.xml \
            --junitxml=reports/junit.xml \
            --cov-fail-under=87
        '''
      }
    }

    stage('Hygiene Gates') {
      steps {
        sh '''
          set -e
          . .venv/bin/activate
          python - <<'PY'
          import json
          from tools.cache_tracer import CacheTracer, ForensicLogger
          t = CacheTracer()
          h = t.allocate(tenant_id="ci", request_id="ci-1", model_id="m", shape=(128,), dtype="float32", device="cpu", framework="numpy")
          t.mark_in_use(h)
          cov = t.sanitize(h, async_=False, verify=True)
          try:
              t.free(h)
          except Exception:
              pass
          t.export_metrics("forensics/coverage.json")
          res = ForensicLogger.verify_chain("forensics/kv_cache.log")
          print("Coverage:", cov)
          print("Forensic chain:", json.dumps(res))
          PY

          . .venv/bin/activate
          python tools/eviction_checker.py forensics/coverage.json --coverage-min 99.9 --unsanitized-max 0 --quarantine-max 0 --out forensics/verdict.json
        '''
      }
    }
  }

  post {
    always {
      junit allowEmptyResults: true, testResults: 'reports/junit.xml'
      publishCoverage adapters: [coberturaAdapter('reports/coverage.xml')], failNoReports: true, sourceDirectories: ['tools']
      archiveArtifacts artifacts: 'reports/**, forensics/**', onlyIfSuccessful: false, allowEmptyArchive: true
    }
  }
}
