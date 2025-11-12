package cicd

default allow = true

deny[msg] {
  input.name == "ci.yml"
  not input.permissions
  msg := "workflow must define top-level permissions"
}

deny[msg] {
  input.name == "ci.yml"
  input.permissions.contents != "read"
  msg := "workflow contents permission must be read-only"
}

allowed_job_writes := {
  "auto_fix",
  "ci_metrics",
  "docker-images",
  "trivy-repo-scan",
  "docker-scan-pr",
}

deny[msg] {
  job_name := name
  job := input.jobs[name]
  job.permissions.contents == "write"
  not allowed_job_writes[job_name]
  msg := sprintf("job %s requests write permissions without approval", [job_name])
}
