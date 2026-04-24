package main

# Conftest evaluates `deny` rules in package `main` against the input document.

default allow := true

deny contains msg if {
	not input.permissions
	msg := "workflow must define top-level permissions"
}

deny contains msg if {
	input.permissions.contents != "read"
	msg := "workflow contents permission must be read-only"
}

allowed_job_writes := {"update-metrics"}

deny contains msg if {
	some job_name
	job := input.jobs[job_name]
	job_perms := job.permissions
	job_perms.contents == "write"
	not job_name in allowed_job_writes
	msg := sprintf("job %s requests contents:write without policy approval", [job_name])
}
