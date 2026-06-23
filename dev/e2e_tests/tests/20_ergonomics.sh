# shellcheck shell=bash source=../lib.sh
# The day-2 / discovery ergonomics that must survive the leasing refactor:
# paths, secrets, status, and the day-2 compose wrappers' graceful behavior.
source "$E2E_ROOT/lib.sh"

step paths-toplevel 'infer-stack paths surfaces the leasing artifacts'
run 'infer-stack paths'
expect_rc 0
expect_out 'config:'
expect_out 'leasing:'
end_step

step paths-leasing 'config paths leasing points at ledger / compose / secrets'
run 'infer-stack config paths leasing'
expect_rc 0
expect_out 'ledger'
expect_out 'docker-compose.yml'
expect_out 'env (secrets)'
end_step

step status 'status runs cleanly and prints its (leasing-aware) summary'
run 'infer-stack status'
expect_rc 0
expect_no_out 'Traceback (most recent call last)'
expect_out 'backend:'
# Leasing-first: config.yaml is now reported as `legacy config:`, and a user
# with a catalog/settings is not nagged to run the legacy setup.
expect_no_out 'Not initialized'
# The one-line leasing summary only appears when the ledger is non-empty
# (by design); that path is covered by the unit test
# test_status_summarizes_leases_when_present. Here we just smoke `status`.
end_step

step env-missing 'env gives a friendly error before any compose acquire'
run 'infer-stack env LITELLM_MASTER_KEY'
expect_rc_not 0
expect_out 'no managed env-file'
note 'expected on a fresh data dir; becomes populated after a compose acquire'
end_step

step day2-fallback 'day-2 ps degrades gracefully with no deployment + no config'
run 'infer-stack ps'
# No leasing compose yet and no legacy config.yaml -> should fail, not crash
# with a traceback. We only assert it does not emit a Python traceback.
expect_no_out 'Traceback (most recent call last)'
note "rc=$RC (a non-zero 'no config' style error is expected here)"
end_step
