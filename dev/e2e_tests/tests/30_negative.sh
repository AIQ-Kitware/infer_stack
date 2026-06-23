# shellcheck shell=bash source=../lib.sh
# Negative / edge cases — these must fail *cleanly* (friendly message, non-zero
# exit), never with a raw traceback. Dry-run friendly.
source "$E2E_ROOT/lib.sh"

step unknown-endpoint 'acquiring an unknown endpoint fails friendly'
run "infer-stack acquire nope --catalog \"$E2E_CAT\" --no-wait"
expect_rc_not 0
expect_no_out 'Traceback (most recent call last)'
expect_re 'unknown|not found|no such|nope'
end_step

step kubeai-unimplemented 'kubeai backend reports not-implemented, not a crash'
run "infer-stack acquire smol-135 --backend kubeai --catalog \"$E2E_CAT\" --no-wait"
expect_rc_not 0
expect_no_out 'Traceback (most recent call last)'
expect_re 'not implemented|kubeai'
end_step

step missing-catalog 'a missing catalog path fails friendly'
run 'infer-stack acquire smol-135 --catalog /no/such/catalog.yaml --no-wait'
expect_rc_not 0
expect_no_out 'Traceback (most recent call last)'
expect_re 'catalog not found|not found'
end_step

step no-names 'acquire with no endpoint names is rejected'
run "infer-stack acquire --catalog \"$E2E_CAT\" --no-wait"
expect_rc_not 0
expect_no_out 'Traceback (most recent call last)'
end_step
