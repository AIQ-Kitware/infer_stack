# shellcheck shell=bash source=../lib.sh
# Coalescing & demand: two leases on the same model (smol-135-dup aliases smol-135)
# must share ONE deployment group (demand 2) and ONE vllm container. This is the
# efficiency headline — no duplicate model per consumer.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip coalesce-acquire 'GPU serving disabled (run with --gpu)'
    skip coalesce-demand 'GPU serving disabled (run with --gpu)'
    skip coalesce-one-container 'GPU serving disabled (run with --gpu)'
    skip coalesce-release 'GPU serving disabled (run with --gpu)'
    exit 0
fi

step coalesce-acquire 'two owners acquire the same underlying model'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner alice --require-generation --timeout 1200"
expect_rc 0
run "infer-stack acquire smol-135-dup --backend compose --catalog \"$E2E_CAT\" \
      --owner bob --require-generation --timeout 1200"
expect_rc 0
end_step

step coalesce-demand 'one group with demand 2, two active leases'
run 'infer-stack leases --json'
expect_rc 0
expect_out '"demand": 2'
count_out '"state": "active"' 2
end_step

step coalesce-one-container 'exactly one live vllm group / running container'
run "infer-stack ps | grep -c vllm || true"
note "running vllm containers (expect 1): $(cat "$LAST_OUT_FILE")"
# Count only LIVE vllm groups — a reclaim:stop group from an earlier tier lingers
# in the ledger as state=stopped and must not count toward 'one container'.
run "infer-stack leases --json | python3 -c 'import json,sys; g=json.load(sys.stdin)[\"groups\"]; print(len([x for x in g if x[\"engine\"]==\"vllm\" and x[\"state\"]==\"live\"]))'"
expect_out '1'
end_step

step coalesce-release 'releasing both leases drops demand to zero'
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release"
expect_rc 0
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
end_step
