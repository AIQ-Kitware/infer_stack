# shellcheck shell=bash source=../lib.sh
# The whole acquire -> leases -> env-file -> release surface on the null backend.
# No docker, no GPU: this isolates the ledger/CLI wiring from serving.
source "$E2E_ROOT/lib.sh"

ENVF="$E2E_RESULTS/dryrun.env"

step acquire-nowait 'acquire smol-135 (null backend, --no-wait, JSON)'
run "infer-stack acquire smol-135 --catalog \"$E2E_CAT\" \
      --env-file \"$ENVF\" --no-wait --json"
expect_rc 0
expect_out '"lease_id"'
expect_out 'smol-135'
end_step

step envfile 'env-file is sourceable and carries the endpoint contract'
run "cat \"$ENVF\""
expect_rc 0
expect_file_has "$ENVF" 'export INFER_STACK_LEASE_ID='
expect_file_has "$ENVF" 'INFER_STACK_ENDPOINT_SMOL_135=smol-135'
expect_file_has "$ENVF" 'INFER_STACK_MODELS=smol-135'
end_step

step leases-active 'leases shows one active lease and one live group (demand 1)'
run 'infer-stack leases --json'
expect_rc 0
expect_out '"state": "active"'
expect_out '"demand": 1'
end_step

step release 'release by env-file tears the lease down cleanly'
run "infer-stack release --env-file \"$ENVF\""
expect_rc 0
end_step

step leases-empty 'after release no active lease remains'
run 'infer-stack leases --json'
expect_rc 0
expect_no_out '"state": "active"'
end_step

step bundle 'acquiring a bundle resolves all its endpoints'
run "infer-stack acquire pair --catalog \"$E2E_CAT\" --no-wait --json"
expect_rc 0
expect_out 'smol-135'
expect_out 'smol-360'
end_step

# tidy the dry-run leases so they don't pollute later GPU tiers' ledger view
run 'infer-stack leases --json | python3 -c "import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"]]" | xargs -r -n1 infer-stack release >/dev/null 2>&1; true'
