# shellcheck shell=bash source=../lib.sh
# Dedicated placement — on yardrat this deliberately exercises the display-GPU
# limit (F5): with only GPU 0 placeable, a 2nd dedicated group can't place. We
# record the behavior so the report shows whether F5 still bites (and whether
# the failure is graceful).
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip placement-dedicated 'GPU serving disabled (run with --gpu)'
    exit 0
fi

step placement-dedicated 'a 2nd dedicated group on a 1-GPU box cannot place (F5)'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner a --require-generation --timeout 1200"
expect_rc 0
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner b --dedicated --require-generation --timeout 45 --json"
note "dedicated acquire rc=$RC (expected non-zero / not-ready on a 1-GPU box)"
run 'infer-stack leases --json'
expect_no_out 'Traceback (most recent call last)'
note 'inspect groups above: the dedicated group should be requested but not live'
# cleanup
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
