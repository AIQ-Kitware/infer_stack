# shellcheck shell=bash source=../lib.sh
# `infer-stack gc`: reclaim leaked (TTL-expired) leases and free their GPUs — the
# standalone backstop a blocking `acquire --queue` performs implicitly on each
# retry. A hard-killed job never runs `release`, so its lease lingers until its
# TTL elapses; gc sweeps expired leases and tears down any stop-policy deployment
# left with no demand. Plain gc leaves a healthy idle keep-warm model resident;
# `--evict` drops idle keep-warm too (like `evict --all`).
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip gc-noop-before-ttl 'GPU serving disabled (run with --gpu)'
    skip gc-reclaims-after-ttl 'GPU serving disabled (run with --gpu)'
    skip gc-keepwarm-survives 'GPU serving disabled (run with --gpu)'
    exit 0
fi

GCENV="$E2E_RESULTS/gc.env"
KWENV="$E2E_RESULTS/gc-keepwarm.env"

# A "leaked" lease: acquire with a short TTL and never release (simulating a
# hard-killed job). gc must be a no-op until the TTL elapses.
step gc-noop-before-ttl 'gc does nothing while a short-TTL lease still protects its group'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner crashed --ttl 120s --require-generation \
      --env-file \"$GCENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
run "infer-stack gc --backend compose --catalog \"$E2E_CAT\" --yes"
expect_rc 0
expect_out 'nothing to reclaim'
run 'infer-stack leases --json'
expect_out '"state": "active"'
note 'before the TTL, the lease protects its group: gc correctly reclaims nothing'
end_step

step gc-reclaims-after-ttl 'after the TTL elapses gc reclaims the leaked lease and frees the GPU'
note 'sleeping 130s to cross the 120s soft TTL...'
run 'sleep 130'
run "infer-stack gc --backend compose --catalog \"$E2E_CAT\" --yes"
expect_rc 0
expect_out 'gc: reclaimed'
expect_no_out 'nothing to reclaim'
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
note 'gc swept the expired lease and tore down the orphaned stop-policy group'
end_step

# Keep-warm survives a plain gc (it is a healthy idle model, not leaked demand);
# only `--evict` tears it down.
step gc-keepwarm-survives 'plain gc leaves an idle keep-warm model; --evict tears it down'
run "infer-stack acquire smol-360 --backend compose --catalog \"$E2E_CAT\" \
      --owner kw --require-generation --env-file \"$KWENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
# Graceful release -> idle, but smol-360 is reclaim:keep-warm so it stays resident.
run "infer-stack release --env-file \"$KWENV\""
expect_rc 0
# Plain gc must NOT disturb a healthy idle keep-warm model.
run "infer-stack gc --backend compose --catalog \"$E2E_CAT\" --yes"
expect_rc 0
expect_out 'nothing to reclaim'
note 'plain gc left the idle keep-warm group resident'
# --evict tears down idle keep-warm too.
run "infer-stack gc --backend compose --catalog \"$E2E_CAT\" --evict --yes"
expect_rc 0
expect_out 'gc: reclaimed'
expect_no_out 'nothing to reclaim'
note 'with --evict the idle keep-warm group is torn down'
# cleanup any stragglers
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
