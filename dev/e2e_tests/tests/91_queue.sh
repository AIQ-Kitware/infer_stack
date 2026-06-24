# shellcheck shell=bash source=../lib.sh
# Admission queue (`acquire --queue` / wait_for_placement): when every placeable
# GPU is busy a queued acquire WAITS (up to --timeout) for one to free instead of
# failing fast, then lands; with nothing freeing it times out and rolls back
# (no phantom active lease). We pin placement to GPU 0 (--allowed-gpus 0) so the
# fleet is deterministically "full" after a single dedicated group, regardless of
# how many GPUs the host has — making the queue contention reproducible.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip queue-timeout-rolls-back 'GPU serving disabled (run with --gpu)'
    skip queue-blocks-then-lands 'GPU serving disabled (run with --gpu)'
    skip queue-cleanup 'GPU serving disabled (run with --gpu)'
    exit 0
fi

AENV="$E2E_RESULTS/queue-a.env"
BENV="$E2E_RESULTS/queue-b.env"
BOUT="$E2E_RESULTS/queue-b.out"
BRC="$E2E_RESULTS/queue-b.rc"

# A takes the only placeable GPU (dedicated -> exclusive), so the fleet is full.
step queue-timeout-rolls-back 'a queued acquire times out and rolls back when nothing frees'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner a --dedicated --allowed-gpus 0 --require-generation \
      --env-file \"$AENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
# A second dedicated group cannot share A's GPU. With --queue and a short timeout
# it must WAIT and then give up (non-zero), leaving no lingering active lease —
# the queue's bounded, rolled-back failure path.
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner d --dedicated --allowed-gpus 0 --queue --require-generation \
      --env-file \"$E2E_RESULTS/queue-d.env\" --timeout 40 --interval 5 --json"
expect_rc_not 0
expect_no_out 'Traceback (most recent call last)'
note 'queued acquire correctly failed after the placement timeout'
run 'infer-stack leases --json'
# Only A remains: the rolled-back attempt must not survive as an active lease.
count_out '"state": "active"' 1
end_step

# A still holds the only placeable GPU. A queued acquire should block, then land
# the moment A is released and the GPU frees.
step queue-blocks-then-lands 'a queued acquire waits for a busy GPU, then lands when it frees'
# Launch B queued in the background; it records its own exit code + output to
# files we poll (it is orphaned from this step's short-lived `bash -c`, so we
# cannot `wait` on it directly).
run "( infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
        --owner b --dedicated --allowed-gpus 0 --queue --require-generation \
        --env-file \"$BENV\" --timeout 900 --interval 5 --json \
        > \"$BOUT\" 2>&1; echo \$? > \"$BRC\" ) & echo \"started queued acquire pid \$!\""
expect_rc 0
note 'B is queued in the background; letting it confirm it cannot place yet'
run 'sleep 20'
# B must still be waiting: its rc file should not exist yet, and only A is active.
run "test ! -f \"$BRC\" && echo 'B still waiting'"
expect_out 'B still waiting'
run 'infer-stack leases --json'
count_out '"state": "active"' 1
note 'A active, B requested-but-waiting (queued behind the busy GPU)'
# Free the GPU: releasing A tears down its reclaim:stop group; B's next reconcile
# (every --interval seconds) then places it on the freed GPU.
run "infer-stack release --env-file \"$AENV\""
expect_rc 0
note 'waiting for the queued acquire to place on the freed GPU (polls B.rc, up to 600s)'
run "for _ in \$(seq 1 120); do [ -f \"$BRC\" ] && break; sleep 5; done; cat \"$BRC\" 2>/dev/null || echo TIMEOUT"
expect_out '0'
run "cat \"$BOUT\""
expect_out '"ready": true'
run 'infer-stack leases --json'
count_out '"state": "active"' 1
note 'the queued acquire landed once the GPU freed: B is now the sole active lease'
end_step

step queue-cleanup 'release the queue tier holders'
run "infer-stack release --env-file \"$BENV\" >/dev/null 2>&1; true"
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
end_step
