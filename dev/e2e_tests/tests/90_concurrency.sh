# shellcheck shell=bash source=../lib.sh
# Concurrency / "last render wins" guard: two acquires race to converge the same
# shared compose file. The file lock must serialize them — no corrupt/half-
# written docker-compose.yml, no dropped service.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip concurrency-race 'GPU serving disabled (run with --gpu)'
    skip concurrency-compose-valid 'GPU serving disabled (run with --gpu)'
    exit 0
fi

step concurrency-race 'two concurrent acquires both land, demand is correct'
run "( infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
        --owner u1 --require-generation --timeout 1200 ) & \
     ( infer-stack acquire smol-135-dup   --backend compose --catalog \"$E2E_CAT\" \
        --owner u2 --require-generation --timeout 1200 ) & \
     wait"
expect_rc 0
run 'infer-stack leases --json'
expect_out '"demand": 2'
count_out '"state": "active"' 2
end_step

step concurrency-compose-valid 'the shared compose file is schema-valid (not half-written)'
COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"
run "docker compose -p infer-stack -f \"$COMPOSE\" config -q"
expect_rc 0
expect_no_out 'error'
# cleanup
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
