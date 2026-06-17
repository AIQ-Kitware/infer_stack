# shellcheck shell=bash source=../lib.sh
# Always-run teardown: release any stragglers and down the compose project so a
# run never leaks containers. Honors E2E_KEEP_RUNNING=1 to leave the stack up.
source "$E2E_ROOT/lib.sh"

step cleanup 'release stragglers and down the leasing compose project'
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin).get(\"leases\",[]) if l[\"state\"]==\"active\"]' 2>/dev/null | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
if [ "${E2E_KEEP_RUNNING:-0}" = "1" ]; then
    note 'E2E_KEEP_RUNNING=1 -> leaving any compose stack up'
else
    COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"
    run "[ -f \"$COMPOSE\" ] && docker compose -p infer-stack -f \"$COMPOSE\" down --remove-orphans 2>&1 || echo 'no compose project to down'"
fi
end_step
