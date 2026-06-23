#!/usr/bin/env bash
# Manually tear down the leasing stack and (optionally) wipe the ledger.
#
# Use after Ctrl-C'ing a run, or any time a dev box is left with leftover
# infer-stack containers / a wedged ledger. The leasing compose project is
# always named "infer-stack", so the container teardown works regardless of
# which run's data dir was in play.
#
#   ./cleanup.sh                       # down the 'infer-stack' compose project
#   ./cleanup.sh --wipe-ledger         # + wipe the ledger at $INFER_STACK_DATA_DIR
#   ./cleanup.sh --wipe-ledger DIR     # + wipe the ledger under DIR (a run's
#                                      #   results/<ts>/infer-stack-data)
set -uo pipefail

WIPE=0
DATA="${INFER_STACK_DATA_DIR:-}"
for arg in "$@"; do
    case "$arg" in
        --wipe-ledger) WIPE=1 ;;
        *) DATA="$arg" ;;
    esac
done

echo "== tearing down the 'infer-stack' compose project =="
# By project name first (works even if no compose file is around)...
docker compose -p infer-stack down --remove-orphans 2>&1 | sed 's/^/  /' || true
# ...and via the rendered file if we can find one (belt and suspenders).
if [ -n "$DATA" ]; then
    cf="$DATA/leasing/compose/docker-compose.yml"
    [ -f "$cf" ] && docker compose -p infer-stack -f "$cf" down \
        --remove-orphans 2>&1 | sed 's/^/  /' || true
fi

# Any stragglers labeled as ours (e.g. left by a hard kill mid-converge).
stragglers="$(docker ps -aq --filter 'label=com.docker.compose.project=infer-stack' 2>/dev/null)"
if [ -n "$stragglers" ]; then
    echo "== removing straggler containers =="
    echo "$stragglers" | xargs -r docker rm -f 2>&1 | sed 's/^/  /' || true
fi

if [ "$WIPE" = 1 ]; then
    if [ -z "$DATA" ]; then
        echo "!! --wipe-ledger needs INFER_STACK_DATA_DIR set or a DIR argument"
        exit 2
    fi
    echo "== wiping ledger under $DATA/leasing =="
    rm -f "$DATA"/leasing/ledger.db \
          "$DATA"/leasing/ledger.db-wal \
          "$DATA"/leasing/ledger.db-shm 2>/dev/null || true
fi

echo "done."
