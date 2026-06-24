# shellcheck shell=bash source=../lib.sh
# LiteLLM no-blip across a model swap (needs 2 GPUs). Bring up A + B on separate
# GPUs, hammer A continuously through the gateway, then release B and bring a
# THIRD model C up on B's freed GPU. With the static superset route table (every
# catalog endpoint is routed from the start), A's upstream and the LiteLLM
# gateway are never recreated as B goes down and C comes up — so the in-flight
# stream to A must see ZERO failed requests, and the litellm container id must be
# unchanged across the swap.
#
# Key prerequisite this also guards: the catalog-less `release` converge must
# still render the static superset, or it falls back to per-deployment routing
# and the gateway churns (a real blip). `release` has no --catalog flag, so it
# reads the DEFAULT path (config_root/catalog.yaml) — which we populate below.
# This mirrors a correctly configured deployment.
source "$E2E_ROOT/lib.sh"

AENV="$E2E_RESULTS/noblip-a.env"
BENV="$E2E_RESULTS/noblip-b.env"
CENV="$E2E_RESULTS/noblip-c.env"
COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"
POLL_STOP="$E2E_RESULTS/noblip-poll.stop"
POLL_RESULT="$E2E_RESULTS/noblip-poll.result"

if ! gpu_enabled; then
    skip noblip-two-models-up 'GPU serving disabled (run with --gpu)'
    skip noblip-stream-uninterrupted 'GPU serving disabled (run with --gpu)'
    skip noblip-cleanup 'GPU serving disabled (run with --gpu)'
    exit 0
fi
NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [ "${NGPU:-0}" -lt 2 ]; then
    skip noblip-two-models-up "needs 2 GPUs, found ${NGPU:-0}"
    skip noblip-stream-uninterrupted "needs 2 GPUs, found ${NGPU:-0}"
    skip noblip-cleanup "needs 2 GPUs, found ${NGPU:-0}"
    exit 0
fi

# Make the catalog discoverable on EVERY converge, including the no-`--catalog`
# release, so the gateway is rendered from the static superset throughout.
cp "$E2E_CAT" "$INFER_STACK_CONFIG_DIR/catalog.yaml"

step noblip-two-models-up 'two distinct models on two GPUs, both routable through the gateway'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner noblip-a --require-generation --env-file \"$AENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
run "infer-stack acquire smol-360 --backend compose --catalog \"$E2E_CAT\" \
      --owner noblip-b --require-generation --env-file \"$BENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
run "set -a; . \"$AENV\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/models\" -H \"Authorization: Bearer \$OPENAI_API_KEY\""
expect_out 'smol-135'
expect_out 'smol-360'
note 'A (smol-135) and B (smol-360) are live on separate GPUs'
end_step

step noblip-stream-uninterrupted 'querying A is uninterrupted while B is swapped for C'
LITELLM_BEFORE="$(docker compose -p infer-stack -f "$COMPOSE" ps -q litellm 2>/dev/null)"
note "litellm container before swap: ${LITELLM_BEFORE:-<none>}"
rm -f "$POLL_STOP" "$POLL_RESULT"
# Background poller: hammer A (smol-135) through the gateway until told to stop,
# counting any non-200 as an interruption. It is orphaned from this step's short
# `bash -c`, so it signals completion via a result file we poll (same pattern as
# the queue tier).
run "( set -a; . \"$AENV\"; set +a; total=0; failed=0; \
      while [ ! -f \"$POLL_STOP\" ]; do \
        total=\$((total+1)); \
        code=\$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
          \"\$OPENAI_BASE_URL/chat/completions\" \
          -H \"Authorization: Bearer \$OPENAI_API_KEY\" -H 'Content-Type: application/json' \
          -d '{\"model\":\"smol-135\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}'); \
        [ \"\$code\" = 200 ] || failed=\$((failed+1)); \
        sleep 0.25; \
      done; \
      echo \"total=\$total failed=\$failed\" > \"$POLL_RESULT\" ) & echo \"poller pid \$!\""
expect_rc 0
note 'poller is streaming requests to A; now swapping B -> C underneath it'
run 'sleep 5'   # let a few baseline requests land before the swap
# Swap: release B (frees its GPU), bring C up on the freed GPU. The release has
# no --catalog and relies on the default-path catalog placed above.
run "infer-stack release --env-file \"$BENV\""
expect_rc 0
run "infer-stack acquire qwen-500 --backend compose --catalog \"$E2E_CAT\" \
      --owner noblip-c --require-generation --env-file \"$CENV\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
note 'stop the poller and collect its tally'
run "touch \"$POLL_STOP\"; for _ in \$(seq 1 40); do [ -f \"$POLL_RESULT\" ] && break; sleep 1; done; \
     cat \"$POLL_RESULT\" 2>/dev/null || echo 'no result'"
expect_rc 0
expect_out 'failed=0'
LITELLM_AFTER="$(docker compose -p infer-stack -f "$COMPOSE" ps -q litellm 2>/dev/null)"
note "litellm container after swap: ${LITELLM_AFTER:-<none>}"
run "[ -n \"$LITELLM_BEFORE\" ] && [ \"$LITELLM_BEFORE\" = \"$LITELLM_AFTER\" ] \
       && echo 'gateway container unchanged across swap'"
expect_out 'gateway container unchanged across swap'
# C is up and A is still routable through the same gateway.
run "set -a; . \"$AENV\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/models\" -H \"Authorization: Bearer \$OPENAI_API_KEY\""
expect_out 'qwen-500'
expect_out 'smol-135'
end_step

step noblip-cleanup 'release the no-blip tier holders'
run "infer-stack release --env-file \"$AENV\" >/dev/null 2>&1; true"
run "infer-stack release --env-file \"$CENV\" >/dev/null 2>&1; true"
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
# Drop the default-path catalog we added so later runs aren't surprised by it.
run "rm -f \"$INFER_STACK_CONFIG_DIR/catalog.yaml\"; true"
end_step
