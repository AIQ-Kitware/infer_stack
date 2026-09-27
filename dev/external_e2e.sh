#!/usr/bin/env bash
# End-to-end test for external endpoints against a REAL LiteLLM gateway.
#
# No GPU, no model weights: the compose backend runs only its front door, and
# the "external" server is infer-stack's own mock OpenAI server on this host,
# which accepts one bearer key and records every request. Runs on the guest
# VM (docker, the pinned litellm and postgres images).
#
#   ./dev/external_e2e.sh
#
# For each routing mode (static, then dynamic):
#   1. an external endpoint whose key has no value is refused before publishing;
#   2. `routes seed` publishes it and the gateway routes the alias to the mock,
#      sending the key from the managed .env by name (`os.environ/NAME`);
#   3. the key changes: `env` says an apply is needed, the old key is refused
#      by the upstream until `apply`, the new key is sent after it;
#   4. `routes prune` unpublishes it and the alias stops routing;
#   5. `access` of the external endpoint alone: no lease, no lease id in the
#      env-file, a client that knows only the env-file reaches it;
#   6. `access` of a managed endpoint alone (llm-d-inference-sim, on CPU):
#      one lease, the same env-file shape;
#   7. `access` of a mixed bundle: one lease for the managed member only;
#      `release --env-file` releases it and the external route stays;
#   8. the managed alias becomes external after release and eviction:
#      same alias, same workflow, no lease.
#
# Knobs (env):
#   E2E_MODES   routing modes to run (default "static dynamic")
#   E2E_PORT    the mock's port (default 18911)
#   E2E_SIM     the simulator image (default ghcr.io/llm-d/llm-d-inference-sim:v0.9.0)
set -euo pipefail

MODES="${E2E_MODES:-static dynamic}"
PORT="${E2E_PORT:-18911}"
SIM="${E2E_SIM:-ghcr.io/llm-d/llm-d-inference-sim:v0.9.0}"
HOST_IP="$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)"
[ -n "$HOST_IP" ] || { echo 'no docker0 address' >&2; exit 1; }

fail() { echo "FAIL: $*" >&2; exit 1; }

rm_work() {                     # Postgres owns its data dir as its own uid
    [ -d "$1/data" ] && docker run --rm -v "$1/data:/d" --entrypoint sh postgres:16.8 \
        -c 'rm -rf /d/postgres-litellm /d/postgres-open-webui' >/dev/null 2>&1
    rm -rf "$1"
}

MOCK_PID=''
start_mock() {                  # start_mock KEY
    stop_mock
    cat > "$WORK/mock.yaml" <<EOF
models:
  Upstream/Model: {ability: 0.9}
EOF
    python -m infer_stack mock serve --config_fpath "$WORK/mock.yaml" \
        --host 0.0.0.0 --port "$PORT" --api_key "$1" >"$WORK/mock.log" 2>&1 &
    MOCK_PID=$!
    for _ in $(seq 50); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0
        sleep 0.2
    done
    fail "mock server did not start: $(cat "$WORK/mock.log")"
}
stop_mock() {
    if [ -n "$MOCK_PID" ]; then kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; fi
    MOCK_PID=''
}

last_bearer() {                 # the Authorization the mock last received
    curl -sf "http://127.0.0.1:$PORT/__mock__/requests" | python -c '
import json, sys
reqs = [r for r in json.load(sys.stdin)["requests"]
        if r.get("path", "").endswith("/chat/completions")]
headers = {k.lower(): v for k, v in reqs[-1]["headers"].items()} if reqs else {}
print(headers.get("authorization", ""))'
}

chat() {                        # chat ALIAS -> http status; body in $WORK/chat.json
    local base key
    base="$(run_is env OPENAI_BASE_URL)"
    key="$(run_is env LITELLM_MASTER_KEY)"
    curl -s -o "$WORK/chat.json" -w '%{http_code}' "$base/chat/completions" \
        -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$1\", \"max_tokens\": 8,
             \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}" || true
}

chat_env() {                    # chat_env ENV_FILE ALIAS -> http status, as a card would
    (
        # shellcheck disable=SC1090
        source "$1"
        var="INFER_STACK_ENDPOINT_$(echo "$2" | tr '[:lower:]-' '[:upper:]_')"
        curl -s -o "$WORK/chat.json" -w '%{http_code}' "$OPENAI_BASE_URL/chat/completions" \
            -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
            -d "{\"model\": \"${!var}\", \"max_tokens\": 8,
                 \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}" || true
    )
}

leases_active() {               # how many ACTIVE leases the ledger holds
    run_is leases --json | python -c '
import json, sys
print(sum(1 for le in json.load(sys.stdin).get("leases", []) if le.get("state") == "active"))'
}

chat_until() {                  # chat_until ALIAS STATUS: routes settle after apply
    local got=''
    for _ in $(seq 60); do
        got="$(chat "$1")"
        [ "$got" = "$2" ] && return 0
        sleep 2
    done
    fail "chat $1: wanted HTTP $2, got $got: $(cat "$WORK/chat.json")"
}

run_mode() {
    local mode=$1
    WORK="$(mktemp -d /tmp/infer-stack-external-e2e.XXXXXX)"
    IS_ENV="INFER_STACK_CONFIG_DIR=$WORK/config INFER_STACK_DATA_DIR=$WORK/data"
    run_is() { env $IS_ENV infer-stack "$@"; }
    echo "== [$mode] work dir: $WORK"
    mkdir -p "$WORK/config"
    run_is config set backend compose >/dev/null
    if [ "$mode" = dynamic ]; then run_is config set dynamic_routing true >/dev/null; fi
    # The runbook's own catalog: nothing external.
    cat > "$WORK/config/catalog.yaml" <<EOF
models: {}
endpoints: {}
EOF
    cat > "$WORK/external.yaml" <<EOF
endpoints:
  remote:
    external:
      api_base: http://$HOST_IP:$PORT/v1
      model: Upstream/Model
      api_key_env: E2E_REMOTE_KEY
EOF
    start_mock key-one

    echo "== [$mode] 1. a key with no value is refused before publishing"
    if run_is routes seed "$WORK/external.yaml" --yes >"$WORK/out" 2>&1; then
        fail 'seed without the key succeeded'
    fi
    grep -q 'infer-stack env E2E_REMOTE_KEY=' "$WORK/out" || fail "$(cat "$WORK/out")"

    echo "== [$mode] 2. publish; the gateway sends the key by name"
    run_is env E2E_REMOTE_KEY=key-one >/dev/null
    run_is routes seed "$WORK/external.yaml" --yes
    run_is routes list
    run_is routes list --json | grep -q '"origin": "external"' || fail 'not listed as external'
    if grep -rq 'key-one' "$WORK/data/leasing/compose/docker-compose.yml"; then
        fail 'the key value is in the compose file'
    fi
    chat_until remote 200
    [ "$(last_bearer)" = 'Bearer key-one' ] || fail "sent $(last_bearer)"

    echo "== [$mode] 3. the key changes"
    start_mock key-two                              # the upstream rotated
    chat_until remote 401
    run_is env E2E_REMOTE_KEY=key-two | tee "$WORK/out"
    grep -q 'after `infer-stack apply`' "$WORK/out" || fail 'env did not say apply'
    run_is apply --yes
    chat_until remote 200
    [ "$(last_bearer)" = 'Bearer key-two' ] || fail "sent $(last_bearer)"

    echo "== [$mode] 4. prune unpublishes it"
    run_is routes prune --yes
    if run_is routes list --json | grep -q '"remote"'; then fail 'still listed after prune'; fi
    chat_until remote 400

    # The runbook's catalog now names a managed endpoint, the external one,
    # and a bundle of both.
    cat > "$WORK/config/catalog.yaml" <<EOF
models:
  smol: {source: hf://HuggingFaceTB/SmolLM2-135M-Instruct}
endpoints:
  local:
    engine: vllm
    model: smol
    runtime:
      image: $SIM
      max_model_len: 2048
      simulator: {kind: llm-d-sim, mode: echo}
  remote:
    external:
      api_base: http://$HOST_IP:$PORT/v1
      model: Upstream/Model
      api_key_env: E2E_REMOTE_KEY
bundles:
  pair: [local, remote]
EOF

    echo "== [$mode] 5. access an external endpoint: no lease"
    run_is access remote --yes --env-file "$WORK/remote.env"
    if grep -q INFER_STACK_LEASE_ID "$WORK/remote.env"; then fail 'a lease id without a lease'; fi
    [ "$(leases_active)" = 0 ] || fail 'external access took a lease'
    [ "$(chat_env "$WORK/remote.env" remote)" = 200 ] || fail "remote: $(cat "$WORK/chat.json")"
    run_is release --env-file "$WORK/remote.env" | grep -q 'nothing to release' \
        || fail 'release of a lease-less env-file'

    echo "== [$mode] 6. access a managed endpoint: one lease"
    run_is access local --yes --timeout 300 --env-file "$WORK/local.env"
    grep -q INFER_STACK_LEASE_ID "$WORK/local.env" || fail 'managed access without a lease id'
    [ "$(chat_env "$WORK/local.env" local)" = 200 ] || fail "local: $(cat "$WORK/chat.json")"
    run_is release --env-file "$WORK/local.env" --yes

    echo "== [$mode] 7. a mixed bundle: one lease, for the managed member"
    run_is access pair --yes --timeout 300 --env-file "$WORK/pair.env" --json > "$WORK/pair.json"
    python - "$WORK/pair.json" <<'PY' || fail "pair: $(cat "$WORK/pair.json")"
import json, sys
d = json.load(open(sys.argv[1]))
assert d['lease_id'] and d['external'] == ['remote'], d
assert set(d['descriptor']['endpoints']) == {'local', 'remote'}, d
PY
    [ "$(leases_active)" = 1 ] || fail 'a mixed bundle took other than one lease'
    [ "$(chat_env "$WORK/pair.env" local)" = 200 ] || fail "pair/local: $(cat "$WORK/chat.json")"
    [ "$(chat_env "$WORK/pair.env" remote)" = 200 ] || fail "pair/remote: $(cat "$WORK/chat.json")"
    run_is release --env-file "$WORK/pair.env" --yes
    [ "$(leases_active)" = 0 ] || fail 'the bundle lease is still held'
    [ "$(chat_env "$WORK/pair.env" remote)" = 200 ] || fail 'remote stopped routing after release'

    echo "== [$mode] 8. the managed alias moves to an external server"
    run_is evict --all --yes
    python - "$WORK/config/catalog.yaml" <<'PY'
import sys, yaml
path = sys.argv[1]
cat = yaml.safe_load(open(path))
cat['endpoints']['local'] = {'external': dict(cat['endpoints']['remote']['external'])}
yaml.safe_dump(cat, open(path, 'w'), sort_keys=False)
PY
    run_is access local --yes --env-file "$WORK/moved.env"
    if grep -q INFER_STACK_LEASE_ID "$WORK/moved.env"; then fail 'moved alias took a lease'; fi
    [ "$(chat_env "$WORK/moved.env" local)" = 200 ] || fail "moved: $(cat "$WORK/chat.json")"
    [ "$(last_bearer)" = 'Bearer key-two' ] || fail "moved sent $(last_bearer)"

    run_is stack down >/dev/null 2>&1 || true
    stop_mock
    rm_work "$WORK"
    echo "== [$mode] PASS"
}

cleanup() {
    status=$?
    set +e
    stop_mock
    if [ -n "${WORK:-}" ] && [ -d "$WORK" ]; then
        env $IS_ENV infer-stack stack down >/dev/null 2>&1
        rm_work "$WORK"
    fi
    exit $status
}
trap cleanup EXIT

for mode in $MODES; do run_mode "$mode"; done
echo '== external endpoints e2e: PASS'
