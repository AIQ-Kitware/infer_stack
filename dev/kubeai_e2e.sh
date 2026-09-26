#!/usr/bin/env bash
# End-to-end smoke test for the kubeai backend against a REAL cluster.
#
# Prereqs (once): a cluster + the KubeAI chart. On a single GPU host:
#   ./scripts/bootstrap_k3s.sh
#   printf 'resourceProfiles:\n  %s:\n    limits:\n      nvidia.com/gpu: "1"\n' \
#       "${E2E_RESOURCE_PROFILE:-nvidia-gpu}" > /tmp/kubeai-values.yaml
#   ./scripts/install_kubeai.sh /tmp/kubeai-values.yaml kubeai
#   kubectl -n kubeai port-forward svc/kubeai 8000:80 &
#
# Without a GPU (verified on k3s): install the chart with
# dev/e2e_tests/kubeai-cpu-values.yaml, whose `cpu` profile runs real vLLM on
# CPU (needs AVX-512), and run with E2E_RESOURCE_PROFILE=cpu.
#
# Then:  ./dev/kubeai_e2e.sh
#
# Knobs (env):
#   E2E_MODEL              hf model id  (default Qwen/Qwen2.5-0.5B-Instruct)
#   E2E_RESOURCE_PROFILE   resourceProfiles key (default nvidia-gpu)
#   E2E_NAMESPACE          chart namespace (default kubeai)
#   E2E_BASE_URL           gateway url (default http://127.0.0.1:8000/openai/v1)
#   E2E_TIMEOUT            acquire readiness budget seconds (default 900 —
#                          first run pulls model weights into the cluster)
#   GATEWAY                1 (default): clients go through infer-stack's LiteLLM
#                          gateway, as on the compose backend. 0: straight to
#                          KubeAI, where the alias below does NOT route (404).
#   E2E_DYNAMIC            1 (default): finish with dynamic routing, two
#                          --dedicated leases of one model (two Models at once).
#   E2E_UI_PORT            Open WebUI's port on this host (default 13000).
set -euo pipefail

MODEL="${E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROFILE="${E2E_RESOURCE_PROFILE:-nvidia-gpu}"
NAMESPACE="${E2E_NAMESPACE:-kubeai}"
BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000/openai/v1}"
TIMEOUT="${E2E_TIMEOUT:-900}"
ALIAS="$MODEL"

# Isolated config/data roots, baked INLINE on every command (never rely on
# exported env reaching subprocesses — see dev/audit notes on tmux env loss).
WORK="$(mktemp -d /tmp/infer-stack-kubeai-e2e.XXXXXX)"
IS_ENV="INFER_STACK_CONFIG_DIR=$WORK/config INFER_STACK_DATA_DIR=$WORK/data"
run_is() { env $IS_ENV infer-stack "$@"; }

echo "== work dir: $WORK"
cleanup() {
  status=$?
  set +e
  run_is release --all --yes >/dev/null 2>&1
  run_is gc --evict --yes >/dev/null 2>&1
  run_is stack down >/dev/null 2>&1          # the gateway, UI and database too
  # nothing managed may remain on the cluster, pass or fail
  leftover=$(kubectl -n "$NAMESPACE" get models.kubeai.org \
      -l infer-stack/managed=true -o name 2>/dev/null | wc -l)
  if [ "$leftover" != 0 ]; then
    echo "!! $leftover managed Model(s) left on the cluster" >&2
    kubectl -n "$NAMESPACE" get models.kubeai.org -l infer-stack/managed=true
    status=1
  fi
  rm -rf "$WORK"
  exit $status
}
trap cleanup EXIT

mkdir -p "$WORK/config"
cat > "$WORK/config/catalog.yaml" <<EOF
models:
  e2e-tiny:
    source: hf://$MODEL
endpoints:
  # Named like real catalogs, unlike its KubeAI Model name: a card sends this.
  $ALIAS:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: stop}
    runtime:
      resource_profile: $PROFILE
      max_model_len: 2048
EOF

run_is config set backend kubeai
run_is config set kubeai_namespace "$NAMESPACE"
run_is config set kubeai_base_url "$BASE_URL"
if [ "${GATEWAY:-1}" = 0 ]; then run_is config set litellm false; fi

echo '== doctor (preflight)'
run_is doctor

echo '== acquire (readiness = a real generation through the gateway)'
run_is acquire "$ALIAS" --yes --ttl 30m --timeout "$TIMEOUT" \
    --env-file "$WORK/lease.env"

echo '== a client that knows only the env file and the alias (like a card)'
# shellcheck disable=SC1090
source "$WORK/lease.env"
# An explicit `if`: in a `curl | grep && echo` list `set -e` does not fire, so
# a 404 used to fall through to PASS.
if curl -sS --fail-with-body "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer ${OPENAI_API_KEY:-EMPTY}" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$ALIAS\", \"max_tokens\": 8,
         \"messages\": [{\"role\": \"user\", \"content\": \"say ok\"}]}" \
  | grep -q 'choices'; then
  echo "   generation ok via $OPENAI_BASE_URL"
else
  echo "!! no generation for model=$ALIAS via $OPENAI_BASE_URL" >&2
  exit 1
fi

if [ "${GATEWAY:-1}" = 1 ]; then
  echo '== the gateway fronts the cluster as on compose: routes, Open WebUI'
  if ! run_is routes list --json | python3 -c '
import json, sys
alias = sys.argv[1]
row = next(r for r in json.load(sys.stdin)["routes"] if r["name"] == alias)
assert row["engine"] == "upstream" and row["live"], row
print("   routes list:", alias, "->", row["target"], "at", row["upstream"])' "$ALIAS"; then
    echo '!! routes list does not route the alias to the cluster' >&2; exit 1
  fi
  ui=''
  for _ in $(seq 60); do
    ui=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${E2E_UI_PORT:-13000}/" || true)
    [ "$ui" = 200 ] && break
    sleep 2
  done
  [ "$ui" = 200 ] || { echo "!! Open WebUI did not answer (HTTP $ui)" >&2; exit 1; }
  echo '   Open WebUI answers in front of the cluster'
fi

echo '== release prunes the Model (reclaim: stop)'
run_is release --env-file "$WORK/lease.env" --yes
remaining=$(kubectl -n "$NAMESPACE" get models.kubeai.org \
    -l infer-stack/managed=true -o name | wc -l)
[ "$remaining" = 0 ] || { echo "!! model not pruned"; exit 1; }

echo '== an unrenderable endpoint is refused before anything is written'
# No resource_profile and no kubeai_resource_profile default: admission's
# preview refuses it, so no lease is committed and kubectl applies nothing;
# --queue fails at once, since waiting cannot make it renderable.
cat >> "$WORK/config/catalog.yaml" <<EOF
  e2e-noprofile:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: stop}
EOF
if run_is acquire e2e-noprofile --yes --timeout 60 --queue \
    --env-file "$WORK/bad.env" > "$WORK/bad.log" 2>&1; then
  echo '!! an endpoint with no resource profile was admitted' >&2; exit 1
fi
grep -q 'resource profile' "$WORK/bad.log" \
  || { cat "$WORK/bad.log" >&2; echo '!! the refusal did not name the cause' >&2; exit 1; }
active=$(run_is leases --json | python3 -c \
  'import json,sys; print(sum(le["state"] == "active" for le in json.load(sys.stdin)["leases"]))')
[ "$active" = 0 ] || { echo "!! $active lease(s) left active" >&2; exit 1; }
remaining=$(kubectl -n "$NAMESPACE" get models.kubeai.org \
    -l infer-stack/managed=true -o name | wc -l)
[ "$remaining" = 0 ] || { echo '!! a Model was applied for it' >&2; exit 1; }
echo '   refused at admission: no lease, no Model'

if [ "${E2E_MAKE_ROOM:-0}" = 1 ]; then
  # Needs a profile only one Model fits at a time (`cpu-half` in
  # dev/e2e_tests/kubeai-cpu-values.yaml): E2E_ROOM_PROFILE=cpu-half.
  echo '== an idle keep-warm model gives way to a leased one'
  cat >> "$WORK/config/catalog.yaml" <<EOF
  e2e-warm:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: keep-warm}
    runtime: {resource_profile: ${E2E_ROOM_PROFILE:-cpu-half}, max_model_len: 2048}
  e2e-big:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: stop}
    runtime: {resource_profile: ${E2E_ROOM_PROFILE:-cpu-half}, max_model_len: 2048}
EOF
  run_is acquire e2e-warm --yes --timeout "$TIMEOUT" --env-file "$WORK/warm.env"
  run_is release --env-file "$WORK/warm.env" --yes     # idle, still resident
  if ! run_is acquire e2e-big --yes --timeout "$TIMEOUT" \
        --env-file "$WORK/big.env" 2>&1 | tee "$WORK/big.log" | grep -q 'ready: True'; then
    echo '!! the leased model never became ready' >&2; exit 1
  fi
  grep -q 'making room for leased demand' "$WORK/big.log" \
    || { echo '!! no idle model was evicted to make room' >&2; exit 1; }
  echo '   the idle keep-warm model was evicted; the leased one is ready'
  run_is release --env-file "$WORK/big.env" --yes
fi

if [ "${GATEWAY:-1}" = 1 ] && [ "${E2E_DYNAMIC:-1}" = 1 ]; then
  echo '== dynamic routing: two --dedicated leases on one model, two Models, one alias'
  run_is stack down >/dev/null 2>&1          # the gateway comes back with Postgres
  run_is config set dynamic_routing true
  for n in 1 2; do
    run_is acquire "$ALIAS" --dedicated --yes --ttl 30m --timeout "$TIMEOUT" \
        --env-file "$WORK/dyn$n.env"
  done
  models=$(kubectl -n "$NAMESPACE" get models.kubeai.org \
      -l infer-stack/managed=true -o name | wc -l)
  [ "$models" = 2 ] || { echo "!! $models Model(s), expected 2" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$WORK/dyn1.env"
  routes=$(curl -s "$OPENAI_BASE_URL/model/info" -H "Authorization: Bearer $OPENAI_API_KEY" \
    | python3 -c 'import json,sys; print(sum(m["model_name"] == sys.argv[1] for m in json.load(sys.stdin)["data"]))' "$ALIAS")
  [ "$routes" = 2 ] || { echo "!! $routes route(s) for $ALIAS, expected 2" >&2; exit 1; }
  if curl -sS --fail-with-body "$OPENAI_BASE_URL/chat/completions" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$ALIAS\", \"max_tokens\": 4,
           \"messages\": [{\"role\": \"user\", \"content\": \"say ok\"}]}" \
    | grep -q 'choices'; then
    echo '   two Models, two routes under one alias, and it answers'
  else
    echo "!! no generation for $ALIAS under dynamic routing" >&2; exit 1
  fi
  run_is release --env-file "$WORK/dyn1.env" --yes
  run_is release --env-file "$WORK/dyn2.env" --yes
fi

echo 'PASS: kubeai backend end-to-end lifecycle'
