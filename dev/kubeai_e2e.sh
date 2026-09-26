#!/usr/bin/env bash
# End-to-end smoke test for the kubeai backend against a REAL cluster.
#
# Prereqs (once): a cluster + the KubeAI chart. On a single GPU host:
#   ./scripts/bootstrap_k3s.sh
#   printf 'resourceProfiles:\n  %s:\n    runtimeClassName: nvidia\n    requests: {nvidia.com/gpu: "1"}\n    limits: {nvidia.com/gpu: "1"}\n' \
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
#   E2E_SIZED              1: check min_vram_gib picks a profile by GPU size,
#                          on two nodes with fake GPU labels. Needs
#                          dev/k3s_agent_container.sh up (E2E_SIZED_NODE names
#                          the node; default k3s-agent-b).
#   E2E_CLUSTER_GATEWAY    1 (default): finish with the gateway in the cluster
#                          (kubeai_gateway cluster): a NodePort, doctor,
#                          secrets rotate, stack down.
#   E2E_REMOTE_NODE        1 (with E2E_SIZED=1): also serve a Model on the
#                          second node through that gateway (pulls the vLLM
#                          CPU image inside the node container once).
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
  if [ "${SIZED_LABELLED:-0}" = 1 ]; then       # the fake GPU labels, off again
    for node in "$NODE_A" "$NODE_B"; do
      kubectl label node "$node" nvidia.com/gpu.product- nvidia.com/gpu.memory- >/dev/null 2>&1
    done
    kubectl patch node "$NODE_B" --subresource=status --type=json \
      -p '[{"op":"remove","path":"/status/capacity/nvidia.com~1gpu"}]' >/dev/null 2>&1
  fi
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
  # Into a file first: `grep -q` quits at its match, and under pipefail the
  # writer's SIGPIPE would fail the check.
  run_is acquire e2e-big --yes --timeout "$TIMEOUT" \
      --env-file "$WORK/big.env" > "$WORK/big.log" 2>&1 || true
  if ! grep -q 'ready: True' "$WORK/big.log"; then
    echo '!! the leased model never became ready' >&2; exit 1
  fi
  grep -q 'making room for leased demand' "$WORK/big.log" \
    || { echo '!! no idle model was evicted to make room' >&2; exit 1; }
  echo '   the idle keep-warm model was evicted; the leased one is ready'
  run_is release --env-file "$WORK/big.env" --yes
fi

if [ "${E2E_SIZED:-0}" = 1 ]; then
  # Needs a second node (dev/k3s_agent_container.sh up) and the sized-*
  # profiles of dev/e2e_tests/kubeai-cpu-values.yaml. Fake GPU labels stand
  # in for GPU Feature Discovery's; cleanup removes them.
  echo '== min_vram_gib picks the smallest resource profile whose GPUs fit'
  NODE_A=$(kubectl get nodes -l node-role.kubernetes.io/control-plane -o jsonpath='{.items[0].metadata.name}')
  NODE_B="${E2E_SIZED_NODE:-k3s-agent-b}"
  kubectl label node "$NODE_A" nvidia.com/gpu.product=FAKE-24G nvidia.com/gpu.memory=24576 --overwrite >/dev/null
  kubectl label node "$NODE_B" nvidia.com/gpu.product=FAKE-80G nvidia.com/gpu.memory=81920 --overwrite >/dev/null
  kubectl patch node "$NODE_B" --subresource=status --type=merge \
    -p '{"status":{"capacity":{"nvidia.com/gpu":"2"}}}' >/dev/null
  SIZED_LABELLED=1
  cat >> "$WORK/config/catalog.yaml" <<EOF
  e2e-sized-big:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: stop}
    placement: {min_vram_gib: 40}
    runtime: {max_model_len: 2048}
  e2e-sized-small:
    engine: vllm
    model: e2e-tiny
    reclaim: {policy: stop}
    placement: {min_vram_gib: 10}
    runtime: {max_model_len: 2048}
EOF
  where() {   # <endpoint> -> "<resourceProfile> <node>" of its Model's pod
    for _ in $(seq 60); do
      out=$(kubectl -n "$NAMESPACE" get models.kubeai.org -l infer-stack/managed=true -o json \
        | python3 -c '
import json, sys
for m in json.load(sys.stdin)["items"]:
    if sys.argv[1].replace("/", "-").lower() in m["metadata"]["name"]:
        print(m["metadata"]["name"], m["spec"]["resourceProfile"])' "$1")
      name=${out%% *}; profile=${out##* }
      node=$(kubectl -n "$NAMESPACE" get pods -l "model=$name" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
      [ -n "$node" ] && { echo "$profile $node"; return; }
      sleep 2
    done
    echo "$profile unscheduled"
  }
  run_is acquire e2e-sized-big --wait false --yes --env-file "$WORK/big-sized.env" >/dev/null
  got=$(where e2e-sized-big)
  [ "$got" = "sized-80g:1 $NODE_B" ] || { echo "!! 40 GiB went to: $got" >&2; exit 1; }
  echo "   min_vram_gib 40 -> $got"
  run_is acquire e2e-sized-small --yes --timeout "$TIMEOUT" --env-file "$WORK/small-sized.env" \
    > "$WORK/small-sized.log" 2>&1 || true
  grep -q 'ready: True' "$WORK/small-sized.log" \
    || { echo '!! the 10 GiB endpoint never became ready' >&2; exit 1; }
  got=$(where e2e-sized-small)
  [ "$got" = "sized-24g:1 $NODE_A" ] || { echo "!! 10 GiB went to: $got" >&2; exit 1; }
  echo "   min_vram_gib 10 -> $got, and it answers"
  run_is catalog suggest --backend kubeai > "$WORK/suggest.yaml" 2> "$WORK/suggest.err"
  grep -q 'nvidia-fake-80g' "$WORK/suggest.err" \
    || { cat "$WORK/suggest.err" >&2; echo '!! suggest proposed no profile for FAKE-80G' >&2; exit 1; }
  echo '   catalog suggest proposes a resource profile per GPU product'
  run_is release --env-file "$WORK/big-sized.env" --yes >/dev/null
  run_is release --env-file "$WORK/small-sized.env" --yes >/dev/null
fi

if [ "${GATEWAY:-1}" = 1 ] && [ "${E2E_DYNAMIC:-1}" = 1 ]; then
  echo '== dynamic routing: two --dedicated leases on one model, two Models, one alias'
  run_is stack down >/dev/null 2>&1          # the gateway comes back with Postgres
  # A routing-mode change is adopted only by a quiescent stack: wait for the
  # last managed pod (a released Model's) to be gone.
  for _ in $(seq 90); do
    [ -z "$(kubectl -n "$NAMESPACE" get pods -l infer-stack/managed=true -o name)" ] && break
    sleep 2
  done
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

if [ "${GATEWAY:-1}" = 1 ] && [ "${E2E_CLUSTER_GATEWAY:-1}" = 1 ]; then
  echo '== the gateway inside the cluster: a card reaches the Model on a NodePort'
  run_is stack down >/dev/null 2>&1            # the host gateway, and any Models
  for _ in $(seq 90); do
    [ -z "$(kubectl -n "$NAMESPACE" get pods -l infer-stack/managed=true -o name)" ] && break
    sleep 2
  done
  run_is config set dynamic_routing false >/dev/null
  run_is config set kubeai_gateway cluster
  run_is acquire "$ALIAS" --yes --ttl 30m --timeout "$TIMEOUT" \
      --env-file "$WORK/cluster.env" > "$WORK/cluster.log" 2>&1 || true
  grep -q 'ready: True' "$WORK/cluster.log" \
    || { tail -20 "$WORK/cluster.log" >&2; echo '!! not ready through the in-cluster gateway' >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$WORK/cluster.env"
  case "$OPENAI_BASE_URL" in
    *:"${E2E_NODE_PORT:-30442}"/v1) ;;
    *) echo "!! the env file points at $OPENAI_BASE_URL, not a NodePort" >&2; exit 1 ;;
  esac
  ask() {   # <model> -> a generation through the env file's gateway
    curl -sS --fail-with-body "$OPENAI_BASE_URL/chat/completions" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$1\", \"max_tokens\": 4,
           \"messages\": [{\"role\": \"user\", \"content\": \"say ok\"}]}" > "$WORK/ask.json" \
      && grep -q choices "$WORK/ask.json"
  }
  ask "$ALIAS" || { echo "!! no generation via $OPENAI_BASE_URL" >&2; exit 1; }
  echo "   generation ok via $OPENAI_BASE_URL (a node's NodePort, no port-forward)"
  run_is doctor > "$WORK/doctor.log" 2>&1 || { cat "$WORK/doctor.log" >&2; exit 1; }
  grep -q '\[ok  \] in-cluster gateway' "$WORK/doctor.log" \
    || { cat "$WORK/doctor.log" >&2; echo '!! doctor did not check the gateway' >&2; exit 1; }
  echo '   doctor checks the in-cluster gateway'
  if [ "${E2E_SIZED:-0}" = 1 ] && [ "${E2E_REMOTE_NODE:-0}" = 1 ]; then
    run_is acquire e2e-sized-big --yes --timeout "$TIMEOUT" \
        --env-file "$WORK/remote.env" > "$WORK/remote.log" 2>&1 || true
    grep -q 'ready: True' "$WORK/remote.log" \
      || { tail -20 "$WORK/remote.log" >&2; echo '!! the Model on the second node never became ready' >&2; exit 1; }
    got=$(where e2e-sized-big)
    [ "$got" = "sized-80g:1 $NODE_B" ] || { echo "!! the remote Model is at: $got" >&2; exit 1; }
    ask e2e-sized-big || { echo '!! no generation from the second node' >&2; exit 1; }
    echo "   a Model on $NODE_B answers through the same gateway"
    run_is release --env-file "$WORK/remote.env" --yes >/dev/null
  fi
  run_is release --env-file "$WORK/cluster.env" --yes >/dev/null
  run_is secrets rotate > "$WORK/rotate.log" 2>&1 \
    || { cat "$WORK/rotate.log" >&2; echo '!! secrets rotate failed' >&2; exit 1; }
  grep -q 'new key accepted, old key rejected' "$WORK/rotate.log" \
    || { cat "$WORK/rotate.log" >&2; exit 1; }
  echo '   secrets rotate: new key accepted, old key rejected (a Secret and a rollout)'
  run_is stack down >/dev/null
  if kubectl -n "$NAMESPACE" get deployment infer-stack-gateway >/dev/null 2>&1; then
    echo '!! stack down left the gateway Deployment' >&2; exit 1
  fi
  echo '   stack down removes it'
fi

echo 'PASS: kubeai backend end-to-end lifecycle'
