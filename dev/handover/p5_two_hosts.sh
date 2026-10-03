#!/usr/bin/env bash
# P5 across two real machines: what a second node in a container cannot show.
#
# Run on the cluster's first node (kubeconfig working) once a second GPU
# workstation has joined it (docs/kubeai-backend.md, "Add a workstation"),
# with the NVIDIA device plugin and GPU Feature Discovery running and the
# KubeAI chart installed. It serves one small model on the second node's GPU
# for a few minutes: run it when that GPU is free.
#
# It checks, in order:
#   1. two Ready nodes, and the second reports GPUs with GFD's labels;
#   2. with the gateway in the cluster, an endpoint whose resource profile
#      selects the second node's GPU product lands there and answers through
#      the gateway: pod-to-pod traffic across the real network (flannel);
#   3. the NodePort answers on the second node's own address too, so a card
#      on either workstation uses the same env file;
#   4. `secrets rotate` (a Secret and a rollout) with the Model still served.
#
# Own config and data directories; nothing of this host's infer-stack state
# is touched. It adds one resource profile for the second node's GPU product
# to the chart's values and leaves it: that is what the node needs.
#
#   dev/handover/p5_two_hosts.sh [SECOND_NODE_NAME] 2>&1 | tee p5-two-hosts.log
set -euo pipefail

NODE_B="${1:-}"
MODEL="${E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NAMESPACE="${NAMESPACE:-kubeai}"
TIMEOUT="${E2E_TIMEOUT:-900}"
PORT="${NODE_PORT:-30442}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/infer-stack-p5.XXXXXX")"
IS_ENV="INFER_STACK_CONFIG_DIR=$WORK/config INFER_STACK_DATA_DIR=$WORK/data"
run_is() { env $IS_ENV infer-stack "$@"; }
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*" >&2; exit 1; }

cleanup() {
  set +e
  run_is release --all --yes >/dev/null 2>&1
  run_is stack down >/dev/null 2>&1
  echo "work dir kept for the report: $WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/config"

echo '== 1. two nodes, the second with GPUs'
if [ -z "$NODE_B" ]; then
  NODE_B=$(kubectl get nodes -l '!node-role.kubernetes.io/control-plane' \
    -o jsonpath='{.items[0].metadata.name}')
fi
[ -n "$NODE_B" ] || fail 'no second node (pass its name as the first argument)'
kubectl wait --for=condition=Ready "node/$NODE_B" --timeout=30s >/dev/null \
  || fail "$NODE_B is not Ready"
PRODUCT=$(kubectl get node "$NODE_B" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.product}')
GPUS=$(kubectl get node "$NODE_B" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
ADDR_B=$(kubectl get node "$NODE_B" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
[ -n "$PRODUCT" ] && [ "${GPUS:-0}" -gt 0 ] \
  || fail "$NODE_B reports no GPUs or no nvidia.com/gpu.product label"
pass "$NODE_B at $ADDR_B: $GPUS x $PRODUCT"

echo '== 2. a Model on the second node, through the in-cluster gateway'
profile="p5-$(echo "$PRODUCT" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-')"
cat > "$WORK/profile.yaml" <<EOF
resourceProfiles:
  $profile:
    imageName: nvidia-gpu
    runtimeClassName: nvidia
    requests: {nvidia.com/gpu: "1"}
    limits: {nvidia.com/gpu: "1"}
    nodeSelector:
      kubernetes.io/hostname: "$NODE_B"
EOF
version=$(helm list -n "$NAMESPACE" -o json | python3 -c \
  'import json,sys; print(json.load(sys.stdin)[0]["chart"].rsplit("-", 1)[1])')
helm upgrade kubeai kubeai/kubeai -n "$NAMESPACE" --version "$version" \
  --reuse-values -f "$WORK/profile.yaml" --wait >/dev/null
run_is config set backend kubeai >/dev/null
run_is config set kubeai_namespace "$NAMESPACE" >/dev/null
run_is config set kubeai_gateway cluster >/dev/null
cat > "$WORK/config/catalog.yaml" <<EOF
models:
  small: {source: hf://$MODEL}
endpoints:
  p5-remote:
    engine: vllm
    model: small
    reclaim: {policy: stop}
    runtime: {resource_profile: $profile, max_model_len: 4096}
EOF
run_is acquire p5-remote --yes --timeout "$TIMEOUT" --env-file "$WORK/lease.env" \
  > "$WORK/acquire.log" 2>&1 || true
grep -q 'ready: True' "$WORK/acquire.log" || { tail -30 "$WORK/acquire.log"; fail 'not ready'; }
node=$(kubectl -n "$NAMESPACE" get pods -l infer-stack/managed=true \
  -o jsonpath='{.items[0].spec.nodeName}')
[ "$node" = "$NODE_B" ] || fail "the Model runs on $node, not $NODE_B"
# shellcheck disable=SC1090
source "$WORK/lease.env"
ask() {   # <base url> -> a generation for p5-remote
  curl -sS --fail-with-body "$1/chat/completions" -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"model": "p5-remote", "max_tokens": 4, "messages": [{"role": "user", "content": "say ok"}]}' \
    > "$WORK/ask.json" && grep -q choices "$WORK/ask.json"
}
ask "$OPENAI_BASE_URL" || fail "no generation via $OPENAI_BASE_URL"
pass "the Model on $NODE_B answers via $OPENAI_BASE_URL"

echo '== 3. the NodePort answers on the second node too'
ask "http://$ADDR_B:$PORT/v1" || fail "no generation via $ADDR_B:$PORT (firewall on the NodePort?)"
pass "a card on $NODE_B can use http://$ADDR_B:$PORT/v1"

echo '== 4. secrets rotate across the cluster'
run_is secrets rotate --force > "$WORK/rotate.log" 2>&1 || { cat "$WORK/rotate.log"; fail 'rotate'; }
grep -q 'new key accepted, old key rejected' "$WORK/rotate.log" || { cat "$WORK/rotate.log"; fail 'rotate'; }
pass 'secrets rotate: new key accepted, old key rejected'

echo 'ALL PASS'
