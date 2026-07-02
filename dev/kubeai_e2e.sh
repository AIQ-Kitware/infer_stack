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
# Then:  ./dev/kubeai_e2e.sh
#
# Knobs (env):
#   E2E_MODEL              hf model id  (default Qwen/Qwen2.5-0.5B-Instruct)
#   E2E_RESOURCE_PROFILE   resourceProfiles key (default nvidia-gpu)
#   E2E_NAMESPACE          chart namespace (default kubeai)
#   E2E_BASE_URL           gateway url (default http://127.0.0.1:8000/openai/v1)
#   E2E_TIMEOUT            acquire readiness budget seconds (default 900 —
#                          first run pulls model weights into the cluster)
set -euo pipefail

MODEL="${E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROFILE="${E2E_RESOURCE_PROFILE:-nvidia-gpu}"
NAMESPACE="${E2E_NAMESPACE:-kubeai}"
BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000/openai/v1}"
TIMEOUT="${E2E_TIMEOUT:-900}"

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
  e2e-tiny:
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

echo '== doctor (preflight)'
run_is doctor

echo '== acquire (readiness = a real generation through the gateway)'
run_is acquire e2e-tiny --yes --ttl 30m --timeout "$TIMEOUT" \
    --env-file "$WORK/lease.env"

echo '== the descriptor works for a plain OpenAI client'
# shellcheck disable=SC1090
source "$WORK/lease.env"
REQUEST_NAME="$INFER_STACK_ENDPOINT_E2E_TINY"
curl -sf "$OPENAI_BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$REQUEST_NAME\", \"max_tokens\": 8,
         \"messages\": [{\"role\": \"user\", \"content\": \"say ok\"}]}" \
  | grep -q 'choices' && echo '   generation ok'

echo '== release prunes the Model (reclaim: stop)'
run_is release --env-file "$WORK/lease.env" --yes
remaining=$(kubectl -n "$NAMESPACE" get models.kubeai.org \
    -l infer-stack/managed=true -o name | wc -l)
[ "$remaining" = 0 ] || { echo "!! model not pruned"; exit 1; }

echo 'PASS: kubeai backend end-to-end lifecycle'
