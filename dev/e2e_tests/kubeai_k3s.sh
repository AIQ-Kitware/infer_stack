#!/usr/bin/env bash
# End-to-end check of the kubeai backend against a real (GPU-less) cluster.
#
# Needs: k3s (or any cluster) with KubeAI installed from
# dev/e2e_tests/kubeai-cpu-values.yaml, KUBECONFIG pointing at it, and the
# KubeAI gateway reachable at KUBEAI_BASE_URL (default: a port-forward,
# `kubectl -n kubeai port-forward svc/kubeai 8000:80`). The CPU vLLM image
# needs AVX-512.
#
# Runs in a throwaway config/data dir, so it never touches a real setup:
#   bash dev/e2e_tests/kubeai_k3s.sh
set -euo pipefail

WORK=${WORK:-$(mktemp -d -t infer-stack-kubeai-XXXX)}
export INFER_STACK_CONFIG_DIR=$WORK/cfg INFER_STACK_DATA_DIR=$WORK/data
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
ENDPOINT=${ENDPOINT:-qwen-tiny}
KUBEAI_BASE_URL=${KUBEAI_BASE_URL:-http://127.0.0.1:8000/openai/v1}
echo "[kubeai-e2e] work dir: $WORK"

infer-stack config init --backend kubeai --yes >/dev/null
infer-stack config set kubeai_resource_profile cpu >/dev/null
infer-stack config set kubeai_base_url "$KUBEAI_BASE_URL" >/dev/null
cat > "$INFER_STACK_CONFIG_DIR/catalog.yaml" <<EOF
models:
  $ENDPOINT:
    source: hf://$MODEL
endpoints:
  $ENDPOINT:
    engine: vllm
    model: $ENDPOINT
    runtime: {max_model_len: 2048}
    reclaim: {policy: stop}
EOF

echo "[kubeai-e2e] doctor"
infer-stack doctor

echo "[kubeai-e2e] run: acquire, one real request, release"
start=$(date +%s)
# `run` exports the lease's env file; the request goes to whatever base URL
# and model name it advertises, exactly as a card would use them.
infer-stack run --endpoint "$ENDPOINT" --timeout 1500 -- bash -c '
    set -euo pipefail
    name_var="INFER_STACK_ENDPOINT_$(echo "'"$ENDPOINT"'" | tr "a-z.-" "A-Z__")"
    model=${!name_var}
    echo "[kubeai-e2e]   base=$OPENAI_BASE_URL model=$model"
    curl -sf "$OPENAI_BASE_URL/chat/completions" \
        -H "Authorization: Bearer ${OPENAI_API_KEY:-EMPTY}" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"max_tokens\": 8,
             \"messages\": [{\"role\": \"user\", \"content\": \"Say ok\"}]}" \
      | python3 -c "import json,sys; print(\"[kubeai-e2e]   reply:\", json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"])"
'
echo "[kubeai-e2e] run finished in $(( $(date +%s) - start ))s"

echo "[kubeai-e2e] the released (reclaim: stop) Model must be pruned"
left=$(kubectl -n "${KUBEAI_NAMESPACE:-kubeai}" get models.kubeai.org \
        -l infer-stack/managed=true -o name)
if [ -n "$left" ]; then
    echo "[kubeai-e2e] FAIL: still on the cluster: $left" >&2
    exit 1
fi
echo "[kubeai-e2e] PASS"
