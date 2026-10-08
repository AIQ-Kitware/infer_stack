#!/usr/bin/env bash
# P4 on real GPUs: what the CPU-only development cluster cannot show.
#
# Run on a GPU machine that is a k3s node with the NVIDIA device plugin and
# GPU Feature Discovery (README: "NVIDIA GPU support"), with the KubeAI chart
# installed and `kubectl -n kubeai port-forward svc/kubeai 8000:80` running.
# It serves one small model on one GPU for a few minutes: run it when that
# GPU is free.
#
# It checks, in order:
#   1. GPU Feature Discovery labels each GPU node (product, memory) and the
#      device plugin advertises nvidia.com/gpu: the facts sizing reads.
#   2. `catalog suggest --backend kubeai` proposes a resource profile for
#      each GPU product, from those labels.
#   3. With the proposed profiles installed, an endpoint that declares only
#      `min_vram_gib` gets the right profile and answers a request on a GPU.
#   4. `infer-stack measure <endpoint> --record` finds vLLM's memory-profiling
#      lines in the pod's log and records a min_vram_gib.
#
# It uses its own config and data directories (never this host's infer-stack
# state). It leaves the proposed profiles in the chart's values: they
# describe this cluster's GPUs and are what min_vram_gib needs.
#
#   dev/handover/p4_gpu_labels.sh 2>&1 | tee p4-gpu.log
#
# Knobs: E2E_MODEL (default Qwen/Qwen2.5-0.5B-Instruct), NAMESPACE (kubeai),
# E2E_TIMEOUT (900 s).
set -euo pipefail

MODEL="${E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NAMESPACE="${NAMESPACE:-kubeai}"
TIMEOUT="${E2E_TIMEOUT:-900}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/infer-stack-p4.XXXXXX")"
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

echo '== 1. the GPU facts sizing reads'
kubectl get nodes -o json | python3 -c '
import json, sys
found = 0
for n in json.load(sys.stdin)["items"]:
    labels = n["metadata"].get("labels", {})
    gpus = n["status"].get("allocatable", {}).get("nvidia.com/gpu")
    product, memory = labels.get("nvidia.com/gpu.product"), labels.get("nvidia.com/gpu.memory")
    print(f"  {n[\"metadata\"][\"name\"]}: gpus={gpus} product={product} memory_mib={memory}")
    found += bool(gpus and product and memory)
sys.exit(0 if found else 1)' || fail 'no node has nvidia.com/gpu and the GFD product/memory labels'
pass 'GPU nodes are labelled'

echo '== 2. catalog suggest proposes profiles from the labels'
run_is config set backend kubeai >/dev/null
run_is config set kubeai_namespace "$NAMESPACE" >/dev/null
run_is catalog suggest --backend kubeai > "$WORK/suggest.yaml" 2> "$WORK/suggest.err" \
  || { cat "$WORK/suggest.err" >&2; fail 'catalog suggest failed'; }
python3 - "$WORK/suggest.err" "$WORK/profiles.yaml" <<'PY' || fail 'suggest printed no resourceProfiles'
import sys, yaml
text = open(sys.argv[1]).read()
block = yaml.safe_load(text[text.index('resourceProfiles:'):])
open(sys.argv[2], 'w').write(yaml.safe_dump(block, sort_keys=False))
print(yaml.safe_dump(block, sort_keys=False))
PY
pass 'suggest proposed a profile per GPU product'

echo '== 3. min_vram_gib picks the proposed profile and serves on a GPU'
version=$(helm list -n "$NAMESPACE" -o json | python3 -c \
  'import json,sys; print(json.load(sys.stdin)[0]["chart"].rsplit("-", 1)[1])')
helm upgrade kubeai kubeai/kubeai -n "$NAMESPACE" --version "$version" \
  --reuse-values -f "$WORK/profiles.yaml" --wait >/dev/null
cat > "$WORK/config/catalog.yaml" <<EOF
models:
  small: {source: hf://$MODEL}
endpoints:
  p4-small:
    engine: vllm
    model: small
    reclaim: {policy: stop}
    placement: {min_vram_gib: 8}
    runtime: {max_model_len: 4096}
EOF
# Into a file first: `grep -q` quits at its match, and under pipefail the
# writer's SIGPIPE would fail the check.
run_is acquire p4-small --yes --timeout "$TIMEOUT" --env-file "$WORK/lease.env" \
  > "$WORK/acquire.log" 2>&1 || true
grep -q 'ready: True' "$WORK/acquire.log" || fail 'p4-small never became ready'
got=$(kubectl -n "$NAMESPACE" get models.kubeai.org -l infer-stack/managed=true \
  -o jsonpath='{.items[0].spec.resourceProfile}')
echo "  Model resourceProfile: $got (profiles proposed: $(tr '\n' ' ' < "$WORK/profiles.yaml" | head -c 200))"
case "$got" in nvidia-*:1) pass "min_vram_gib 8 -> $got, and it answers" ;;
                *) fail "unexpected resource profile $got" ;; esac

echo '== 4. measure reads the pod log on a GPU'
run_is measure p4-small --record | tee "$WORK/measure.log"
grep -q 'min_vram_gib' "$WORK/measure.log" || fail 'measure printed no min_vram_gib'
test -s "$WORK/data/leasing/kubeai/measurements.json" || fail 'measure --record wrote nothing'
pass 'measure recorded a min_vram_gib on kubeai'

echo 'ALL PASS'
