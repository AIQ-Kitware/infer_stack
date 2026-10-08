#!/usr/bin/env bash
# Round 3: measure whether prefix caching makes full-context Qwen3.8-27B practical
# for an interactive coding-agent session on a Quadro RTX 8000 (Turing / sm75).
#
# Round 1 established that the W4A16 target with FP16/auto KV can expose the full
# 262144-token model context on this 48 GiB card. Round 2 established that TRITON_ATTN
# is the viable attention backend in the pinned image, 4096/8192 prefill chunks are
# effectively tied, and the prepared "-fast" target improves short decode modestly.
# See dev/qwen38_rtx8000_findings.md for the measurements and known dead ends.
#
# This round answers a different question: as a Pi-like conversation grows, can the
# server reuse the existing prompt prefix cheaply enough that long sessions remain
# interactive even though a cold long-context prefill is expensive?
#
# For each endpoint/context depth:
#   1. cold base prompt, max_tokens=1;
#   2. exact repeat, max_tokens=1 (pure prefix-cache reuse);
#   3. same prefix plus a small appended turn, max_tokens=1;
#   4. exact repeat of the appended turn, max_tokens=256 (decode at context depth).
#
# All benchmark traffic goes directly to vLLM inside its container. LiteLLM is not in
# the benchmark path because its 600 s request timeout/retry behavior invalidated the
# first near-full probe. Every model variant is an explicit infer-stack endpoint.
# Failures are recorded and the script continues; only this script's leases are released.
# Endpoints remain in the catalog for manual inspection.

set -uo pipefail

BASE_URL="${BASE_URL:-0.0.0.0:14042}"  # metadata / infer-stack sanity only
MODEL_NAME="qwen3.8-27b-dbirks-hyperqwen"
MODEL_SOURCE="hf://dbirks/Qwen3.8-27B-W4A16-AutoRound"
IMAGE="ghcr.io/syv-ai/hyperqwen:sha-684e927"
GPU_NAME="${GPU_NAME:-Quadro RTX 8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1200}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
MAX_MODEL_LEN="262144"
MAX_NUM_BATCHED_TOKENS="8192"
APPEND_MAX_TOKENS="${APPEND_MAX_TOKENS:-256}"
WARM_TIMEOUT="${WARM_TIMEOUT:-1800}"
STOP_AFTER_COLD_FAILURE="${STOP_AFTER_COLD_FAILURE:-1}"
RUN_DEEP="${RUN_DEEP:-0}"
CONTEXT_TARGETS="${CONTEXT_TARGETS:-8192 32000 65536 131072}"
if [[ "$RUN_DEEP" == 1 ]]; then
  CONTEXT_TARGETS="$CONTEXT_TARGETS 196608 240000"
fi
RESULT_DIR="${RESULT_DIR:-$PWD/dev/benchmark-results/qwen38-rtx8000-round3-$(date +%Y%m%dT%H%M%S)}"

mkdir -p "$RESULT_DIR"
SUMMARY_TSV="$RESULT_DIR/summary.tsv"
RUN_LOG="$RESULT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || { log "ERROR: missing command: $1"; exit 2; }; }
for cmd in infer-stack nvidia-smi docker curl jq awk sed grep timeout python3; do need_cmd "$cmd"; done

GPU_INDEX="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | awk -F',' -v want="$GPU_NAME" '
  index($2, want) { gsub(/[[:space:]]/, "", $1); print $1; exit }
')"
[[ -n "$GPU_INDEX" ]] || { log "ERROR: could not find GPU matching: $GPU_NAME"; exit 2; }

log "Using GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,power.limit --format=csv,noheader || true

{
  echo "date=$(date --iso-8601=seconds)"
  echo "base_url=$BASE_URL"
  echo "gpu_index=$GPU_INDEX"
  echo "gpu_name=$GPU_NAME"
  echo "gpu_util=$GPU_UTIL"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "context_targets=$CONTEXT_TARGETS"
  echo "run_deep=$RUN_DEEP"
  echo "infer_stack_git=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "host=$(hostname)"
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader || true
} > "$RESULT_DIR/metadata.txt"

log "Ensuring model entry exists"
infer-stack catalog model add "$MODEL_NAME" --source "$MODEL_SOURCE" --force

# Fields: endpoint|model_path|notes
VARIANTS=(
  "qwen38-rtx8000-r3-standard-262k|/app/models/Qwen3.8-27B-W4A16-AutoRound|standard prepared W4A16 target"
  "qwen38-rtx8000-r3-fast-262k|/app/models/Qwen3.8-27B-W4A16-AutoRound-fast|HyperQwen fast target checkpoint"
)

cleanup_endpoint_leases() {
  local endpoint="$1" ids id
  ids="$(infer-stack leases --json 2>/dev/null | jq -r --arg ep "$endpoint" '
    .leases[]? | select(.state == "active") | select((.endpoints // []) | index($ep)) | .id
  ' 2>/dev/null || true)"
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    infer-stack release "$id" --evict --yes >/dev/null 2>&1 || true
  done <<< "$ids"
}

CURRENT_ENDPOINT=""
CURRENT_ENV_FILE=""
cleanup_current() {
  if [[ -n "$CURRENT_ENV_FILE" && -f "$CURRENT_ENV_FILE" ]]; then
    infer-stack release --env-file "$CURRENT_ENV_FILE" --evict --yes >/dev/null 2>&1 || true
  fi
  [[ -n "$CURRENT_ENDPOINT" ]] && cleanup_endpoint_leases "$CURRENT_ENDPOINT"
  CURRENT_ENDPOINT=""
  CURRENT_ENV_FILE=""
}
trap cleanup_current EXIT INT TERM

add_endpoint() {
  local endpoint="$1" model_path="$2"
  # batch/start_qwen.sh does not have a native FP16/auto-KV profile: its default
  # KV=fp8 is an Ampere+/FlashInfer-oriented profile. Keep the launcher's preparation,
  # tool parser and prefix-cache integration, but override the KV dtype/backend in the
  # trailing EXTRA_ARGS, which is deliberately expanded last by HyperQwen.
  local extra env_yaml
  extra="--served-model-name={served_model_name} --dtype=half --kv-cache-dtype=auto --attention-backend=TRITON_ATTN --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}"
  env_yaml="{PORT: \"{port}\", VERIFY: 0, MODEL: \"$model_path\", KV: fp8, PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"\", INT8_LAYERS: \"\", EXTRA_ARGS: \"$extra\"}"

  infer-stack catalog endpoint add "$endpoint" \
    --engine vllm \
    --model "$MODEL_NAME" \
    --min-vram-gib 40 \
    --gpu "$GPU_INDEX" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-mem "$GPU_UTIL" \
    --reclaim stop \
    --force \
    --runtime \
      "image=$IMAGE" \
      'enable_prefix_caching=true' \
      'max_num_seqs=1' \
      'command=[batch]' \
      "env=$env_yaml" \
      'mounts={"/app/models": "hyperqwen/qwen3.8-27b/models", "/cache": "hyperqwen/qwen3.8-27b/cache"}'
}

for row in "${VARIANTS[@]}"; do
  IFS='|' read -r endpoint model_path notes <<< "$row"
  log "Catalog: $endpoint -- $notes"
  add_endpoint "$endpoint" "$model_path"
done
infer-stack catalog validate

printf 'endpoint\tmodel_path\ttarget_tokens\tstatus\tboot_seconds\tkv_cache_tokens\tmax_concurrency\tcold_prompt_tokens\tcold_cached_tokens\tcold_wall_s\trepeat_cached_tokens\trepeat_wall_s\tappend_prompt_tokens\tappend_cached_tokens\tappend_wall_s\tdecode_prompt_tokens\tdecode_cached_tokens\tdecode_completion_tokens\tdecode_wall_s\tdecode_tok_s\tnotes\terror\n' > "$SUMMARY_TSV"

sanitize() { printf '%s' "$1" | tr '\t\r\n' '   '; }
append_result() {
  local endpoint="$1" model_path="$2" target="$3" status="$4" boot="$5" kv="$6" conc="$7"
  local cold_pt="$8" cold_ct="$9" cold_w="${10}" repeat_ct="${11}" repeat_w="${12}"
  local append_pt="${13}" append_ct="${14}" append_w="${15}" decode_pt="${16}" decode_ct="${17}"
  local decode_comp="${18}" decode_w="${19}" decode_tps="${20}" notes="${21}" error="${22}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize "$endpoint")" "$(sanitize "$model_path")" "$(sanitize "$target")" "$(sanitize "$status")" \
    "$(sanitize "$boot")" "$(sanitize "$kv")" "$(sanitize "$conc")" "$(sanitize "$cold_pt")" \
    "$(sanitize "$cold_ct")" "$(sanitize "$cold_w")" "$(sanitize "$repeat_ct")" "$(sanitize "$repeat_w")" \
    "$(sanitize "$append_pt")" "$(sanitize "$append_ct")" "$(sanitize "$append_w")" "$(sanitize "$decode_pt")" \
    "$(sanitize "$decode_ct")" "$(sanitize "$decode_comp")" "$(sanitize "$decode_w")" "$(sanitize "$decode_tps")" \
    "$(sanitize "$notes")" "$(sanitize "$error")" >> "$SUMMARY_TSV"
}

find_container() {
  docker ps -a --filter "name=infer-stack-vllm-$1" --format '{{.Names}}' | head -n 1
}

capture_logs() {
  local container="$1" outfile="$2"
  [[ -n "$container" ]] && docker logs "$container" > "$outfile" 2>&1 || true
}

parse_boot_facts() {
  local logf="$1"
  KV_CACHE_TOKENS="$(grep -Eo 'GPU KV cache size: [0-9,]+ tokens' "$logf" | tail -1 | sed -E 's/.*size: ([0-9,]+) tokens/\1/' | tr -d ',' || true)"
  MAX_CONCURRENCY="$(grep -Eo 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$logf" | tail -1 | sed -E 's/.*request: ([0-9.]+)x/\1/' || true)"
}

wait_with_crash_detection() {
  local endpoint="$1" wait_log="$2" deadline=$(( $(date +%s) + BOOT_TIMEOUT )) wait_pid container status restarts
  infer-stack wait "$endpoint" --timeout "$BOOT_TIMEOUT" --interval "$WAIT_INTERVAL" > "$wait_log" 2>&1 &
  wait_pid=$!
  while kill -0 "$wait_pid" >/dev/null 2>&1; do
    if (( $(date +%s) >= deadline )); then
      kill "$wait_pid" >/dev/null 2>&1 || true
      wait "$wait_pid" >/dev/null 2>&1 || true
      echo 'timeout waiting for readiness' >> "$wait_log"
      return 1
    fi
    container="$(find_container "$endpoint")"
    if [[ -n "$container" ]]; then
      status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
      restarts="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
      if [[ "$status" == restarting || "$status" == dead || "$status" == exited ]]; then
        sleep 8
        status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
        restarts="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
        if [[ "$status" == restarting || "$status" == dead || "$status" == exited ]]; then
          echo "container failure: status=$status restarts=$restarts" >> "$wait_log"
          kill "$wait_pid" >/dev/null 2>&1 || true
          wait "$wait_pid" >/dev/null 2>&1 || true
          return 1
        fi
      fi
    fi
    sleep 10
  done
  wait "$wait_pid"
}

# Direct request to vLLM. This intentionally bypasses LiteLLM, whose 600-second
# request timeout/retry policy made the round-1 near-full measurement unusable.
direct_request_file() {
  local container="$1" request_file="$2" response_file="$3" timeout_s="$4"
  local meta="$response_file.curl.txt"
  if ! timeout "$((timeout_s + 30))" bash -c \
    'docker exec -i "$1" curl -sS --max-time "$2" -o /tmp/infer-stack-r3-response.json -w "%{http_code}\t%{time_total}\n" -H "Content-Type: application/json" --data-binary @- http://127.0.0.1:8000/v1/chat/completions < "$3" > "$4" && docker exec "$1" cat /tmp/infer-stack-r3-response.json > "$5"' \
    _ "$container" "$timeout_s" "$request_file" "$meta" "$response_file"; then
    return 1
  fi
  [[ "$(cut -f1 "$meta" 2>/dev/null || true)" == 200 ]] || return 1
  jq -e '.choices | length > 0' "$response_file" >/dev/null 2>&1
}

make_request() {
  local endpoint="$1" prompt_file="$2" max_tokens="$3" request_file="$4"
  jq -n --arg model "$endpoint" --rawfile prompt "$prompt_file" --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:$max_tokens}' > "$request_file"
}

make_prompt() {
  local container="$1" target="$2" tag="$3" outfile="$4"
  log "Generating tokenizer-measured ~${target}-token prompt: $tag"
  docker exec -i "$container" /app/venv/bin/python - "$target" "$tag" <<'PY' > "$outfile"
import sys
from transformers import AutoTokenizer

target = int(sys.argv[1])
tag = sys.argv[2]
tok = AutoTokenizer.from_pretrained('/app/models/Qwen3.8-27B-W4A16-AutoRound')
tok.model_max_length = 10**12
unit = (
    f'[{tag}] A software repository contains source files, tests, API contracts, '
    'design notes, invariants, dependency metadata, and repeated deterministic context '
    'for a prefix-cache benchmark. Preserve exact earlier context across turns.\n'
)

def count(n):
    return len(tok.encode(unit * n, add_special_tokens=False))

lo, hi = 0, max(2, target // max(1, count(1)) * 2 + 16)
while count(hi) < target:
    hi *= 2
while lo + 1 < hi:
    mid = (lo + hi) // 2
    if count(mid) <= target:
        lo = mid
    else:
        hi = mid
text = unit * lo
body = len(tok.encode(text, add_special_tokens=False))
text += f'\n[{tag} target={target} generated_body_tokens={body}]\n'
sys.stdout.write(text)
PY
}

make_appended_prompt() {
  local base="$1" target="$2" outfile="$3"
  cat "$base" > "$outfile"
  cat >> "$outfile" <<EOF_APPEND

[round3 appended turn after approximately ${target} tokens]
The repository has changed since the previous turn. Summarize the important architectural invariants that should remain stable, explain one plausible regression risk, and reason carefully about the next change. Continue until the requested token limit.
EOF_APPEND
}

usage_field() {
  local response="$1" jq_expr="$2"
  jq -r "$jq_expr // 0" "$response" 2>/dev/null || printf '0'
}

cold_timeout_for_target() {
  local target="$1"
  if (( target <= 8192 )); then echo "${COLD_TIMEOUT_8K:-900}"
  elif (( target <= 32000 )); then echo "${COLD_TIMEOUT_32K:-1800}"
  elif (( target <= 65536 )); then echo "${COLD_TIMEOUT_64K:-3600}"
  elif (( target <= 131072 )); then echo "${COLD_TIMEOUT_128K:-10800}"
  elif (( target <= 196608 )); then echo "${COLD_TIMEOUT_192K:-18000}"
  else echo "${COLD_TIMEOUT_240K:-21600}"
  fi
}

run_request() {
  local container="$1" endpoint="$2" prompt_file="$3" max_tokens="$4" response="$5" timeout_s="$6"
  local request="$response.request.json"
  make_request "$endpoint" "$prompt_file" "$max_tokens" "$request"
  direct_request_file "$container" "$request" "$response" "$timeout_s"
}

run_context_probe() {
  local container="$1" endpoint="$2" model_path="$3" target="$4" outdir="$5" boot="$6" notes="$7"
  local prompt="$RESULT_DIR/prompts/${target}.txt"
  local appended="$RESULT_DIR/prompts/${target}-appended.txt"
  local timeout_s
  timeout_s="$(cold_timeout_for_target "$target")"
  mkdir -p "$RESULT_DIR/prompts" "$outdir"
  [[ -s "$prompt" ]] || make_prompt "$container" "$target" "round3-${target}" "$prompt" || return 1
  [[ -s "$appended" ]] || make_appended_prompt "$prompt" "$target" "$appended"

  local cold="$outdir/cold-${target}.json"
  local repeat="$outdir/repeat-${target}.json"
  local append="$outdir/append-${target}.json"
  local decode="$outdir/decode-${target}.json"
  local error="" status="ok"

  log "  context=$target: cold base request (timeout=${timeout_s}s)"
  if ! run_request "$container" "$endpoint" "$prompt" 1 "$cold" "$timeout_s"; then
    error="cold request failed or timed out"
    status="cold_failed"
    append_result "$endpoint" "$model_path" "$target" "$status" "$boot" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" \
      "" "" "" "" "" "" "" "" "" "" "" "" "" "$notes" "$error"
    return 1
  fi

  local cold_pt cold_ct cold_w repeat_ct repeat_w append_pt append_ct append_w decode_pt decode_ct decode_comp decode_w decode_tps
  cold_pt="$(usage_field "$cold" '.usage.prompt_tokens')"
  cold_ct="$(usage_field "$cold" '.usage.prompt_tokens_details.cached_tokens')"
  cold_w="$(cut -f2 "$cold.curl.txt" 2>/dev/null || true)"

  log "  context=$target: exact cached repeat"
  if run_request "$container" "$endpoint" "$prompt" 1 "$repeat" "$WARM_TIMEOUT"; then
    repeat_ct="$(usage_field "$repeat" '.usage.prompt_tokens_details.cached_tokens')"
    repeat_w="$(cut -f2 "$repeat.curl.txt" 2>/dev/null || true)"
  else
    repeat_ct=""; repeat_w=""; status="partial"; error="exact repeat failed"
  fi

  log "  context=$target: append a small new turn over the cached prefix"
  if run_request "$container" "$endpoint" "$appended" 1 "$append" "$WARM_TIMEOUT"; then
    append_pt="$(usage_field "$append" '.usage.prompt_tokens')"
    append_ct="$(usage_field "$append" '.usage.prompt_tokens_details.cached_tokens')"
    append_w="$(cut -f2 "$append.curl.txt" 2>/dev/null || true)"
  else
    append_pt=""; append_ct=""; append_w=""; status="partial"; error="${error:+$error; }append turn failed"
  fi

  log "  context=$target: 256-token decode from the fully cached appended prompt"
  if run_request "$container" "$endpoint" "$appended" "$APPEND_MAX_TOKENS" "$decode" "$WARM_TIMEOUT"; then
    decode_pt="$(usage_field "$decode" '.usage.prompt_tokens')"
    decode_ct="$(usage_field "$decode" '.usage.prompt_tokens_details.cached_tokens')"
    decode_comp="$(usage_field "$decode" '.usage.completion_tokens')"
    decode_w="$(cut -f2 "$decode.curl.txt" 2>/dev/null || true)"
    decode_tps="$(awk -v t="$decode_comp" -v w="$decode_w" 'BEGIN{if(w>0)printf "%.3f",t/w}')"
  else
    decode_pt=""; decode_ct=""; decode_comp=""; decode_w=""; decode_tps=""
    status="partial"; error="${error:+$error; }cached decode failed"
  fi

  append_result "$endpoint" "$model_path" "$target" "$status" "$boot" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" \
    "$cold_pt" "$cold_ct" "$cold_w" "$repeat_ct" "$repeat_w" "$append_pt" "$append_ct" "$append_w" \
    "$decode_pt" "$decode_ct" "$decode_comp" "$decode_w" "$decode_tps" "$notes" "$error"

  log "  context=$target: cold=${cold_w:-NA}s repeat=${repeat_w:-NA}s repeat_cached=${repeat_ct:-NA}/${cold_pt:-NA} append=${append_w:-NA}s append_cached=${append_ct:-NA}/${append_pt:-NA} decode=${decode_tps:-NA} tok/s"
  return 0
}

run_variant() {
  local row="$1"
  local endpoint model_path notes
  IFS='|' read -r endpoint model_path notes <<< "$row"
  local outdir="$RESULT_DIR/$endpoint" env_file="$RESULT_DIR/$endpoint/lease.env"
  mkdir -p "$outdir"
  cleanup_current
  cleanup_endpoint_leases "$endpoint"
  CURRENT_ENDPOINT="$endpoint"
  CURRENT_ENV_FILE="$env_file"
  rm -f "$env_file"

  log "======================================================================"
  log "$endpoint -- $notes"
  infer-stack catalog endpoint show "$endpoint" > "$outdir/catalog.yaml" 2>&1 || true
  local start boot container err
  start="$(date +%s)"
  if ! timeout "$BOOT_TIMEOUT" infer-stack acquire "$endpoint" --allowed_gpus "$GPU_INDEX" --env-file "$env_file" --no-wait --yes > "$outdir/acquire.log" 2>&1; then
    boot=$(( $(date +%s) - start ))
    append_result "$endpoint" "$model_path" "boot" acquire_failed "$boot" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "$notes" "acquire failed"
    cleanup_current
    return 1
  fi
  if ! wait_with_crash_detection "$endpoint" "$outdir/wait.log"; then
    boot=$(( $(date +%s) - start ))
    container="$(find_container "$endpoint")"
    capture_logs "$container" "$outdir/container.log"
    parse_boot_facts "$outdir/container.log"
    err="$(grep -E 'ERROR|ValueError|RuntimeError|OutOfMemory|not supported|requires' "$outdir/container.log" | tail -n 5 | tr '\n' ' ')"
    append_result "$endpoint" "$model_path" "boot" readiness_failed "$boot" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" "" "" "" "" "" "" "" "" "" "" "" "" "" "$notes" "$err"
    cleanup_current
    return 1
  fi

  boot=$(( $(date +%s) - start ))
  container="$(find_container "$endpoint")"
  capture_logs "$container" "$outdir/container.log"
  parse_boot_facts "$outdir/container.log"
  infer-stack test "$endpoint" --max-tokens 32 --timeout 300 > "$outdir/infer-stack-test.log" 2>&1 || true

  local target failed=0
  for target in $CONTEXT_TARGETS; do
    if ! run_context_probe "$container" "$endpoint" "$model_path" "$target" "$outdir" "$boot" "$notes"; then
      failed=1
      if [[ "$STOP_AFTER_COLD_FAILURE" == 1 ]]; then
        log "  stopping deeper contexts for $endpoint after cold failure at $target"
        break
      fi
    fi
  done

  capture_logs "$container" "$outdir/container.log"
  cleanup_current
  return "$failed"
}

log "Removing active leases for round-3 endpoints"
for row in "${VARIANTS[@]}"; do
  IFS='|' read -r endpoint _ <<< "$row"
  cleanup_endpoint_leases "$endpoint"
done

for row in "${VARIANTS[@]}"; do
  run_variant "$row" || true
done

python3 - "$SUMMARY_TSV" "$RESULT_DIR/summary.csv" <<'PY'
import csv
import sys
src, dst = sys.argv[1:]
with open(src, newline='') as f, open(dst, 'w', newline='') as g:
    csv.writer(g).writerows(csv.reader(f, delimiter='\t'))
PY

# A compact view sorted by context then cached-decode throughput descending.
{
  head -n1 "$SUMMARY_TSV"
  tail -n +2 "$SUMMARY_TSV" | sort -t $'\t' -k3,3n -k20,20nr
} > "$RESULT_DIR/ranking.tsv"

log "======================================================================"
log "Round 3 complete"
log "Summary: $SUMMARY_TSV"
log "Ranking: $RESULT_DIR/ranking.tsv"
log "Logs/responses: $RESULT_DIR"
log "All round-3 experiment leases have been released; endpoints remain in the catalog."
column -t -s $'\t' "$RESULT_DIR/ranking.tsv" 2>/dev/null || cat "$RESULT_DIR/ranking.tsv"
