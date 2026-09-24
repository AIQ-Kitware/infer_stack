#!/usr/bin/env bash
# Round 2: optimize the known-good full-context Qwen3.8-27B path on Quadro RTX 8000.
#
# Goals:
#   1. Keep the proven W4A16 + FP16/auto KV + 262144 context configuration.
#   2. Benchmark the two sm75-compatible attention backends vLLM reported:
#        TRITON_ATTN and FLEX_ATTENTION.
#   3. Sweep chunked-prefill size, because long-prompt prefill is the bottleneck.
#   4. After finding the fastest base arm, probe the fast checkpoint and speculative
#      decoding on that backend/chunk size without low-bit KV.
#   5. Bypass LiteLLM for benchmark traffic so its 600 s timeout/retries cannot turn
#      a slow long prefill into a false model failure.
#
# Every arm is a separate infer-stack endpoint. Arms run one at a time. Failures are
# recorded and the run continues. Only leases in this script's endpoint namespace are
# released. Endpoints remain in the catalog for manual follow-up.

set -uo pipefail

BASE_URL="${BASE_URL:-0.0.0.0:14042}"   # retained for metadata / infer-stack sanity tests
MODEL_NAME="qwen3.8-27b-dbirks-hyperqwen"
MODEL_SOURCE="hf://dbirks/Qwen3.8-27B-W4A16-AutoRound"
IMAGE="ghcr.io/syv-ai/hyperqwen:sha-684e927"
GPU_NAME="Quadro RTX 8000"
GPU_UTIL="${GPU_UTIL:-0.90}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1200}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
SHORT_RUNS="${SHORT_RUNS:-3}"
SMALL_PROMPT_TOKENS="${SMALL_PROMPT_TOKENS:-8192}"
MEDIUM_PROMPT_TOKENS="${MEDIUM_PROMPT_TOKENS:-32000}"
RUN_SPECULATION="${RUN_SPECULATION:-1}"
RUN_NEARFULL="${RUN_NEARFULL:-0}"
NEARFULL_PROMPT_TOKENS="${NEARFULL_PROMPT_TOKENS:-258000}"
NEARFULL_TIMEOUT="${NEARFULL_TIMEOUT:-21600}"
RESULT_DIR="${RESULT_DIR:-$PWD/dev/benchmark-results/qwen38-rtx8000-round2-$(date +%Y%m%dT%H%M%S)}"

mkdir -p "$RESULT_DIR"
SUMMARY_TSV="$RESULT_DIR/summary.tsv"
RUN_LOG="$RESULT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || { log "ERROR: missing command: $1"; exit 2; }; }
for cmd in infer-stack nvidia-smi docker curl jq awk sed grep timeout python3 sort; do need_cmd "$cmd"; done

GPU_INDEX="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | awk -F',' -v want="$GPU_NAME" '
  index($2, want) { gsub(/[[:space:]]/, "", $1); print $1; exit }
')"
[[ -n "$GPU_INDEX" ]] || { log "ERROR: could not find $GPU_NAME"; exit 2; }

log "Using GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,power.limit --format=csv,noheader || true

{
  echo "date=$(date --iso-8601=seconds)"
  echo "base_url=$BASE_URL"
  echo "gpu_index=$GPU_INDEX"
  echo "gpu_name=$GPU_NAME"
  echo "gpu_util=$GPU_UTIL"
  echo "infer_stack_git=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "host=$(hostname)"
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader || true
} > "$RESULT_DIR/metadata.txt"

log "Ensuring model entry exists"
infer-stack catalog model add "$MODEL_NAME" --source "$MODEL_SOURCE" --force

# Fields: endpoint|launcher|backend|mbt|model_path|spec|draft_tokens|notes
BASE_VARIANTS=()
for backend in TRITON_ATTN FLEX_ATTENTION; do
  backend_slug="triton"; [[ "$backend" == FLEX_ATTENTION ]] && backend_slug="flex"
  for mbt in 2048 4096 8192 16384; do
    BASE_VARIANTS+=("qwen38-rtx8000-${backend_slug}-bt${mbt}-262k|batch|${backend}|${mbt}|/app/models/Qwen3.8-27B-W4A16-AutoRound|off|0|W4A16 FP16/auto KV, ${backend}, max-num-batched-tokens=${mbt}")
  done
done

ALL_ENDPOINTS=()
for row in "${BASE_VARIANTS[@]}"; do IFS='|' read -r ep _ <<< "$row"; ALL_ENDPOINTS+=("$ep"); done

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
  CURRENT_ENDPOINT=""; CURRENT_ENV_FILE=""
}
trap cleanup_current EXIT INT TERM

add_endpoint() {
  local endpoint="$1" launcher="$2" backend="$3" mbt="$4" model_path="$5" spec="$6" draft_tokens="$7"
  local extra env_yaml
  extra="--served-model-name={served_model_name} --dtype=half --kv-cache-dtype=auto --attention-backend=${backend} --max-num-batched-tokens=${mbt}"

  if [[ "$launcher" == batch ]]; then
    env_yaml="{PORT: \"{port}\", VERIFY: 0, MODEL: \"$model_path\", KV: fp8, PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"\", INT8_LAYERS: \"\", EXTRA_ARGS: \"$extra\"}"
  elif [[ "$spec" == mtp ]]; then
    env_yaml="{PORT: \"{port}\", VERIFY: 0, MODEL: \"$model_path\", CTX: fast, SPEC: mtp, SPEC_ATTN: 0, DRAFT_TOKENS: $draft_tokens, PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"\", INT8_LAYERS: \"\", PREFILL_ATTN: \"\", EXTRA_ARGS: \"$extra\"}"
  else
    # DFlash2's 24 GiB profiles pin a ~5.2 GiB KV pool. KV_MEM="" deliberately
    # disables that pin so the 48 GiB card sizes its FP16/auto pool from GPU_UTIL.
    env_yaml="{PORT: \"{port}\", VERIFY: 0, MODEL: \"$model_path\", CTX: fast, SPEC: dflash2, SPEC_ATTN: 0, DFLASH_TOKENS: $draft_tokens, KV_MEM: \"\", PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"\", INT8_LAYERS: \"\", PREFILL_ATTN: \"\", EXTRA_ARGS: \"$extra\"}"
  fi

  infer-stack catalog endpoint add "$endpoint" \
    --engine vllm \
    --model "$MODEL_NAME" \
    --min-vram-gib 40 \
    --gpu "$GPU_INDEX" \
    --max-model-len 262144 \
    --gpu-mem "$GPU_UTIL" \
    --reclaim stop \
    --force \
    --runtime \
      "image=$IMAGE" \
      'enable_prefix_caching=true' \
      'max_num_seqs=1' \
      "command=[$launcher]" \
      "env=$env_yaml" \
      'mounts={"/app/models": "hyperqwen/qwen3.8-27b/models", "/cache": "hyperqwen/qwen3.8-27b/cache"}'
}

for row in "${BASE_VARIANTS[@]}"; do
  IFS='|' read -r endpoint launcher backend mbt model_path spec draft_tokens notes <<< "$row"
  log "Catalog: $endpoint -- $notes"
  add_endpoint "$endpoint" "$launcher" "$backend" "$mbt" "$model_path" "$spec" "$draft_tokens"
done
infer-stack catalog validate

printf 'endpoint\tlauncher\tbackend\tmbt\tmodel_path\tspec\tdraft_tokens\tstatus\tboot_seconds\tkv_cache_tokens\tmax_concurrency\tshort_tok_s\tshort_wall_s\tsmall_prompt_tokens\tsmall_wall_s\tmedium_prompt_tokens\tmedium_wall_s\tnotes\terror\n' > "$SUMMARY_TSV"

sanitize() { printf '%s' "$1" | tr '\t\r\n' '   '; }
append_result() {
  local endpoint="$1" launcher="$2" backend="$3" mbt="$4" model_path="$5" spec="$6" draft_tokens="$7" status="$8" boot="$9"
  local kv="${10}" conc="${11}" short_tps="${12}" short_wall="${13}" small_tok="${14}" small_wall="${15}" med_tok="${16}" med_wall="${17}" notes="${18}" error="${19}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize "$endpoint")" "$(sanitize "$launcher")" "$(sanitize "$backend")" "$(sanitize "$mbt")" "$(sanitize "$model_path")" "$(sanitize "$spec")" "$(sanitize "$draft_tokens")" \
    "$(sanitize "$status")" "$(sanitize "$boot")" "$(sanitize "$kv")" "$(sanitize "$conc")" "$(sanitize "$short_tps")" "$(sanitize "$short_wall")" \
    "$(sanitize "$small_tok")" "$(sanitize "$small_wall")" "$(sanitize "$med_tok")" "$(sanitize "$med_wall")" "$(sanitize "$notes")" "$(sanitize "$error")" >> "$SUMMARY_TSV"
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
      kill "$wait_pid" >/dev/null 2>&1 || true; wait "$wait_pid" >/dev/null 2>&1 || true
      echo 'timeout waiting for readiness' >> "$wait_log"; return 1
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
          kill "$wait_pid" >/dev/null 2>&1 || true; wait "$wait_pid" >/dev/null 2>&1 || true
          return 1
        fi
      fi
    fi
    sleep 10
  done
  wait "$wait_pid"
}

# Long benchmark requests go straight to the vLLM process inside its container. This
# deliberately bypasses LiteLLM's 600 s request timeout and any gateway retry policy.
direct_request_file() {
  local container="$1" request_file="$2" response_file="$3" timeout_s="$4"
  local meta="$response_file.curl.txt"
  if ! timeout "$((timeout_s + 30))" bash -c \
    'docker exec -i "$1" curl -sS --max-time "$2" -o /tmp/infer-stack-bench-response.json -w "%{http_code}\t%{time_total}\n" -H "Content-Type: application/json" --data-binary @- http://127.0.0.1:8000/v1/chat/completions < "$3" > "$4" && docker exec "$1" cat /tmp/infer-stack-bench-response.json > "$5"' \
    _ "$container" "$timeout_s" "$request_file" "$meta" "$response_file"; then
    return 1
  fi
  [[ "$(cut -f1 "$meta" 2>/dev/null || true)" == 200 ]] || return 1
  jq -e '.choices | length > 0' "$response_file" >/dev/null 2>&1
}

direct_request_prompt() {
  local container="$1" endpoint="$2" prompt_file="$3" max_tokens="$4" response_file="$5" timeout_s="$6"
  local req="$response_file.request.json"
  jq -n --arg model "$endpoint" --rawfile prompt "$prompt_file" --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:$max_tokens}' > "$req"
  direct_request_file "$container" "$req" "$response_file" "$timeout_s"
}

direct_request_text() {
  local container="$1" endpoint="$2" prompt="$3" max_tokens="$4" response_file="$5" timeout_s="$6"
  local req="$response_file.request.json"
  jq -n --arg model "$endpoint" --arg prompt "$prompt" --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:$max_tokens}' > "$req"
  direct_request_file "$container" "$req" "$response_file" "$timeout_s"
}

make_prompt() {
  local container="$1" target="$2" tag="$3" outfile="$4"
  log "Generating tokenizer-measured ~${target}-token prompt: $tag"
  docker exec -i "$container" /app/venv/bin/python - "$target" "$tag" <<'PY' > "$outfile"
import sys
from transformers import AutoTokenizer

target = int(sys.argv[1]); tag = sys.argv[2]
tok = AutoTokenizer.from_pretrained('/app/models/Qwen3.8-27B-W4A16-AutoRound')
# Suppress the tokenizer's model_max_length warning while searching; the final prompt
# target is independently bounded below the served max context.
tok.model_max_length = 10**12
unit = f'[{tag}] A database page stores ordered keys, child pointers, transaction metadata, checksums, recovery information, and repeated context for a deterministic prefill benchmark.\n'

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

SMALL_PROMPT="$RESULT_DIR/prompt-${SMALL_PROMPT_TOKENS}.txt"
MEDIUM_PROMPT="$RESULT_DIR/prompt-${MEDIUM_PROMPT_TOKENS}.txt"
NEARFULL_PROMPT="$RESULT_DIR/prompt-${NEARFULL_PROMPT_TOKENS}.txt"
PROMPTS_READY=0
ensure_prompts() {
  local container="$1"
  [[ -s "$SMALL_PROMPT" ]] || make_prompt "$container" "$SMALL_PROMPT_TOKENS" small "$SMALL_PROMPT" || return 1
  [[ -s "$MEDIUM_PROMPT" ]] || make_prompt "$container" "$MEDIUM_PROMPT_TOKENS" medium "$MEDIUM_PROMPT" || return 1
  if [[ "$RUN_NEARFULL" == 1 ]]; then
    [[ -s "$NEARFULL_PROMPT" ]] || make_prompt "$container" "$NEARFULL_PROMPT_TOKENS" nearfull "$NEARFULL_PROMPT" || return 1
  fi
  PROMPTS_READY=1
}

benchmark_short() {
  local container="$1" endpoint="$2" outdir="$3"
  local prompt='Explain virtual memory, page tables, TLBs, page faults, and memory-mapped files in technical detail. Continue until the token limit.'
  local i resp wall toks
  local tps=() walls=()
  direct_request_text "$container" "$endpoint" "$prompt" 64 "$outdir/short-warmup.json" 300 || true
  for ((i=1; i<=SHORT_RUNS; i++)); do
    resp="$outdir/short-$i.json"
    if direct_request_text "$container" "$endpoint" "$prompt" 512 "$resp" 900; then
      wall="$(cut -f2 "$resp.curl.txt")"; toks="$(jq -r '.usage.completion_tokens // 0' "$resp")"
      rate="$(awk -v t="$toks" -v w="$wall" 'BEGIN{if(w>0)printf "%.3f",t/w}')"
      tps+=("$rate")
      walls+=("$wall")
    fi
  done
  if ((${#tps[@]})); then
    SHORT_TPS="$(printf '%s\n' "${tps[@]}" | awk '{s+=$1;n++}END{printf "%.3f",s/n}')"
    SHORT_WALL="$(printf '%s\n' "${walls[@]}" | awk '{s+=$1;n++}END{printf "%.3f",s/n}')"
    return 0
  fi
  SHORT_TPS=""; SHORT_WALL=""; return 1
}

run_variant() {
  local row="$1"
  local endpoint launcher backend mbt model_path spec draft_tokens notes
  IFS='|' read -r endpoint launcher backend mbt model_path spec draft_tokens notes <<< "$row"
  local outdir="$RESULT_DIR/$endpoint" env_file="$RESULT_DIR/$endpoint/lease.env"
  mkdir -p "$outdir"
  cleanup_current
  cleanup_endpoint_leases "$endpoint"
  CURRENT_ENDPOINT="$endpoint"; CURRENT_ENV_FILE="$env_file"; rm -f "$env_file"

  log "======================================================================"
  log "$endpoint -- $notes"
  infer-stack catalog endpoint show "$endpoint" > "$outdir/catalog.yaml" 2>&1 || true
  local start="$(date +%s)"
  if ! timeout "$BOOT_TIMEOUT" infer-stack acquire "$endpoint" --allowed_gpus "$GPU_INDEX" --env-file "$env_file" --no-wait --yes > "$outdir/acquire.log" 2>&1; then
    local boot=$(( $(date +%s)-start ))
    append_result "$endpoint" "$launcher" "$backend" "$mbt" "$model_path" "$spec" "$draft_tokens" acquire_failed "$boot" "" "" "" "" "" "" "" "" "$notes" "acquire failed"
    cleanup_current; return 1
  fi
  if ! wait_with_crash_detection "$endpoint" "$outdir/wait.log"; then
    local boot=$(( $(date +%s)-start )) container="$(find_container "$endpoint")"
    capture_logs "$container" "$outdir/container.log"; parse_boot_facts "$outdir/container.log"
    local err="$(grep -E 'ERROR|ValueError|RuntimeError|OutOfMemory|not supported|requires' "$outdir/container.log" | tail -n 4 | tr '\n' ' ')"
    append_result "$endpoint" "$launcher" "$backend" "$mbt" "$model_path" "$spec" "$draft_tokens" readiness_failed "$boot" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" "" "" "" "" "" "" "$notes" "$err"
    cleanup_current; return 1
  fi

  local boot=$(( $(date +%s)-start )) container="$(find_container "$endpoint")"
  capture_logs "$container" "$outdir/container.log"; parse_boot_facts "$outdir/container.log"
  infer-stack test "$endpoint" --max-tokens 32 --timeout 300 > "$outdir/infer-stack-test.log" 2>&1 || true
  ensure_prompts "$container" || true

  SHORT_TPS=""; SHORT_WALL=""; SMALL_TOK=""; SMALL_WALL=""; MED_TOK=""; MED_WALL=""; err=""
  benchmark_short "$container" "$endpoint" "$outdir" || err="short benchmark failed"
  if [[ -s "$SMALL_PROMPT" ]] && direct_request_prompt "$container" "$endpoint" "$SMALL_PROMPT" 8 "$outdir/small.json" 900; then
    SMALL_TOK="$(jq -r '.usage.prompt_tokens // 0' "$outdir/small.json")"; SMALL_WALL="$(cut -f2 "$outdir/small.json.curl.txt")"
  else
    err="${err:+$err; }small prefill failed"
  fi
  if [[ -s "$MEDIUM_PROMPT" ]] && direct_request_prompt "$container" "$endpoint" "$MEDIUM_PROMPT" 8 "$outdir/medium.json" 1800; then
    MED_TOK="$(jq -r '.usage.prompt_tokens // 0' "$outdir/medium.json")"; MED_WALL="$(cut -f2 "$outdir/medium.json.curl.txt")"
  else
    err="${err:+$err; }medium prefill failed"
  fi

  capture_logs "$container" "$outdir/container.log"
  local status=ok; [[ -n "$err" ]] && status=partial
  append_result "$endpoint" "$launcher" "$backend" "$mbt" "$model_path" "$spec" "$draft_tokens" "$status" "$boot" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" "$SHORT_TPS" "$SHORT_WALL" "$SMALL_TOK" "$SMALL_WALL" "$MED_TOK" "$MED_WALL" "$notes" "$err"
  log "  short=${SHORT_TPS:-NA} tok/s; 8k=${SMALL_WALL:-NA}s; 32k=${MED_WALL:-NA}s; KV=${KV_CACHE_TOKENS:-NA}"
  cleanup_current
  return 0
}

log "Removing active leases for round-2 base endpoints"
for ep in "${ALL_ENDPOINTS[@]}"; do cleanup_endpoint_leases "$ep"; done

for row in "${BASE_VARIANTS[@]}"; do run_variant "$row" || true; done

# Pick fastest successful 32k base arm.
BEST_LINE="$(awk -F'\t' 'NR>1 && ($8=="ok" || $8=="partial") && $17!="" {print $17 "\t" $0}' "$SUMMARY_TSV" | sort -g -k1,1 | head -n1 || true)"
if [[ -z "$BEST_LINE" ]]; then
  log "No base arm completed the 32k probe; skipping follow-up speculative arms."
else
  BEST_ROW="${BEST_LINE#*$'\t'}"
  IFS=$'\t' read -r BEST_EP _ BEST_BACKEND BEST_MBT _ _ _ _ <<< "$BEST_ROW"
  log "Best 32k base arm: $BEST_EP backend=$BEST_BACKEND max-num-batched-tokens=$BEST_MBT"
  backend_slug=triton; [[ "$BEST_BACKEND" == FLEX_ATTENTION ]] && backend_slug=flex

  FOLLOWUPS=(
    "qwen38-rtx8000-${backend_slug}-bt${BEST_MBT}-fastmodel-262k|batch|${BEST_BACKEND}|${BEST_MBT}|/app/models/Qwen3.8-27B-W4A16-AutoRound-fast|off|0|Fast target checkpoint on winning backend/chunk"
  )
  if [[ "$RUN_SPECULATION" == 1 ]]; then
    FOLLOWUPS+=(
      "qwen38-rtx8000-${backend_slug}-bt${BEST_MBT}-mtp2-262k|single|${BEST_BACKEND}|${BEST_MBT}|/app/models/Qwen3.8-27B-W4A16-AutoRound|mtp|2|MTP k=2 on FP16/auto KV full context"
      "qwen38-rtx8000-${backend_slug}-bt${BEST_MBT}-mtp4-262k|single|${BEST_BACKEND}|${BEST_MBT}|/app/models/Qwen3.8-27B-W4A16-AutoRound|mtp|4|MTP k=4 on FP16/auto KV full context"
      "qwen38-rtx8000-${backend_slug}-bt${BEST_MBT}-dflash7-262k|single|${BEST_BACKEND}|${BEST_MBT}|/app/models/Qwen3.8-27B-W4A16-AutoRound|dflash2|7|DFlash2 k=7, unpinned FP16/auto KV full context"
    )
  fi

  for row in "${FOLLOWUPS[@]}"; do
    IFS='|' read -r endpoint launcher backend mbt model_path spec draft_tokens notes <<< "$row"
    log "Catalog follow-up: $endpoint -- $notes"
    add_endpoint "$endpoint" "$launcher" "$backend" "$mbt" "$model_path" "$spec" "$draft_tokens" || true
    ALL_ENDPOINTS+=("$endpoint")
  done
  infer-stack catalog validate || true
  for row in "${FOLLOWUPS[@]}"; do run_variant "$row" || true; done

  # Optional near-full proof: run only the best *successful* no-spec arm, directly against
  # vLLM, with a large timeout. This is off by default because the current Triton sm75
  # prefill may take hours at ~258k even though the context fits in memory.
  if [[ "$RUN_NEARFULL" == 1 ]]; then
    BEST_NOSPEC="$(awk -F'\t' 'NR>1 && ($8=="ok" || $8=="partial") && $6=="off" && $17!="" {print $17 "\t" $1}' "$SUMMARY_TSV" | sort -g -k1,1 | head -n1 | cut -f2 || true)"
    if [[ -n "$BEST_NOSPEC" ]]; then
      log "Near-full proof requested; reacquiring $BEST_NOSPEC"
      outdir="$RESULT_DIR/$BEST_NOSPEC"; env_file="$outdir/nearfull-lease.env"
      cleanup_current; cleanup_endpoint_leases "$BEST_NOSPEC"; CURRENT_ENDPOINT="$BEST_NOSPEC"; CURRENT_ENV_FILE="$env_file"
      infer-stack acquire "$BEST_NOSPEC" --allowed_gpus "$GPU_INDEX" --env-file "$env_file" --no-wait --yes > "$outdir/nearfull-acquire.log" 2>&1 || true
      if wait_with_crash_detection "$BEST_NOSPEC" "$outdir/nearfull-wait.log"; then
        container="$(find_container "$BEST_NOSPEC")"; ensure_prompts "$container" || true
        if [[ -s "$NEARFULL_PROMPT" ]]; then
          log "Starting direct ~${NEARFULL_PROMPT_TOKENS}-token probe; timeout=${NEARFULL_TIMEOUT}s"
          if direct_request_prompt "$container" "$BEST_NOSPEC" "$NEARFULL_PROMPT" 8 "$outdir/nearfull-direct.json" "$NEARFULL_TIMEOUT"; then
            log "Near-full proof succeeded: prompt_tokens=$(jq -r '.usage.prompt_tokens' "$outdir/nearfull-direct.json") wall=$(cut -f2 "$outdir/nearfull-direct.json.curl.txt")s"
          else
            log "Near-full direct probe did not complete successfully; see $outdir/nearfull-direct.json* and container logs."
          fi
        fi
      fi
      cleanup_current
    fi
  fi
fi

python3 - "$SUMMARY_TSV" "$RESULT_DIR/summary.csv" <<'PY'
import csv, sys
src, dst = sys.argv[1:]
with open(src, newline='') as f, open(dst, 'w', newline='') as g:
    r = csv.reader(f, delimiter='\t'); w = csv.writer(g); w.writerows(r)
PY

{
  head -n1 "$SUMMARY_TSV"
  tail -n +2 "$SUMMARY_TSV" | awk -F'\t' 'BEGIN{OFS="\t"} {k=($17==""?1e99:$17); print k,$0}' | sort -g -k1,1 | cut -f2-
} > "$RESULT_DIR/ranking.tsv"

log "======================================================================"
log "Round 2 complete"
log "Summary: $SUMMARY_TSV"
log "Ranking: $RESULT_DIR/ranking.tsv"
log "Logs: $RESULT_DIR"
column -t -s $'\t' "$RESULT_DIR/ranking.tsv" 2>/dev/null || cat "$RESULT_DIR/ranking.tsv"
