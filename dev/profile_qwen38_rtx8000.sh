#!/usr/bin/env bash
# Profile Qwen3.8-27B HyperQwen variants on a Quadro RTX 8000 using infer-stack.
#
# This script is intentionally failure-tolerant:
#   * every serving configuration is a separate infer-stack endpoint;
#   * variants are acquired one at a time on the RTX 8000;
#   * a failed boot/readiness/API/long-context probe is recorded and the matrix continues;
#   * every lease created by this script is released before moving on;
#   * endpoints remain in the catalog after the run for manual follow-up.
#
# It only releases active leases whose endpoint names are in VARIANTS below. It does not
# release unrelated infer-stack workloads.

set -uo pipefail

BASE_URL="${BASE_URL:-0.0.0.0:14042}"
MODEL_NAME="qwen3.8-27b-dbirks-hyperqwen"
MODEL_SOURCE="hf://dbirks/Qwen3.8-27B-W4A16-AutoRound"
IMAGE="ghcr.io/syv-ai/hyperqwen:sha-684e927"
GPU_NAME="Quadro RTX 8000"
GPU_UTIL="${GPU_UTIL:-0.90}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1800}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
SHORT_RUNS="${SHORT_RUNS:-3}"
RUN_CONTEXT_PROBES="${RUN_CONTEXT_PROBES:-1}"
MEDIUM_PROMPT_TOKENS="${MEDIUM_PROMPT_TOKENS:-32000}"
NEAR_FULL_PROMPT_TOKENS="${NEAR_FULL_PROMPT_TOKENS:-260000}"
RESULT_DIR="${RESULT_DIR:-$PWD/dev/benchmark-results/qwen38-rtx8000-$(date +%Y%m%dT%H%M%S)}"

mkdir -p "$RESULT_DIR"
SUMMARY_TSV="$RESULT_DIR/summary.tsv"
SUMMARY_CSV="$RESULT_DIR/summary.csv"
RUN_LOG="$RESULT_DIR/run.log"

exec > >(tee -a "$RUN_LOG") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "ERROR: required command not found: $1"
        exit 2
    fi
}

for cmd in infer-stack nvidia-smi docker curl jq awk sed grep timeout python3; do
    need_cmd "$cmd"
done

GPU_INDEX="$({
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
} | awk -F',' -v want="$GPU_NAME" '
    index($2, want) {
        gsub(/[[:space:]]/, "", $1)
        print $1
        exit
    }
')"

if [[ -z "$GPU_INDEX" ]]; then
    log "ERROR: could not find a GPU named '$GPU_NAME'."
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    exit 2
fi

log "Using GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,power.limit \
    --format=csv,noheader || true

{
    echo "date=$(date --iso-8601=seconds)"
    echo "base_url=$BASE_URL"
    echo "gpu_index=$GPU_INDEX"
    echo "gpu_name=$GPU_NAME"
    echo "gpu_util=$GPU_UTIL"
    echo "infer_stack_version=$(infer-stack --version 2>/dev/null || true)"
    echo "infer_stack_git=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "host=$(hostname)"
    nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader || true
} > "$RESULT_DIR/metadata.txt"

API_KEY="$(infer-stack env LITELLM_MASTER_KEY 2>/dev/null || true)"
if [[ -n "$API_KEY" ]]; then
    CURL_AUTH=(-H "Authorization: Bearer $API_KEY")
else
    CURL_AUTH=()
    log "WARNING: infer-stack env LITELLM_MASTER_KEY was empty; API calls will be unauthenticated."
fi

# endpoint|launcher|kv|activation|layers|max_len|ctx|spec|extra_args|notes
#
# The first three auto-KV endpoints deliberately use KV=fp8 in the HyperQwen batch launcher
# and override the launcher's flag with --kv-cache-dtype=auto. The pinned HyperQwen image has
# no batch KV=auto selector. vLLM warns about the duplicate flag, but EXTRA_ARGS expands last;
# this is the exact mechanism already proven to select FP16/auto KV on the RTX 8000.
#
# sm75 is expected to reject or slow some FP8/speculative paths. Those are included because
# establishing the failure boundary is useful and the harness continues automatically.
VARIANTS=(
'qwen38-rtx8000-fp16kv-262k|batch|fp8|off||262144|||--served-model-name={served_model_name} --dtype=half --kv-cache-dtype=auto|W4A16, auto/FP16 KV, no speculation; known-good baseline'
'qwen38-rtx8000-fp16kv-gateup-int8-262k|batch|fp8|int8|gate_up|262144|||--served-model-name={served_model_name} --dtype=half --kv-cache-dtype=auto|W4A8 gate_up only, auto/FP16 KV'
'qwen38-rtx8000-fp16kv-mlp-int8-262k|batch|fp8|int8|mlp|262144|||--served-model-name={served_model_name} --dtype=half --kv-cache-dtype=auto|W4A8 MLP, auto/FP16 KV'
'qwen38-rtx8000-fp8kv-262k|batch|fp8|off||262144|||--served-model-name={served_model_name} --dtype=half|FP8 KV attempt on sm75; may fail backend selection'
'qwen38-rtx8000-int4kv-262k|batch|int4pth|off||262144|||--served-model-name={served_model_name} --dtype=half|int4 per-token-head KV, W4A16'
'qwen38-rtx8000-int4kv-mlp-int8-262k|batch|int4pth|int8|mlp|262144|||--served-model-name={served_model_name} --dtype=half|int4 per-token-head KV plus W4A8 MLP'
'qwen38-rtx8000-kvarn-262k|batch|kvarn|off||262144|||--served-model-name={served_model_name} --dtype=half|KVarN KV, W4A16, no speculation'
'qwen38-rtx8000-kvarn-mlp-int8-262k|batch|kvarn|int8|mlp|262144|||--served-model-name={served_model_name} --dtype=half|KVarN KV plus W4A8 MLP, no speculation'
'qwen38-rtx8000-mtp-kvarn-262k|single||off||262144|huge|mtp|--served-model-name={served_model_name} --dtype=half|MTP + KVarN at full 262144; risky on sm75'
'qwen38-rtx8000-mtp-kvarn-mlp-int8-262k|single||int8|mlp|262144|huge|mtp|--served-model-name={served_model_name} --dtype=half|MTP + KVarN + W4A8 MLP at full 262144; risky on sm75'
'qwen38-rtx8000-dflash2-kvarn-245k|single||off||245760|huge|dflash2|--served-model-name={served_model_name} --dtype=half|HyperQwen reference huge DFlash2 profile length; sm75 compatibility probe'
'qwen38-rtx8000-dflash2-kvarn-262k|single||off||262144|huge|dflash2|--served-model-name={served_model_name} --dtype=half|DFlash2 + KVarN pushed to full 262144; risky on sm75'
'qwen38-rtx8000-dflash2-kvarn-mlp-int8-262k|single||int8|mlp|262144|huge|dflash2|--served-model-name={served_model_name} --dtype=half|DFlash2 + KVarN + W4A8 MLP at full 262144; risky on sm75'
)

endpoint_names() {
    local row endpoint
    for row in "${VARIANTS[@]}"; do
        IFS='|' read -r endpoint _ <<< "$row"
        printf '%s\n' "$endpoint"
    done
}

cleanup_endpoint_leases() {
    local endpoint="$1"
    local ids id
    ids="$(infer-stack leases --json 2>/dev/null | jq -r --arg ep "$endpoint" '
        .leases[]? |
        select(.state == "active") |
        select((.endpoints // []) | index($ep)) |
        .id
    ' 2>/dev/null || true)"
    while IFS= read -r id; do
        [[ -n "$id" ]] || continue
        log "Releasing pre-existing lease $id for $endpoint"
        infer-stack release "$id" --evict --yes >/dev/null 2>&1 || true
    done <<< "$ids"
}

cleanup_all_experiment_leases() {
    local endpoint
    while IFS= read -r endpoint; do
        cleanup_endpoint_leases "$endpoint"
    done < <(endpoint_names)
}

CURRENT_ENDPOINT=""
CURRENT_ENV_FILE=""
cleanup_current() {
    if [[ -n "$CURRENT_ENV_FILE" && -f "$CURRENT_ENV_FILE" ]]; then
        infer-stack release --env-file "$CURRENT_ENV_FILE" --evict --yes >/dev/null 2>&1 || true
    fi
    if [[ -n "$CURRENT_ENDPOINT" ]]; then
        cleanup_endpoint_leases "$CURRENT_ENDPOINT"
    fi
    CURRENT_ENDPOINT=""
    CURRENT_ENV_FILE=""
}
trap cleanup_current EXIT INT TERM

add_endpoint() {
    local endpoint="$1"
    local launcher="$2"
    local kv="$3"
    local activation="$4"
    local layers="$5"
    local max_len="$6"
    local ctx="$7"
    local spec="$8"
    local extra_args="$9"

    local activation_value=""
    local layers_value=""
    if [[ "$activation" == "int8" ]]; then
        activation_value="int8"
        layers_value="$layers"
    fi

    local env_yaml
    if [[ "$launcher" == "batch" ]]; then
        env_yaml="{PORT: \"{port}\", VERIFY: 0, KV: \"$kv\", PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"$activation_value\", INT8_LAYERS: \"$layers_value\", EXTRA_ARGS: \"$extra_args\"}"
    else
        env_yaml="{PORT: \"{port}\", VERIFY: 0, CTX: \"$ctx\", SPEC: \"$spec\", PREFIX_CACHE: 1, MAX_LEN: \"{max_model_len}\", MAX_SEQS: 1, GPU_UTIL: \"{gpu_memory_utilization}\", INT8_ACT: \"$activation_value\", INT8_LAYERS: \"$layers_value\", EXTRA_ARGS: \"$extra_args\"}"
    fi

    infer-stack catalog endpoint add \
        "$endpoint" \
        --engine vllm \
        --model "$MODEL_NAME" \
        --min-vram-gib 40 \
        --gpu "$GPU_INDEX" \
        --max-model-len "$max_len" \
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

log "Ensuring model entry exists"
infer-stack catalog model add \
    "$MODEL_NAME" \
    --source "$MODEL_SOURCE" \
    --force

log "Removing active leases for this experiment namespace only"
cleanup_all_experiment_leases

log "Adding/updating ${#VARIANTS[@]} experiment endpoints"
for row in "${VARIANTS[@]}"; do
    IFS='|' read -r endpoint launcher kv activation layers max_len ctx spec extra_args notes <<< "$row"
    log "Catalog: $endpoint -- $notes"
    if ! add_endpoint "$endpoint" "$launcher" "$kv" "$activation" "$layers" "$max_len" "$ctx" "$spec" "$extra_args"; then
        log "WARNING: failed to add endpoint $endpoint; it will be recorded as catalog_failed."
    fi
done

if ! infer-stack catalog validate; then
    log "ERROR: catalog validation failed. Refusing to launch variants."
    exit 3
fi

printf '%s\n' \
'endpoint	launcher	kv	activation	layers	max_len	ctx	spec	status	boot_seconds	attention_backend	kv_cache_tokens	max_concurrency	short_tok_s_mean	short_wall_s_mean	medium_prompt_tokens	medium_wall_s	nearfull_prompt_tokens	nearfull_wall_s	gpu_mem_used_mib	gpu_mem_free_mib	notes	error' \
> "$SUMMARY_TSV"

sanitize_field() {
    printf '%s' "$1" | tr '\t\r\n' '   '
}

append_result() {
    local endpoint="$1" launcher="$2" kv="$3" activation="$4" layers="$5" max_len="$6" ctx="$7" spec="$8"
    local status="$9" boot_seconds="${10}" attention="${11}" kv_tokens="${12}" max_conc="${13}"
    local short_tps="${14}" short_wall="${15}" medium_tokens="${16}" medium_wall="${17}"
    local near_tokens="${18}" near_wall="${19}" mem_used="${20}" mem_free="${21}" notes="${22}" error="${23}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(sanitize_field "$endpoint")" \
        "$(sanitize_field "$launcher")" \
        "$(sanitize_field "$kv")" \
        "$(sanitize_field "$activation")" \
        "$(sanitize_field "$layers")" \
        "$(sanitize_field "$max_len")" \
        "$(sanitize_field "$ctx")" \
        "$(sanitize_field "$spec")" \
        "$(sanitize_field "$status")" \
        "$(sanitize_field "$boot_seconds")" \
        "$(sanitize_field "$attention")" \
        "$(sanitize_field "$kv_tokens")" \
        "$(sanitize_field "$max_conc")" \
        "$(sanitize_field "$short_tps")" \
        "$(sanitize_field "$short_wall")" \
        "$(sanitize_field "$medium_tokens")" \
        "$(sanitize_field "$medium_wall")" \
        "$(sanitize_field "$near_tokens")" \
        "$(sanitize_field "$near_wall")" \
        "$(sanitize_field "$mem_used")" \
        "$(sanitize_field "$mem_free")" \
        "$(sanitize_field "$notes")" \
        "$(sanitize_field "$error")" \
        >> "$SUMMARY_TSV"
}

find_container() {
    local endpoint="$1"
    docker ps -a \
        --filter "name=infer-stack-vllm-$endpoint" \
        --format '{{.Names}}' | head -n 1
}

capture_logs() {
    local endpoint="$1" container="$2" out="$3"
    if [[ -n "$container" ]]; then
        docker logs "$container" > "$out" 2>&1 || true
    else
        printf 'No container found for %s\n' "$endpoint" > "$out"
    fi
}

parse_boot_facts() {
    local logfile="$1"
    ATTENTION_BACKEND="$(grep -Eo 'Using [A-Z0-9_]+ attention backend' "$logfile" | tail -n 1 | awk '{print $2}' || true)"
    KV_CACHE_TOKENS="$(grep -Eo 'GPU KV cache size: [0-9,]+ tokens' "$logfile" | tail -n 1 | sed -E 's/.*size: ([0-9,]+) tokens/\1/' | tr -d ',' || true)"
    MAX_CONCURRENCY="$(grep -Eo 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$logfile" | tail -n 1 | sed -E 's/.*request: ([0-9.]+)x/\1/' || true)"
}

api_request() {
    local endpoint="$1" prompt="$2" max_tokens="$3" response_file="$4" timeout_s="$5"
    local request_file="$response_file.request.json"
    local meta_file="$response_file.curl.txt"

    jq -n \
        --arg model "$endpoint" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        '{
          model: $model,
          messages: [{role: "user", content: $prompt}],
          temperature: 0,
          max_tokens: $max_tokens
        }' > "$request_file"

    curl -sS \
        --max-time "$timeout_s" \
        -o "$response_file" \
        -w '%{http_code}\t%{time_total}\n' \
        "$BASE_URL/chat/completions" \
        "${CURL_AUTH[@]}" \
        -H 'Content-Type: application/json' \
        --data-binary "@$request_file" \
        > "$meta_file"
}

api_request_file_prompt() {
    local endpoint="$1" prompt_file="$2" max_tokens="$3" response_file="$4" timeout_s="$5"
    local request_file="$response_file.request.json"
    local meta_file="$response_file.curl.txt"

    jq -n \
        --arg model "$endpoint" \
        --rawfile prompt "$prompt_file" \
        --argjson max_tokens "$max_tokens" \
        '{
          model: $model,
          messages: [{role: "user", content: $prompt}],
          temperature: 0,
          max_tokens: $max_tokens
        }' > "$request_file"

    curl -sS \
        --max-time "$timeout_s" \
        -o "$response_file" \
        -w '%{http_code}\t%{time_total}\n' \
        "$BASE_URL/chat/completions" \
        "${CURL_AUTH[@]}" \
        -H 'Content-Type: application/json' \
        --data-binary "@$request_file" \
        > "$meta_file"
}

request_ok() {
    local response_file="$1"
    local meta_file="$response_file.curl.txt"
    [[ -s "$response_file" && -s "$meta_file" ]] || return 1
    local code
    code="$(cut -f1 "$meta_file" 2>/dev/null || true)"
    [[ "$code" == "200" ]] || return 1
    jq -e '.choices | length > 0' "$response_file" >/dev/null 2>&1
}

REQUESTED_CONTEXT_PROMPTS=0
MEDIUM_PROMPT_FILE="$RESULT_DIR/prompt-${MEDIUM_PROMPT_TOKENS}.txt"
NEAR_FULL_PROMPT_FILE="$RESULT_DIR/prompt-${NEAR_FULL_PROMPT_TOKENS}.txt"

make_exactish_prompt() {
    local container="$1" target_tokens="$2" tag="$3" outfile="$4"
    log "Generating a tokenizer-measured ~${target_tokens}-token '$tag' prompt once"
    docker exec -i "$container" /app/venv/bin/python - "$target_tokens" "$tag" <<'PY' > "$outfile"
import sys
from transformers import AutoTokenizer

target = int(sys.argv[1])
tag = sys.argv[2]
tok = AutoTokenizer.from_pretrained('/app/models/Qwen3.8-27B-W4A16-AutoRound')
unit = (
    f"[{tag}] The database page stores ordered keys, child pointers, transaction metadata, "
    "checksums, and recovery information. This sentence exists only to exercise long context.\n"
)

def n_tokens(n):
    return len(tok.encode(unit * n, add_special_tokens=False))

u = max(1, n_tokens(1))
lo = max(1, target // u // 2)
hi = max(lo + 1, target // u * 2 + 8)
while n_tokens(hi) < target:
    lo = hi
    hi *= 2
while lo + 1 < hi:
    mid = (lo + hi) // 2
    if n_tokens(mid) <= target:
        lo = mid
    else:
        hi = mid
text = unit * lo
count = len(tok.encode(text, add_special_tokens=False))
# Put the measured count into the text itself only as a tiny suffix; the API usage field
# remains the authoritative final count after chat-template tokens are added.
text += f"\n[{tag} target={target} generated_body_tokens={count}]\n"
sys.stdout.write(text)
PY
}

ensure_context_prompts() {
    local container="$1"
    [[ "$RUN_CONTEXT_PROBES" == "1" ]] || return 0
    if [[ ! -s "$MEDIUM_PROMPT_FILE" ]]; then
        make_exactish_prompt "$container" "$MEDIUM_PROMPT_TOKENS" medium "$MEDIUM_PROMPT_FILE" || return 1
    fi
    if [[ ! -s "$NEAR_FULL_PROMPT_FILE" ]]; then
        make_exactish_prompt "$container" "$NEAR_FULL_PROMPT_TOKENS" nearfull "$NEAR_FULL_PROMPT_FILE" || return 1
    fi
    REQUESTED_CONTEXT_PROMPTS=1
}

benchmark_short() {
    local endpoint="$1" outdir="$2"
    local prompt='Explain in technical detail how virtual memory, page tables, TLBs, page faults, and memory-mapped files interact in a modern operating system. Continue until the token limit.'
    local i response wall tokens
    local tps_values=() wall_values=()

    # One warm request before timed runs. A failure here is informative but does not stop
    # the timed attempts; transient first-request compilation can happen on exotic paths.
    api_request "$endpoint" "$prompt" 64 "$outdir/warmup.json" 600 || true

    for ((i = 1; i <= SHORT_RUNS; i++)); do
        response="$outdir/short-$i.json"
        if ! api_request "$endpoint" "$prompt" 512 "$response" 900; then
            log "  short run $i: curl failed"
            continue
        fi
        if ! request_ok "$response"; then
            log "  short run $i: HTTP/API failure"
            continue
        fi
        wall="$(cut -f2 "$response.curl.txt")"
        tokens="$(jq -r '.usage.completion_tokens // 0' "$response")"
        if awk -v w="$wall" 'BEGIN {exit !(w > 0)}'; then
            tps_values+=("$(awk -v t="$tokens" -v w="$wall" 'BEGIN {printf "%.3f", t / w}')")
            wall_values+=("$wall")
        fi
    done

    if ((${#tps_values[@]} == 0)); then
        SHORT_TPS=""
        SHORT_WALL=""
        return 1
    fi
    SHORT_TPS="$(printf '%s\n' "${tps_values[@]}" | awk '{s+=$1;n++} END {if(n) printf "%.3f", s/n}')"
    SHORT_WALL="$(printf '%s\n' "${wall_values[@]}" | awk '{s+=$1;n++} END {if(n) printf "%.3f", s/n}')"
    return 0
}

probe_context() {
    local endpoint="$1" prompt_file="$2" max_tokens="$3" response="$4" timeout_s="$5"
    if ! api_request_file_prompt "$endpoint" "$prompt_file" "$max_tokens" "$response" "$timeout_s"; then
        return 1
    fi
    request_ok "$response"
}

wait_with_crash_detection() {
    local endpoint="$1" wait_log="$2"
    local deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
    local wait_pid container status restarts

    infer-stack wait \
        "$endpoint" \
        --timeout "$BOOT_TIMEOUT" \
        --interval "$WAIT_INTERVAL" \
        > "$wait_log" 2>&1 &
    wait_pid=$!

    while kill -0 "$wait_pid" >/dev/null 2>&1; do
        if (( $(date +%s) >= deadline )); then
            kill "$wait_pid" >/dev/null 2>&1 || true
            wait "$wait_pid" >/dev/null 2>&1 || true
            echo "timeout waiting for readiness" >> "$wait_log"
            return 1
        fi

        container="$(find_container "$endpoint")"
        if [[ -n "$container" ]]; then
            status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
            restarts="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
            if [[ "$status" == "restarting" || "$status" == "dead" || "$status" == "exited" ]]; then
                # Give Docker one brief chance to settle, then classify a repeated crash as a
                # variant failure instead of spending the full boot timeout in a restart loop.
                sleep 8
                status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
                restarts="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
                if [[ "$status" == "restarting" || "$status" == "dead" || "$status" == "exited" ]]; then
                    echo "container remained in failure state: status=$status restarts=$restarts" >> "$wait_log"
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

variant_number=0
for row in "${VARIANTS[@]}"; do
    variant_number=$((variant_number + 1))
    IFS='|' read -r endpoint launcher kv activation layers max_len ctx spec extra_args notes <<< "$row"

    outdir="$RESULT_DIR/$endpoint"
    mkdir -p "$outdir"
    env_file="$outdir/lease.env"
    acquire_log="$outdir/acquire.log"
    wait_log="$outdir/wait.log"
    boot_log="$outdir/container.log"

    log "======================================================================"
    log "[$variant_number/${#VARIANTS[@]}] $endpoint"
    log "$notes"
    infer-stack catalog endpoint show "$endpoint" > "$outdir/catalog.yaml" 2>&1 || true

    cleanup_current
    cleanup_endpoint_leases "$endpoint"
    CURRENT_ENDPOINT="$endpoint"
    CURRENT_ENV_FILE="$env_file"
    rm -f "$env_file"

    start_epoch="$(date +%s)"
    if ! timeout "$BOOT_TIMEOUT" infer-stack acquire \
            "$endpoint" \
            --allowed_gpus "$GPU_INDEX" \
            --env-file "$env_file" \
            --no-wait \
            --yes \
            > "$acquire_log" 2>&1; then
        boot_seconds="$(( $(date +%s) - start_epoch ))"
        container="$(find_container "$endpoint")"
        capture_logs "$endpoint" "$container" "$boot_log"
        error="acquire failed: $(tail -n 5 "$acquire_log" | tr '\n' ' ')"
        append_result "$endpoint" "$launcher" "$kv" "$activation" "$layers" "$max_len" "$ctx" "$spec" \
            acquire_failed "$boot_seconds" "" "" "" "" "" "" "" "" "" "" "" "$notes" "$error"
        log "FAILED acquire: $error"
        cleanup_current
        continue
    fi

    if ! wait_with_crash_detection "$endpoint" "$wait_log"; then
        boot_seconds="$(( $(date +%s) - start_epoch ))"
        container="$(find_container "$endpoint")"
        capture_logs "$endpoint" "$container" "$boot_log"
        parse_boot_facts "$boot_log"
        error="readiness failed: $(tail -n 5 "$wait_log" | tr '\n' ' ') | $(grep -E 'ERROR|ValueError|RuntimeError|OutOfMemory|not supported|requires' "$boot_log" | tail -n 3 | tr '\n' ' ')"
        append_result "$endpoint" "$launcher" "$kv" "$activation" "$layers" "$max_len" "$ctx" "$spec" \
            readiness_failed "$boot_seconds" "$ATTENTION_BACKEND" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" "" "" "" "" "" "" "" "" "$notes" "$error"
        log "FAILED readiness: $error"
        cleanup_current
        continue
    fi

    boot_seconds="$(( $(date +%s) - start_epoch ))"
    container="$(find_container "$endpoint")"
    capture_logs "$endpoint" "$container" "$boot_log"
    parse_boot_facts "$boot_log"

    if ! infer-stack test "$endpoint" --max-tokens 64 --timeout 300 > "$outdir/infer-stack-test.log" 2>&1; then
        error="infer-stack test failed: $(tail -n 5 "$outdir/infer-stack-test.log" | tr '\n' ' ')"
        append_result "$endpoint" "$launcher" "$kv" "$activation" "$layers" "$max_len" "$ctx" "$spec" \
            api_test_failed "$boot_seconds" "$ATTENTION_BACKEND" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" "" "" "" "" "" "" "" "" "$notes" "$error"
        log "FAILED API test: $error"
        cleanup_current
        continue
    fi

    SHORT_TPS=""
    SHORT_WALL=""
    if benchmark_short "$endpoint" "$outdir"; then
        log "  short decode mean: $SHORT_TPS completion tokens/s, wall $SHORT_WALL s"
    else
        log "  WARNING: no successful short benchmark runs"
    fi

    MEDIUM_TOKENS=""
    MEDIUM_WALL=""
    NEAR_TOKENS=""
    NEAR_WALL=""
    context_error=""

    if [[ "$RUN_CONTEXT_PROBES" == "1" ]]; then
        if ensure_context_prompts "$container"; then
            if probe_context "$endpoint" "$MEDIUM_PROMPT_FILE" 32 "$outdir/medium.json" 1800; then
                MEDIUM_TOKENS="$(jq -r '.usage.prompt_tokens // 0' "$outdir/medium.json")"
                MEDIUM_WALL="$(cut -f2 "$outdir/medium.json.curl.txt")"
                log "  medium probe: prompt_tokens=$MEDIUM_TOKENS wall=${MEDIUM_WALL}s"
            else
                context_error="medium context probe failed"
                log "  WARNING: medium context probe failed"
            fi

            # Only send the near-full probe when the endpoint declares enough room for it.
            # The 245760 DFlash2 reference arm gets a 230k body instead of the 250k body.
            near_prompt="$NEAR_FULL_PROMPT_FILE"
            if (( max_len < NEAR_FULL_PROMPT_TOKENS + 512 )); then
                reference_prompt="$RESULT_DIR/prompt-230000.txt"
                if [[ ! -s "$reference_prompt" ]]; then
                    make_exactish_prompt "$container" 230000 reference230k "$reference_prompt" || true
                fi
                near_prompt="$reference_prompt"
            fi
            if [[ -s "$near_prompt" ]] && probe_context "$endpoint" "$near_prompt" 16 "$outdir/nearfull.json" 3600; then
                NEAR_TOKENS="$(jq -r '.usage.prompt_tokens // 0' "$outdir/nearfull.json")"
                NEAR_WALL="$(cut -f2 "$outdir/nearfull.json.curl.txt")"
                log "  near-full probe: prompt_tokens=$NEAR_TOKENS wall=${NEAR_WALL}s"
            else
                context_error="${context_error:+$context_error; }near-full context probe failed"
                log "  WARNING: near-full context probe failed"
            fi
        else
            context_error="failed to generate tokenizer-measured context prompts"
            log "  WARNING: $context_error"
        fi
    fi

    mem_line="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    MEM_USED="$(printf '%s' "$mem_line" | cut -d, -f1 | tr -d ' ' || true)"
    MEM_FREE="$(printf '%s' "$mem_line" | cut -d, -f2 | tr -d ' ' || true)"

    status="ok"
    error="$context_error"
    if [[ -z "$SHORT_TPS" ]]; then
        status="benchmark_partial"
        error="${error:+$error; }short benchmark failed"
    elif [[ "$RUN_CONTEXT_PROBES" == "1" && -z "$NEAR_TOKENS" ]]; then
        status="benchmark_partial"
    fi

    capture_logs "$endpoint" "$container" "$boot_log"
    parse_boot_facts "$boot_log"

    append_result "$endpoint" "$launcher" "$kv" "$activation" "$layers" "$max_len" "$ctx" "$spec" \
        "$status" "$boot_seconds" "$ATTENTION_BACKEND" "$KV_CACHE_TOKENS" "$MAX_CONCURRENCY" \
        "$SHORT_TPS" "$SHORT_WALL" "$MEDIUM_TOKENS" "$MEDIUM_WALL" "$NEAR_TOKENS" "$NEAR_WALL" \
        "$MEM_USED" "$MEM_FREE" "$notes" "$error"

    cleanup_current
    sleep 2
done

cleanup_current

# Convert TSV to CSV without depending on Python packages.
RANKING_TSV="$RESULT_DIR/ranking.tsv"
python3 - "$SUMMARY_TSV" "$SUMMARY_CSV" "$RANKING_TSV" <<'PY'
import csv, sys
src, dst, ranking = sys.argv[1:]
with open(src, newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
with open(dst, 'w', newline='') as g:
    w = csv.DictWriter(g, fieldnames=rows[0].keys() if rows else [], extrasaction='ignore')
    if rows:
        w.writeheader(); w.writerows(rows)

def fnum(v, default=-1.0):
    try:
        return float(v)
    except Exception:
        return default

def rank_key(r):
    # Context capacity is the primary goal: a successful ~260k probe outranks a
    # faster 230k reference profile. Within equal context reach, prefer complete
    # runs and then higher short-prompt decode throughput.
    near_tokens = fnum(r.get('nearfull_prompt_tokens'))
    medium = 1 if fnum(r.get('medium_prompt_tokens')) >= 30000 else 0
    ok = 1 if r.get('status') == 'ok' else 0
    return (near_tokens, medium, ok, fnum(r.get('short_tok_s_mean')))

ranked = sorted(rows, key=rank_key, reverse=True)
fields = [
    'endpoint', 'status', 'short_tok_s_mean', 'short_wall_s_mean',
    'medium_prompt_tokens', 'medium_wall_s', 'nearfull_prompt_tokens',
    'nearfull_wall_s', 'attention_backend', 'kv_cache_tokens',
    'max_concurrency', 'boot_seconds', 'error'
]
with open(ranking, 'w', newline='') as g:
    w = csv.DictWriter(g, fieldnames=fields, delimiter='\t', extrasaction='ignore')
    w.writeheader(); w.writerows(ranked)
PY

log "======================================================================"
log "Experiment complete"
log "TSV: $SUMMARY_TSV"
log "CSV: $SUMMARY_CSV"
log "Ranking: $RANKING_TSV"
log "Logs/responses: $RESULT_DIR"
log "All experiment leases have been released; endpoints remain in the catalog."

echo
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$RANKING_TSV" | sed -n '1,40p'
else
    cat "$RANKING_TSV"
fi
