# shellcheck shell=bash source=../lib.sh
# Explicit GPU pinning of an Ollama daemon to GPU 1 (yardrat's display GPU), the
# 2nd-GPU capability we couldn't exercise before. The daemon must land on GPU 1
# via the docker device reservation (device_ids) and run ON the GPU — NOT fall
# back to CPU, which the old host-index CUDA_VISIBLE_DEVICES did (the reserved
# GPU is renumbered to 0 inside the container, so CUDA_VISIBLE_DEVICES=1 pointed
# at nothing). Placement uses every GPU by default (including the display GPU 1),
# so no extra flag is needed; `config set skip_display_gpus true` would exclude
# it.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip ollama-gpu1-pin 'GPU serving disabled (run with --gpu)'
    skip ollama-gpu1-on-gpu 'GPU serving disabled (run with --gpu)'
    exit 0
fi

COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"

step ollama-gpu1-pin 'daemon pinned to GPU 1: device_ids=[1], no host-index CUDA var'
run "infer-stack acquire smol-ollama-gpu1 --backend compose \
      --catalog \"$E2E_CAT\" --require-generation --timeout 300 --json"
expect_rc 0
expect_out '"ready": true'
run "python3 \"$E2E_ROOT/inspect.py\" \"$COMPOSE\""
expect_rc 0
expect_out "device_ids: ['1']"
# the fix: pin by reservation only; the host-index env var must NOT be set
expect_out 'CUDA_VISIBLE_DEVICES: None'
SVC="$(grep -oE 'ollama-[^ ]+ host_port' "$LAST_OUT_FILE" | head -n1 | cut -d' ' -f1)"
OPORT="$(grep -E 'ollama-.* host_port' "$LAST_OUT_FILE" | awk '{print $NF}' | head -n1)"
note "ollama service: $SVC  host port: $OPORT"
end_step

step ollama-gpu1-on-gpu 'the pinned daemon runs ON the GPU, not a CPU fallback'
# load the model fresh, then ask the daemon which processor it is using
run "curl -s \"http://127.0.0.1:$OPORT/api/generate\" \
      -d '{\"model\":\"smollm2:135m\",\"prompt\":\"hi\",\"stream\":false}' >/dev/null; \
     docker compose -p infer-stack -f \"$COMPOSE\" exec -T \"$SVC\" ollama ps"
expect_rc 0
# `ollama ps` PROCESSOR column shows GPU usage when on-GPU; full CPU fallback
# (the old bug) shows "100% CPU".
expect_out 'GPU'
expect_no_out '100% CPU'
note 'with the old host-index CUDA_VISIBLE_DEVICES bug this read 100% CPU'
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
