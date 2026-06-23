# shellcheck shell=bash source=../lib.sh
# Lean --no-litellm Ollama + Open WebUI stack: no gateway at all. Open WebUI
# talks to the daemon's NATIVE Ollama API directly (manage-your-own-models), the
# declared tag is still pulled even without a gateway (the probe_ready fix), and
# a chat hits the daemon's own OpenAI-compatible port. This is the "replace my
# ollama server" path from the manual tutorial. Run with GPU 0 free.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip ollama-lean-serve 'GPU serving disabled (run with --gpu)'
    skip ollama-lean-chat 'GPU serving disabled (run with --gpu)'
    exit 0
fi

COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"

step ollama-lean-serve '--no-litellm: daemon up + tag pulled, no gateway, UI wired direct'
run "infer-stack acquire smol-ollama --no-litellm --backend compose \
      --catalog \"$E2E_CAT\" --require-generation --timeout 300 --json"
expect_rc 0
expect_out '"ready": true'
# the tag was pulled even with NO gateway (probe_ready pulls before the
# litellm-routability check it would otherwise short-circuit on)
run 'infer-stack logs --tail=160'
expect_re 'pull|smollm2'
# rendered compose: no litellm; Open WebUI present and pointed straight at the
# ollama daemon's native API, OpenAI connection off (nothing to point at)
run "python3 \"$E2E_ROOT/inspect.py\" \"$COMPOSE\""
expect_rc 0
expect_out 'has litellm: False'
expect_out 'has open-webui: True'
expect_out 'open-webui host_port: 13000'   # UI front door is up even with no gateway
expect_out 'open-webui ENABLE_OLLAMA_API: True'
expect_out 'open-webui ENABLE_OPENAI_API: False'
expect_re 'open-webui OLLAMA_BASE_URL: http://ollama-'
note 'lean stack: Open WebUI pulls/runs models straight from the ollama daemon'
end_step

step ollama-lean-chat 'chat hits the ollama daemon directly (no gateway)'
# pull the daemon host port out of the rendered compose
run "python3 \"$E2E_ROOT/inspect.py\" \"$COMPOSE\" | grep -E 'ollama-.* host_port'"
expect_rc 0
OPORT="$(awk '{print $NF}' "$LAST_OUT_FILE" | head -n1)"
note "ollama daemon host port: $OPORT"
# the daemon's own OpenAI-compatible endpoint; model is the TAG, not the alias
run "curl -s \"http://127.0.0.1:$OPORT/v1/chat/completions\" \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"smollm2:135m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}'"
expect_rc 0
expect_out '"choices"'
note 'chat answered straight from the daemon — no gateway in the path'
# status still works with no gateway (holistic overview, not the UI URL)
run 'infer-stack status'
expect_rc 0
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
