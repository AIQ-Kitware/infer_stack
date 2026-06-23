# shellcheck shell=bash source=../lib.sh
# Ollama daemon + lazy tag pull/warmup readiness. Run with GPU 0 free.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip ollama-acquire 'GPU serving disabled (run with --gpu)'
    skip ollama-webui-wiring 'GPU serving disabled (run with --gpu)'
    skip ollama-chat 'GPU serving disabled (run with --gpu)'
    exit 0
fi

ENVF="$E2E_RESULTS/ollama.env"
COMPOSE="$INFER_STACK_DATA_DIR/leasing/compose/docker-compose.yml"

step ollama-acquire 'acquire an ollama endpoint -> daemon up, tag pulled+warmed'
run "infer-stack acquire smol-ollama --backend compose --catalog \"$E2E_CAT\" \
      --require-generation --timeout 300 --env-file \"$ENVF\" --json"
expect_rc 0
expect_out '"ready": true'
run 'infer-stack logs --tail=120'
expect_re 'pull|smollm2'
end_step

step ollama-webui-wiring 'with the gateway on, Open WebUI gets BOTH connections'
# OpenAI (chat) -> the litellm gateway; native Ollama (pull/manage) -> the daemon
run "python3 \"$E2E_ROOT/inspect.py\" \"$COMPOSE\""
expect_rc 0
expect_out 'has litellm: True'
expect_out 'has open-webui: True'
expect_out 'open-webui OPENAI_API_BASE_URL: http://litellm:4000/v1'
expect_out 'open-webui ENABLE_OLLAMA_API: True'
expect_re 'open-webui OLLAMA_BASE_URL: http://ollama-'
note 'declared models route via the gateway; ad-hoc pulls show via the ollama conn'
end_step

step ollama-chat 'a chat completion routes to the ollama daemon'
run "set -a; . \"$ENVF\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/chat/completions\" \
       -H \"Authorization: Bearer \$OPENAI_API_KEY\" -H 'Content-Type: application/json' \
       -d '{\"model\":\"smol-ollama\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}'"
expect_rc 0
expect_out '"choices"'
run "infer-stack release --env-file \"$ENVF\" >/dev/null 2>&1; true"
end_step
