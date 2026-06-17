# shellcheck shell=bash source=../lib.sh
# Ollama daemon + lazy tag pull/warmup readiness. Run with GPU 0 free.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip ollama-acquire 'GPU serving disabled (run with --gpu)'
    skip ollama-chat 'GPU serving disabled (run with --gpu)'
    return 0
fi

ENVF="$E2E_RESULTS/ollama.env"

step ollama-acquire 'acquire an ollama endpoint -> daemon up, tag pulled+warmed'
run "infer-stack acquire qwen-ollama --backend compose --catalog \"$E2E_CAT\" \
      --require-generation --timeout 900 --env-file \"$ENVF\" --json"
expect_rc 0
expect_out '"ready": true'
run 'infer-stack logs --tail=120'
expect_re 'pull|qwen2.5'
end_step

step ollama-chat 'a chat completion routes to the ollama daemon'
run "set -a; . \"$ENVF\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/chat/completions\" \
       -H \"Authorization: Bearer \$OPENAI_API_KEY\" -H 'Content-Type: application/json' \
       -d '{\"model\":\"qwen-ollama\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}'"
expect_rc 0
expect_out '"choices"'
run "infer-stack release --env-file \"$ENVF\" >/dev/null 2>&1; true"
end_step
