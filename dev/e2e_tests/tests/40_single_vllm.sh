# shellcheck shell=bash source=../lib.sh
# The core path: acquire a single vLLM model on the Compose backend, prove it
# serves a real chat completion through the LiteLLM front door, exercise the
# day-2 wrappers against the live stack, then release (reclaim:stop -> gone).
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip single-vllm-acquire 'GPU serving disabled (run with --gpu)'
    skip single-vllm-models 'GPU serving disabled (run with --gpu)'
    skip single-vllm-chat 'GPU serving disabled (run with --gpu)'
    skip single-vllm-day2-ps 'GPU serving disabled (run with --gpu)'
    skip single-vllm-secrets 'GPU serving disabled (run with --gpu)'
    skip single-vllm-release 'GPU serving disabled (run with --gpu)'
    return 0
fi

ENVF="$E2E_RESULTS/single.env"

step single-vllm-acquire 'acquire qwen-small --backend compose --require-generation'
run "infer-stack acquire qwen-small --backend compose --catalog \"$E2E_CAT\" \
      --require-generation --env-file \"$ENVF\" --timeout 1200 --json"
expect_rc 0
expect_out '"ready": true'
note 'duration is the cold-start serve latency (image assumed pre-pulled)'
end_step

step single-vllm-secrets 'managed LITELLM_MASTER_KEY is now fetchable'
run 'infer-stack secrets LITELLM_MASTER_KEY'
expect_rc 0
expect_re '^sk-'
end_step

step single-vllm-models 'the gateway lists qwen-small at the descriptor base_url'
run "set -a; . \"$ENVF\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/models\" -H \"Authorization: Bearer \$OPENAI_API_KEY\""
expect_rc 0
expect_out 'qwen-small'
end_step

step single-vllm-chat 'a real chat completion returns content'
run "set -a; . \"$ENVF\"; set +a; \
     curl -s \"\$OPENAI_BASE_URL/chat/completions\" \
       -H \"Authorization: Bearer \$OPENAI_API_KEY\" \
       -H 'Content-Type: application/json' \
       -d '{\"model\":\"qwen-small\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":8}'"
expect_rc 0
expect_out '"choices"'
expect_no_out 'model=None'
end_step

step single-vllm-day2-ps 'day-2 wrappers target the live leasing stack'
run 'infer-stack ps'
expect_rc 0
expect_re 'vllm|litellm'
end_step

step single-vllm-release 'release tears down a reclaim:stop group'
run "infer-stack release --env-file \"$ENVF\""
expect_rc 0
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
end_step
