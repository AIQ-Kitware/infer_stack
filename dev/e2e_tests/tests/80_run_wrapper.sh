# shellcheck shell=bash source=../lib.sh
# The `run` wrapper — the kwdagger pipeline-node seam: acquire, inject endpoint
# env into a child, release on exit, propagate the child's exit code.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip run-injects-env 'GPU serving disabled (run with --gpu)'
    skip run-exit-code 'GPU serving disabled (run with --gpu)'
    exit 0
fi

step run-injects-env 'run -- <cmd> sees a working OpenAI endpoint, then releases'
run "infer-stack run --endpoint qwen-small --backend compose --catalog \"$E2E_CAT\" \
      --require-generation --timeout 1200 -- \
      bash -c 'curl -s \"\$OPENAI_BASE_URL/chat/completions\" \
        -H \"Authorization: Bearer \$OPENAI_API_KEY\" -H \"Content-Type: application/json\" \
        -d \"{\\\"model\\\":\\\"\$INFER_STACK_ENDPOINT_QWEN_SMALL\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"hi\\\"}],\\\"max_tokens\\\":8}\"'"
expect_rc 0
expect_out '"choices"'
run 'infer-stack leases --json'
expect_no_out '"state": "active"'
note 'lease auto-released on child exit'
end_step

step run-exit-code 'run propagates the child exit code'
run "infer-stack run --endpoint qwen-small --backend compose --catalog \"$E2E_CAT\" \
      --require-generation --timeout 1200 -- bash -c 'exit 7'"
expect_rc 7
end_step
