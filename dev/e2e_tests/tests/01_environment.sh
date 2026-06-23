# shellcheck shell=bash source=../lib.sh
# Environment capture + the bare-minimum preconditions. Always runs.
source "$E2E_ROOT/lib.sh"

step version 'infer-stack is importable and reports a version'
run 'infer-stack version'
expect_rc 0
expect_re 'infer-stack.*[0-9]+\.[0-9]+\.[0-9]+'
end_step

step docker-compose 'docker compose v2 is available'
run 'docker compose version'
expect_rc 0
expect_re 'v?2\.'
end_step

if gpu_enabled; then
    step docker-gpu 'docker can see the GPUs (nvidia runtime)'
    run 'docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi'
    expect_rc 0
    expect_out 'NVIDIA-SMI'
    end_step
else
    skip docker-gpu 'GPU serving disabled (run with --gpu)'
fi
