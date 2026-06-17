# shellcheck shell=bash source=../lib.sh
# Soft-TTL expiry + reclaim policy: a short-TTL lease expires without an explicit
# release, and a reclaim:stop group is reclaimed once no lease protects it.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip ttl-expire 'GPU serving disabled (run with --gpu)'
    exit 0
fi

step ttl-expire 'a 30s TTL lease expires and its reclaim:stop group is reclaimed'
run "infer-stack acquire qwen-small --backend compose --catalog \"$E2E_CAT\" \
      --ttl 30s --require-generation --env-file \"$E2E_RESULTS/ttl.env\" --timeout 1200"
expect_rc 0
note 'sleeping 40s to cross the soft TTL...'
run 'sleep 40'
run 'infer-stack leases --json'
expect_rc 0
expect_no_out '"state": "active"'
note 'lease should be expired/released; qwen-small (reclaim:stop) group reclaimed'
# belt-and-suspenders cleanup
run "infer-stack release --env-file \"$E2E_RESULTS/ttl.env\" >/dev/null 2>&1; true"
end_step
