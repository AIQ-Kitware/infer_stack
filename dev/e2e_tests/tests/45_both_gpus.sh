# shellcheck shell=bash source=../lib.sh
# Both GPUs at once. Placement uses every GPU by default — including yardrat's
# display-attached GPU 1 — so two distinct models land on two distinct GPUs with
# no extra flag (`config set skip_display_gpus true` would exclude the display
# GPU). Proves the multi-GPU spread.
source "$E2E_ROOT/lib.sh"

if ! gpu_enabled; then
    skip both-gpus-spread 'GPU serving disabled (run with --gpu)'
    exit 0
fi

SIDECAR="$INFER_STACK_DATA_DIR/leasing/compose/leasing-compose-state.json"

step both-gpus-spread 'two distinct models spread across GPU 0 and GPU 1'
run "infer-stack acquire smol-135 --backend compose --catalog \"$E2E_CAT\" \
      --owner g0 --require-generation --timeout 1200"
expect_rc 0
run "infer-stack acquire smol-360 --backend compose --catalog \"$E2E_CAT\" \
      --owner g1 --require-generation --timeout 1200"
expect_rc 0
run "python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
a = d.get(\"assignments\", {})
gpus = sorted({g for v in a.values() for g in v})
print(\"assignments:\", a)
print(\"distinct_gpus:\", gpus)
sys.exit(0 if len(gpus) >= 2 else 1)
' \"$SIDECAR\""
expect_rc 0
expect_out 'distinct_gpus: [0, 1]'
note 'if this fails placement (only GPU 0 used) the display-GPU override regressed'
end_step

step both-gpus-both-routable 'both endpoints answer through the one gateway'
run "key=\$(infer-stack env LITELLM_MASTER_KEY); base=http://127.0.0.1:14042/v1; \
     for m in smol-135 smol-360; do \
       echo \"== \$m ==\"; \
       curl -s \"\$base/chat/completions\" -H \"Authorization: Bearer \$key\" \
         -H 'Content-Type: application/json' \
         -d \"{\\\"model\\\":\\\"\$m\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"hi\\\"}],\\\"max_tokens\\\":8}\"; \
       echo; done"
expect_rc 0
count_out '"choices"' 2
# cleanup
run "infer-stack leases --json | python3 -c 'import json,sys;[print(l[\"id\"]) for l in json.load(sys.stdin)[\"leases\"] if l[\"state\"]==\"active\"]' | xargs -r -n1 infer-stack release >/dev/null 2>&1; true"
end_step
