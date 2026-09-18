# Host verification for the admission plan (P1-P10)

Plan: `plan-keep-warm-admission-2026-09-16.md` §7. The code is on `dev/0.7.1`;
this runbook closes what can only be measured on the GPU host. Run everything
from the repo checkout the host uses. Record the output next to this file.

## 0. Before upgrading a live host

The first lease operation after the upgrade does a one-time migration:

- **Profile.** It freezes the current settings and catalog.
- **Allocations.** Running model containers adopt their GPUs as allocations.
- **Ownership.** Pre-label containers are adopted as managed.

Do this while the stack is idle if you can:

```bash
infer-stack leases                      # note anything live
infer-stack config publish <every catalog runbooks on this host use> --yes
infer-stack leases                      # health: block should be empty or explained
```

A GPU reservation (`acquire --reserve-gpus`) made before the upgrade stays
*unresolved* and blocks new allocations until it is released.

## 1. Docker-only checks (safe with a live stack)

```bash
GPU=<a free GPU index> bash dev/tmp/host_verification_docker.sh 2>&1 | tee host-verification-docker.log
# optional V8: TORCH_IMAGE=pytorch/pytorch:latest GPU=<idx> bash ...
```

This covers V1, V2, V3, V5, V6, V10, V13 and, optionally, V8. Each check uses a
throwaway Compose project (`isv-*`).

Already observed on the guest daemon (no GPU): V2, V5, V13 pass. For V10, a
**stopped container does not keep its static address**. That is why the address
table is append-only in the ledger and an unmanaged holder of an address blocks
the apply; it needs no further change.

## 2. Checks that need infer-stack (an idle stack; no active leases)

### V15: lock hold per operation (decides D22 and D25)

Measure the whole command, which is dominated by the lock hold:

```bash
T() { /usr/bin/time -f "%e s  $*" "$@" ; }
T infer-stack apply --yes                                   # no-op apply
T infer-stack acquire <small-endpoint> --no-wait --yes      # add one model (static gateway)
T infer-stack release --all --evict --yes                   # remove it
# dynamic routing: same add and remove after `infer-stack config publish` with dynamic_routing on
# fresh dynamic bootstrap: `infer-stack stack down`, then the first acquire, measured separately
# concurrent per-shard callers: N parallel `infer-stack run --queue` with a trivial command
```

Pass criterion: steady-state holds stay well inside the bounds in
`compose.py` (`DOCKER_TIMEOUT_*`, `ROUTE_RECONCILE_STEADY_S = 20`,
`APPLY_HEALTH_WAIT_S = 180`). If per-shard churn makes lock waits dominate,
report the numbers. The known-limitations fallback is to skip an apply whose
render is identical to the last success.

### Interrupted apply (P2 settle and P8 per-state rules)

```bash
infer-stack acquire <endpoint> --no-wait --yes &   # while its first `up` runs:
pid=$!
sleep 3; kill -INT "$pid"                          # only THIS acquire, not anyone else's
infer-stack leases                                  # expect: pending change, interrupted
infer-stack apply --yes                             # waits for a settled runtime, then applies
docker ps -a --filter label=com.docker.compose.project=infer-stack \
  --format '{{.Names}} {{.State}} {{.Label "infer-stack.fingerprint"}}'
```

Expect exactly one container per wanted service, and no duplicate model
containers. Repeat with an endpoint whose image is not yet pulled, interrupting
during the pull.

### V9 and V11: stable addresses and routing check

```bash
infer-stack network migrate --subnet 172.30.0.0/24 --yes   # choose a free /24
infer-stack network check                                   # every upstream: healthy
```

- **V11:** time the gateway's downtime during `migrate` (every container is
  recreated once), for example by polling `/v1/models`.
- **Subnet change:** run `network migrate --subnet <another free /24> --yes`
  again, then check that `docker network inspect infer-stack-net` shows the new
  subnet and that `network check` is healthy.
- **V9:** `network check` itself exercises `docker exec <gateway> python3`.

### Test 60: the misroute reproduction, re-run with stable addresses

Re-run the reproduction from
`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`, after
`network migrate`. Pass: zero misrouted samples.
