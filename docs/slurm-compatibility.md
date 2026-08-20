# Running under Slurm

infer-stack and Slurm do not overlap. Slurm decides which GPUs a job owns;
infer-stack decides which models run on the GPUs it has been given.

```
Slurm            allocates GPUs to jobs, queues jobs, enforces limits
  |              $SLURM_JOB_GPUS = the indices this job owns
infer-stack      places model servers within those indices, leases them to callers
```

Multi-node placement, preemption and migration are **out of scope** for
infer-stack — that is Slurm's job. Single-host placement within an allocation
is infer-stack's.

## The one thing you must do

`$SLURM_JOB_GPUS` is **not read automatically**. Pass it:

```bash
infer-stack acquire "$ENDPOINT" --allowed_gpus "$SLURM_JOB_GPUS" -- ...
infer-stack run --endpoint "$ENDPOINT" --allowed_gpus "$SLURM_JOB_GPUS" -- <cmd>
```

Without it, placement considers every GPU on the machine, including cards
Slurm gave to somebody else. Nothing stops you — infer-stack cannot see the
allocation — and the failure lands later as a CUDA OOM in whichever job loses.

## The stack is shared; the allow-list is per call

All jobs on a host share one ledger and one set of deployments. That is
deliberate: two jobs asking for the same endpoint should share the model
already loaded rather than each starting their own copy.

So `allowed_gpus` gates only where **new** deployments may land. A deployment
already placed keeps its GPU even if that GPU is outside this call's allow-list,
because it belongs to another job's allocation and is running that job's model.
Validating existing placements against the current call's allow-list would
defer and reshuffle other jobs' models — see the comment at
`plan_placement()` in `infer_stack/leasing/placement.py`.

The practical consequence: `infer-stack ps` and `infer-stack leases` show the
whole host, not your slice. Your job's own GPUs are the ones in
`$SLURM_JOB_GPUS`.

## Concurrency

The ledger is SQLite with `BEGIN IMMEDIATE` writes under a lock, so concurrent
`acquire`/`release` from several Slurm jobs is safe. The lock file must be
writable by every uid that runs the CLI; a second user hitting `EACCES` on it is
the usual symptom of a lock file created by the first.

## What this does not solve

**Nothing reconciles Slurm's accounting with infer-stack's.** If you request
`--gres=gpu:2` and then ask infer-stack for a model needing four cards, Slurm's
allocation is not consulted at placement time — you get whatever
`--allowed_gpus` says, or the whole machine if you omitted it.

**A job that exceeds its own allocation is not rejected at submission.**
Slurm accepts the job; infer-stack only sees the mismatch at `acquire`. It does
fail fast there — a request that cannot fit the allow-list even on an idle host
is rejected immediately rather than queued, including one whose deployments are
each placeable but cannot fit together — but that is minutes into the job
rather than at `sbatch`.

Declare resources to Slurm and pass the same slice to infer-stack, and the two
agree. Skip either and they will not.
