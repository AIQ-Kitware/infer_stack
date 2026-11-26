## 2026-11-25 15:40:56 -0500

Model: ChatGPT (GPT-6.2; serving configuration not independently identifiable).

The operator is running a multi-endpoint benchmark and wants to inspect data
before the sweep ends. The ledger already appends and fsyncs each observation,
but reports were only generated during the CLI's final cleanup and stdout
showed requested context rather than measured rates. Keep the ledger as the
single source of truth and publish a report immediately after each observation,
with a running/finished/interrupted state and browser refresh while running.
Print the observed request metrics separately from the requested prompt size.

Live publication uses in-memory observations from the current run rather than
rescanning the entire historical ledger after every trial. The final snapshot
reads the ledger to include any durable records missed during cancellation.
Each report artifact is atomically replaced; publication errors are advisory
while benchmarking and cannot invalidate a completed observation or skip lease
cleanup. The full report is regenerated synchronously between observations, not
within measured request time. Large sweeps may prefer --final-report-only.
A SIGKILL can leave the derived HTML marked Running, so recover from the ledger.
No changes to request generation, hardware grouping, ledger schema, or leases.

Validation: 9 tests passed and 1 skipped with --noconftest (ubelt unavailable);
all 10 passed with the existing import-only ubelt placeholder. CLI dispatch and
real GPU operation remain untested here because kwconf and GPU/Docker are not
available. This separate session journal avoids replacing the shared journal
from the preceding registry overlay.
