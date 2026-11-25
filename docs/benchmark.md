# Benchmarking managed inference endpoints

On a Compose host with the normal LiteLLM gateway and catalog:

```bash
infer-stack benchmark run qwen3.8-27b inkling-2-mini-maya-long
# Or select endpoint patterns (quote to prevent shell expansion):
infer-stack benchmark run 'qwen3.8-*' 'inkling-2-mini-maya*'
```

Use the endpoint names from your catalog. No GPU selection or special Docker
commands are necessary. The runner acquires each endpoint using normal placement,
waits for readiness, measures it, then releases its own lease and evicts only the
newly idled deployment (even if normally keep-warm). Existing active leases are
not released; an already-shared deployment is skipped to avoid interfering with
another workload. This is not an exclusive GPU reservation: other workloads may
arrive later or run on the other GPU, which can affect shared hardware resources.
If release fails the sweep stops rather than proceeding to the next endpoint.
Ctrl-C and SIGTERM unwind through the lease cleanup; a bounded TTL is the hard
kill/host crash backstop.

The default approximate prompt targets are 1K, 8K, 32K, 64K, 128K and 160 Ki
(163,840 tokens), with up to 512 output tokens and one repeat. Each trial makes
three requests: a unique synthetic code review prompt, the identical prompt,
then the same prompt extended with additional code. The latter two measure
attempted prefix reuse, not guaranteed cache hits. Two small calibration requests
estimate tokenization costs. API usage gives the actual counts; if unavailable,
no fake token count or token throughput is emitted. Targets without configured
context headroom are recorded as skipped. Fresh means a fresh user prefix, not a
cold process or a globally flushed cache. Startup time includes image/model
preparation, lease realization, and the existing readiness generation probe.

```bash
# Short smoke benchmark
infer-stack benchmark run inkling-2-mini-maya-long --contexts 1024,8192 --generation-tokens 128
# Focus on long-context performance, optionally select one physical GPU by index
infer-stack benchmark run 'qwen3.8-*' --contexts 131072,163840 --repeats 3 --allowed-gpus 1
# Explicit effort; otherwise the endpoint's configured default is used
infer-stack benchmark run inkling-2-mini-maya-long --effort high
```

Generation uses streaming Chat Completions (or Completions for endpoints that
declare it), through LiteLLM, with temperature 0 and `stream_options.include_usage`.
Observed first/last output times include reasoning chunks when exposed by the
server. Streaming token rate is an estimate, not native-engine decode counters;
prompt tokens / TTFT is an effective ingestion metric, not pure prefill. Different
model tokenizers, stopping behavior and speculative acceptance rates affect the
comparison. This synthetic workload tests serving performance, not coding quality.
Unsupported streaming/usage and request failures are observable in the ledger.

## Portable records and reports

The default append-only JSONL file is
`<infer-stack data root>/benchmark/observations.jsonl`. Use `--results PATH` to
select another file. Each record has a unique ID, run ID, schema and timestamp.
Records are appended under an advisory file lock and fsynced; files should be
local, then copied to an aggregation host rather than written by several machines
to a network filesystem. The ledger contains no prompts, completions, API keys,
container env, or raw upstream error bodies. It does contain hostnames, a hash of
the host machine ID, GPU UUIDs, and hardware/model identifiers: treat it as
infrastructure metadata when sharing. A partially written last line from a crash
is rejected explicitly; preserve a copy and repair before appending.

Each run prints the report directory at startup and writes `report.html`,
`report.md`, `summary.csv` and `configurations.json` there before the first
acquisition and after each durable ledger observation. Open the HTML in a
browser while the sweep is running; it refreshes every ten seconds until the
final report is published. The terminal prints the reported prompt and output
token counts, TTFT, estimated streamed and end-to-end generation rates, cached
tokens (if available), and elapsed time after each request. Failed and skipped
trials are also printed. No metrics are fabricated for missing API usage.

Reports are replaced atomically per file. They are derived snapshots, not a
second source of truth. A failed live report update warns without aborting the
benchmark. The final snapshot is rebuilt from the ledger even on graceful
interruption. A hard kill can leave a stale report marked Running; use the
ledger to recover all durable observations. Use `--final-report-only` to omit
intermediate report writes for large sweeps; terminal results and the ledger
remain incremental.

You can also generate a snapshot from another terminal during a run, including
runs launched with an older infer-stack version. Use the run ID printed at
startup and a separate output directory to avoid racing the live publisher:

```bash
infer-stack benchmark report --run-id RUN_ID --output /tmp/infer-stack-live
xdg-open /tmp/infer-stack-live/report.html
```

Repeat the report command and reload the page to see newer results. To combine
historical ledgers from multiple machines, copy them to a local directory and run:

```bash
infer-stack benchmark report
infer-stack benchmark report results/*.jsonl --output ./benchmark-report
infer-stack benchmark report --run-id RUN_ID --output ./single-run
```

The report is generated offline, without Docker, GPU access or an API key.
Identical copies of a record are deduplicated; conflicting IDs are errors. The
report groups by host, GPU UUID and PCI bus address, upstream PCI topology,
firmware/driver and power limit, configured serving provenance, resolved Docker
image ID (when observable), infer-stack version, and workload settings. The
configuration identity deliberately does not include transient PCIe downshifts.
The raw telemetry includes sampled current GPU and upstream bridge link speeds
and widths, GPU utilization, clocks, temperature, power and memory use. Compare
negotiated links during high utilization, not merely idle measurements. Host
inventory and PCI topology are best-effort (Linux sysfs, nvidia-smi); a missing
field is not evidence that two devices are equivalent. Use GPU UUIDs rather than
CUDA indices to compare individual cards across reboots. The configured
provenance digest does not attest the loaded weights of mutable custom launchers.

This first version targets local Compose with LiteLLM, not remote control,
KubeAI, external endpoints, concurrent load, or full energy accounting. Reports
are descriptive; no performance or correctness claim is made for a model based
solely on an endpoint starting successfully.
