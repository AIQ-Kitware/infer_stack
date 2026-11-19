# Request observations and retrospective lookup

Status: proposed, November 18, 2026. No automatic request collector or
request-observation lookup API is implemented in infer-stack today.

An evaluation client measures the time its caller waited. Local serving can
also explain where that time went and what work was performed. Preserve both
observations, joined by explicit IDs, without repeating inference to obtain
missing telemetry.

## Existing pieces

- The front door is LiteLLM. Its response headers can identify a gateway call
  and selected model and report gateway duration, overhead and retry counts.
  See [response headers](https://docs.litellm.ai/docs/proxy/response_headers).
- vLLM offers aggregate Prometheus metrics and OpenTelemetry export. Detailed
  timing collection is optional and can affect performance; see the
  [v0.25.1 observability configuration](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/config/observability.py).
- Newer vLLM documentation describes response-body per-request queue,
  first-token and decode timings behind `--enable-per-request-metrics`.
  Check the actual image's capabilities and gateway preservation before
  enabling it; the latest documentation is not a compatibility guarantee for
  older images. See [per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/).
- `UsageStore`, `infer-stack usage` and controller `GET /v1/usage` store/read
  imported deployment/replica intervals, power samples and explicit energy
  estimates. They do not automatically collect live telemetry or retain
  per-request records. Request records must not inherit this store's rule
  prohibiting overlapping device intervals: requests commonly overlap.

## First implementation

Use existing gateway/engine observability seams. Keep inference payloads,
model sampling configuration and pipeline computation identity independent
of observation settings.

1. Correlate client logical calls, SDK invocations, gateway requests and each
   upstream attempt. Record completion IDs separately from HTTP request IDs;
   they are not interchangeable. Preserve failed, cancelled, retried and
   cache-served requests. A client timeout does not prove the engine stopped.
   A single gateway call or SDK invocation may contain several attempts.
2. Persist gateway request/attempt observations through callbacks or tracing,
   with an explicit retention policy. Store server timestamps, durations,
   routing/replica identity, runtime/image version, status, returned token
   usage and any observed cache facts. Exclude authentication and prompt/
   response bodies by default. Keep raw advertised facts separate from
   derived quantities and identify their measurement source/version.
3. Export engine request spans where available. Verify trace/header propagation
   across the gateway, including retries, instead of matching by model and
   approximate timestamps. Queue, prefill, decode and request lifetime remain
   distinct fields with documented clock boundaries and units.
4. Provide bounded, paginated, read-only lookup by explicit request/trace IDs
   and time window, plus an immutable export. Scope access to authorized
   records. Expose retention gaps and collector health; no record means
   unknown, not zero work. Survive gateway/controller/engine restarts.
5. Let evaluation readers enrich saved client tables from an explicit exported
   evidence set. Record the join keys, source and unmatched rows. Enrichment
   is a readout operation and schedules no generation or cache invalidation.
   A later server implementation can supply the same evidence contract.

Streaming, Responses API, direct-engine access and non-vLLM engines need
explicit capability coverage. An initial supported subset can be non-streaming
chat completions through the managed gateway; unsupported paths retain client
observations and explicit missing server evidence.

## Resource measurements

Request residence time is not exclusive accelerator execution time. Continuous
batching, prefix-cache reuse, tensor parallelism and shared replicas prevent
duration multiplied by GPU count from becoming measured request compute.

Collect GPU power/utilization and engine counters at the deployment/device
level, with stable device and replica identities. Publish sampled energy,
sample coverage, allocation/residency time and background work separately.
Any distribution of batch/device energy or execution time across requests is
an attribution model with a named method and assumptions. Do not label it
measured request energy or exact FLOPs. The existing
[resource-accounting roadmap](kubernetes-service-roadmap.md#phase-5--resource-accounting-and-energy-telemetry-v1)
and [northstar](northstar.md) define the broader accounting direction.

## Acceptance evidence

- A client invocation joins to its gateway and engine observations through
  success, upstream retry, failure, timeout and concurrent traffic.
- Saved records remain queryable after release and collector restart; exports
  remain usable after retention expiry. Missing evidence stays explicit.
- Original response content, generation parameters and computation IDs stay
  unchanged. Observation failures do not cause successful inference to retry.
- Test token/latency fields against a deterministic local HTTP fixture, then
  prove propagation against the pinned LiteLLM and engine images. Compare
  enabled/disabled overhead on a bounded live workload before a large run.
- Shared-batch tests do not double-count deployment/device resource totals as
  independent measured costs for every request.

Client elapsed-time recording and capture of server-advertised response facts
can ship independently. Durable retrospective lookup and physical resource
attribution are separate milestones, not prerequisites for client measurement.
