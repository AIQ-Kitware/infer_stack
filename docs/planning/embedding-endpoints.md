# Embedding endpoints: exploratory scope

Status: exploratory, 2026-11-23. This is an option to investigate, not an
implemented capability or a commitment to a new serving engine.

## Question

Should the existing catalog, placement, lease, gateway and cleanup machinery
manage local embedding servers as well as generation servers? Repeated clients
could share loaded weights and a stable endpoint instead of launching an
encoder separately for every experiment. In-process batch encoding remains a
useful alternative, especially for one-off jobs and CPU-sized models.

## Potential boundary

Infer-stack would own server configuration, model/runtime provenance, placement,
readiness, shared access and resource cleanup. Clients would own text selection,
query/passage instructions, vector caches, reference datasets and downstream
analysis. Pooling, normalization, dimensions and truncation need explicit
ownership and configuration because they affect the resulting vectors.

An external embedding API needs no local GPU lease. Existing external-endpoint
routing may be useful, but a direct client should remain possible. Do not make
this exploratory capability a prerequisite for using a hosted provider.

## What appears reusable, and what is missing

- Reuse existing runtime ownership, command-scoped leases, image/weight caches,
  placement, gateway authentication and provenance. Avoid another controller.
- Current catalog protocol validation and readiness probes support chat and
  text completions. Embedding-only models need an explicit capability and a
  small embedding request through the selected access path.
- vLLM already offers an OpenAI-compatible `/v1/embeddings` API for supported
  pooling models. Investigate this before adding another serving engine.
- A specialized engine such as Text Embeddings Inference could be worthwhile
  for compatibility or measured performance. It would add image, discovery,
  backend, diagnostics and maintenance work; do not assume it is required.

## Tradeoffs to measure

- Shared warm weights and cross-client batching versus in-process startup,
  throughput, serialization overhead and memory use for one large batch.
- Encoder GPU residency versus generation capacity. Existing protected leases
  must not be evicted merely because a different workload asks for embeddings.
- Explicit embedding semantics versus an expanding collection of model-specific
  settings. An HTTP vector response alone does not establish equivalence to a
  previous in-process encoding configuration.
- Compose/KubeAI lifecycle consistency versus implementing a Compose-only
  capability that cannot follow the common backend contract.

## Investigation before deciding an implementation

1. Try one small encoder and one larger supported encoder with existing serving
   engines, outside production catalog changes. Compare against in-process batch
   encoding on representative input lengths and concurrency.
2. Identify the smallest endpoint-capability, readiness and gateway changes.
   Record dimensions and relevant declared encoder settings as provenance.
3. Exercise shared access, final release and cancellation alongside a generation
   deployment; measure contention and report unavailable capacity honestly.
4. Decide whether repeated use justifies managed serving, whether another
   engine is needed, and what acceptance evidence would support both backends.

No endpoint registrations, new engine, embedding client, discovery behavior or
placement policy are added by this document.

References:

- [vLLM embedding API](https://docs.vllm.ai/en/stable/models/pooling_models/embed/)
- [vLLM pooling configuration](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/index)
- [External endpoint design](external-endpoints.md)
- [GPU ownership and serving capacity](gpu-ownership-and-serving-capacity.md)
