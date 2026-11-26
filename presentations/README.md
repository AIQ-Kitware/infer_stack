# Presentations

Version-controlled presentation sources. Generated slides, PDF exports and
installed dependencies are not tracked.

[Infer-stack presentation suite](infer-stack/README.md) — choose the deck for
the audience:

| Talk | Goal |
|---|---|
| Why infer-stack? | Adoption, use cases, value proposition |
| Using infer-stack | Practical quickstart and client integration |
| Measuring inference | Benchmark method, evidence and tradeoffs |
| Infer-stack live | Demonstrate interactive discovery, shared leases, scoped jobs, GPU placement, simulators, live reports |
| Technical overview | Detailed architecture, catalog, backends and original case study |

The five decks share one local Node project and the dated benchmark dataset.
See the [build instructions](infer-stack/README.md#build) and
[presenter runbook](infer-stack/DEMO.md). For another presentation series, use a
separate subdirectory when it has its own build needs. Keep sources and
non-sensitive data in Git; put generated artifacts in ignored `build/` folders.
