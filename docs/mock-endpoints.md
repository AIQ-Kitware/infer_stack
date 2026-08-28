# Mock endpoints: leasing a simulator instead of a GPU

infer-stack can lease an endpoint that is served by a simulator rather than
a real engine. The point is to exercise the *whole* path — `acquire` →
placement → compose render → container → gateway route → readiness probe →
your command → `release` → teardown — on a laptop, with no GPU, no API key,
and no model weights.

There are two simulators, and they are not interchangeable.

| | `llm-d-inference-sim` | `infer_stack.mockserver` |
|---|---|---|
| catalog | `dev/e2e_tests/catalog-mock.yaml` | `dev/e2e_tests/catalog-mock-oracle.yaml` |
| answers | random text from a sentence bank | correct at a configured rate, against a shipped answer key |
| API surface | vLLM's, closely | the parts a card uses |
| streaming | SSE, with real TTFT / inter-token delay | no |
| token usage | real prompt/completion counts | zeroed |
| `/metrics` | vLLM-compatible subset | no |
| failure injection | `rate_limit`, `server_error`, `context_length`, … | failure rate only |
| slow startup | `startup_duration` (503 on `/health/ready`) | ready instantly |
| bearer auth | no | yes |
| answers the question | *does the client break?* | *does the math compute?* |

Neither says anything about a real model. A run against either **must never
be submitted as an evaluation result.**

## Which one to reach for

Use the **API-fidelity** mock (`catalog-mock.yaml`) by default. It is an
external, independently maintained simulator of vLLM's server, so a client
that works against it is much more likely to work against vLLM. It is the
one that finds unhandled 429s, streaming assumptions, token-budget
arithmetic, and code that assumes a lease is usable the instant the
container is `running`.

Use the **oracle** mock (`catalog-mock-oracle.yaml`) when the thing under
test is the *statistics* — a fan-in that silently drops a model, a metric
that collapses to NaN on constant input, an AUROC computed over a single
class. Random text cannot distinguish "the pipeline works" from "the
pipeline produces garbage quietly", because garbage is the expected input.

Running both is cheap and they fail for different reasons.

## Getting the images

```bash
docker pull ghcr.io/llm-d/llm-d-inference-sim:v0.9.0
docker build -f dockerfiles/mock-vllm.dockerfile -t aiq-mock-vllm:latest .
```

## Running

```bash
infer-stack run --catalog dev/e2e_tests/catalog-mock.yaml \
    --backend compose --endpoint mock-smol -- \
    python -c 'import os, openai; print(openai.OpenAI().models.list())'
```

The command sees `OPENAI_BASE_URL` and `OPENAI_API_KEY` exactly as it would
for a real lease, so nothing downstream needs to know it is talking to a
simulator.

## `runtime.simulator`

`llm-d-inference-sim` speaks vLLM's *API* but not vLLM's *CLI*: no
positional model argument, no `--host`, no `--tensor-parallel-size` or
`--gpu-memory-utilization`, and it exits on an unknown flag. Declaring
`runtime.simulator` in an endpoint switches three things:

* the rendered command comes from `profile_runtime.simulator_args` instead
  of `vllm_args`, driven by the same endpoint fields;
* `placement.required_gpu_count` drops to **0**, so the endpoint is
  placeable on a host with no GPUs at all;
* the container healthcheck is disabled — it shells out to `curl`, which
  the distroless simulator image does not contain, so it would report a
  perfectly healthy container as unhealthy forever.

Readiness is *not* relaxed: `probe_ready` still requires a real generation
over HTTP, which is a stronger signal than any healthcheck and works
against any image.

```yaml
endpoints:
  mock-smol:
    engine: vllm
    model: smol135
    runtime:
      image: ghcr.io/llm-d/llm-d-inference-sim:v0.9.0
      max_model_len: 2048
      simulator:
        kind: llm-d-sim
        mode: random          # or `echo` to get the prompt back
        seed: 20260731
        time_to_first_token: 120ms
        inter_token_latency: 8ms
        startup_duration: 10s
```

`kind` and `model` are the only keys interpreted here. Everything else in
the block becomes a simulator flag verbatim (`snake_case` → `--kebab-case`,
`true` → a bare flag, a list → repeated values), so any knob the simulator
grows is reachable from the catalog without a change to infer-stack.

`--model` carries neither the HF repo id nor the served name: it gets a
`sim-<slug>` that cannot resolve to a repo. The simulator decides how to
tokenize by looking that value up on HuggingFace — a real repo id means
real tokenization, which it delegates to a separate render service and
dies at startup without. Since a catalog endpoint is normally named after
a real model (`Qwen/Qwen3-8B`), passing either through verbatim would turn
the ordinary case into a crash loop. Set `simulator.model` if you do want
the render-service path. The served alias is unaffected either way.

## Known fidelity gaps

Close is not identical. As of `v0.9.0`:

* **`n > 1` is ignored** — both `/v1/completions` and
  `/v1/chat/completions` return exactly one choice however many you ask
  for. A client that fans K samples out into K requests is unaffected; one
  that relies on the server gets K = 1 and degrades quietly. The oracle
  mock does honour `n`.
* **No `logprobs`** — the field is accepted and dropped.
* **No bearer auth** — it cannot reject an unauthenticated request, so the
  auth path is only covered by the oracle mock.
* **Unknown request fields are accepted**, where a stricter server might
  400. It does return 400 when the prompt exceeds `--max-model-len`.

The oracle mock needs none of this — its entrypoint parses vLLM's command
line, so `runtime.image` deploys it.

It does need two more things in `runtime`, and they are both explicit:

* `cpu_only: true` — it serves from CPU, so it must be placeable on a host
  with no GPU. This says *only* that. It is deliberately not
  `runtime.simulator`, which additionally means "this process does not take
  vLLM's command line" and would switch the renderer this mock depends on.
  Nothing is inferred from `gpu_memory_utilization: 0.0` or from the image
  name.
* `--mock-config=` in `extra_args`, naming the answer-key fixture.

### Where the answer key comes from

The fixture is `infer_stack/mockserver/data/oracle_questions.yaml`, and it is
baked into the image by `COPY infer_stack ./infer_stack`, so the catalog names
an absolute in-image path and no bind-mount has to be arranged:

```yaml
extra_args:
  - '--mock-ability=0.6'
  - '--mock-config=/opt/infer-stack/infer_stack/mockserver/data/oracle_questions.yaml'
```

It holds the **question corpus only**: `questions`, the matching `answer_key`,
and the `composition` map that makes a compound question correct only when
every step is. Per-model behaviour stays on the endpoint — `--mock-ability`
and `--mock-mode` — so one corpus serves the whole cohort and a model's
competence is described where its deployment is.

**What "answer-aware" does and does not mean.** For a question in the fixture,
a model that gets it right returns the literal gold answer. For any other
question the verdict is still deterministic and still correlated across the
cohort — which is what a card's arithmetic is being exercised against — but
the "correct" text is a synthetic `answer-…` string, because nothing has told
the server what the right answer would be. Neither case says anything about a
real model.

A different corpus can be supplied by mounting one and pointing
`--mock-config` at it.

## Endpoints worth knowing about

`catalog-mock.yaml` ships four:

* `mock-smol`, `mock-qwen17` — the ordinary case, with realistic latency
  and a 10s startup window.
* `mock-echo` — the response *is* the prompt, so the recorded transcript
  becomes a copy of what your card actually put on the wire.
* `mock-flaky` — one request in five fails with a real provider error
  shape. This is the endpoint that finds out whether retry handling works
  before a twelve-hour run finds out for you.
