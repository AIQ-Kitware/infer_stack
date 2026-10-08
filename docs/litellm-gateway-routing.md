# LiteLLM gateway routing: today (static superset) and where we're headed (admin API)

This explains how the leasing stack exposes models through the LiteLLM gateway,
why the current design avoids restarting the gateway, and the direction we
intend to take it.

## How it works today: a static superset route table, derived at render

The gateway is rendered with a **static superset route table**: one LiteLLM
`model_list` entry per endpoint alias, each pointed at the endpoint's
deterministic upstream host (`vllm_service_name_for` /
`ollama_service_name_for`), or, for an external endpoint, at its own server.

The table is **derived at every render**, never stored, from these layers
(a later one wins for the same alias; `gateway.front_door_routes`):

1. the **route registry** (`<state_dir>/litellm_registry.json`): routes of
   deployments no published catalog defines, remembered past their release,
   plus rows an older infer-stack wrote for catalog endpoints;
2. the **published catalog union**: every endpoint every runbook on this
   stack has published (kept in the recovery profile, merged on every
   acquire; see docs/planning/external-endpoints.md, decision 1);
3. every **placed deployment in the full `desired` set**, which spans all
   runbooks via the shared ledger;
4. routes another backend supplies (a KubeAI cluster's Models).

Each is a `GatewayRoute` (`infer_stack/leasing/routes.py`), rendered by one
function, `GatewayRoute.entry()`. The registry stores *semantic* inputs
(served name / engine / host), never rendered LiteLLM entries.

Two key properties:

- **The gateway config depends only on shared state, not on which runbook
  invoked the converge or which deployments are currently placed.** Acquiring
  or releasing a model does not change the rendered `litellm_config.yaml`, so
  its `config_hash` label is stable and `docker compose up -d` leaves the
  gateway container untouched: **no blip.** A route whose upstream is not up
  yet cools down until the container appears; `router_settings` make that
  warmup self-healing. An ad-hoc deployment (one no catalog defines) is
  remembered in the registry after an approved render, so releasing it does
  not change the config either.
- **A cross-catalog converge cannot strip another runbook's routes.** The
  published union holds every runbook's catalogs, so converging runbook B's
  disjoint catalog does not drop runbook A's routes. This is the fix for the
  "olmo healthy, gateway 400 `Invalid model name`" incident: another runbook's
  converge used to re-render the shared gateway from *its* catalog alone and
  evict olmo's routes. Routes **persist past release**: a released endpoint
  stays routable until explicitly pruned.

The only gateway recreations are the first appearance of an endpoint (or an
ad-hoc deployment) and a genuinely changed endpoint definition.

### Managing routes: `infer-stack routes`

- `infer-stack routes list`: the routes the gateway is to serve (alias,
  origin, upstream model, upstream, and whether a live deployment backs it).
  It is desired state, derived from the published endpoints and placed
  deployments, and says when a publication is pending (the gateway may not
  have them yet; `infer-stack apply` finishes it). Origins are `catalog`,
  `external`, `deployment`, `upstream` (KubeAI) and `registry`.
- `infer-stack routes seed <catalog.yaml> [...]`: publish sibling runbooks'
  catalogs (merge them into the published union) and publish once. The
  operational key to blip-free concurrency: **seed all overlapping runbooks'
  catalogs once while idle**, and no converge from any of them recreates the
  gateway afterwards. A seed that would redefine a published alias refuses
  unless `--replace`, and never redefines one a resident deployment runs.
- `infer-stack routes prune`: the explicit "unpublish" verb. It drops every
  published endpoint and registry row the invoking catalog does not define and
  no deployment serves, then publishes (one accepted recreate). It prints the
  exact drop list and asks first (`--yes` / non-terminal skips), because a
  prune from the wrong `INFER_STACK_CONFIG_DIR` would drop every other
  runbook's endpoints. Nothing prunes automatically: any catalog-keyed rule
  would reintroduce the cross-catalog alternation churn.

Both commit under the host-wide publication lock, like any desired-state
change.

### Upgrading

The published union and the registry are read as they are. A registry an
older infer-stack wrote holds catalog rows too; they stay routed (lowest
precedence, so a published definition wins) until `routes prune` drops the
ones nothing uses, and the file is not rewritten for them. A corrupt registry
is ignored; an *unknown* schema version with a parseable `entries` map is
rendered as-is and never rewritten, so a binary rollback loses nothing. A
state dir from before the registry existed (a `litellm_config.yaml` and no
registry) is no longer seeded from that config: its runbooks' catalogs
publish their endpoints again at their next acquire, or at once with
`routes seed`.

LiteLLM reads `--config /etc/litellm/config.yaml` **once at startup** and does
not watch the file, so this static-config approach is what lets us avoid
recreating the container as models come and go. The only times the gateway is
recreated are first bring-up and an actual **catalog** change.

### Consequences / limitations

- `/v1/models` advertises the whole catalog, including endpoints whose upstream
  is currently down (they error until placed). This is intentional.
- **Warmup log noise:** because readiness is polled with a real generation
  through the gateway (see `ComposeBackend.probe_ready`), every poll before the
  upstream is serving logs a LiteLLM `InternalServerError`/`Connection error`
  stack. This is expected and self-healing; we want to submit an upstream patch
  for a quiet, generation-level readiness probe — see
  `dev/leasing-followups.md` ("Upstream LiteLLM: a quiet, generation-level
  readiness probe"). Do not globally silence LiteLLM logs to hide it.
- An endpoint that is **not in the catalog** (ad-hoc / interactive `acquire`)
  is routed too: every placed deployment is a route layer, and the registry
  remembers it past release, so it stays routable without a recreate.
- **Same-model `--dedicated` collapses.** Because the upstream host is derived
  from the served name alone (`vllm-<served>`), N dedicated deployments of one
  model render to ONE compose service — one container, one GPU — defeating
  multi-GPU dedicated use. The opt-in **dynamic routing** mode below fixes it.

## Dynamic routing: LiteLLM admin API + Postgres (implemented, opt-in)

Dynamic routing is now **implemented** as an opt-in mode (off by default; static
superset remains the default). Enable it with `--dynamic-routing` or
`config set dynamic_routing true`. It is the *proper* fix for the dedicated
collision and the way to route non-catalog/interactive acquires with zero blip.

### Important correction (verified against the pinned LiteLLM source)

The admin API is **not** DB-less. In `litellm v1.82.3`,
`add_new_model` (`/model/new`) returns HTTP 500 unless **both** a database is
connected (`prisma_client is not None`) **and** `STORE_MODEL_IN_DB=true`; the
same gate guards `/model/delete` and `/model/update`. There is no in-memory
runtime model-add. So "admin-API dynamic routing" **requires Postgres** — this
is the revival of the `postgres-litellm` store (the prior follow-up #2), now done
deliberately. (An earlier draft of this doc implied plain `/model/new` worked
in-memory but non-durably; that was wrong.)

### How it works (render desired routes → apply as a diff)

It follows the project's render/apply separation exactly:

- **Render** (`converge(apply=False)` → `render_compose(dynamic_routing=True)`):
  - Each vLLM deployment gets its **own** unique service `vllm-<served>-<id>`
    (`vllm_service_name(unique=True)`), so same-model `--dedicated` deployments
    become distinct containers on distinct GPUs (the bug the static gateway
    couldn't fix).
  - A `postgres-litellm` service is rendered, and the `litellm` service gets
    `DATABASE_URL` + `STORE_MODEL_IN_DB=true` env and `depends_on` the DB being
    healthy. The rendered `litellm_config.yaml` is a **static base** (empty
    `model_list`), so its `config_hash` never changes as models come/go — the
    gateway is never recreated (**no blip**).
  - The desired route set (one entry per live `(deployment, endpoint)`, each
    with a deterministic `model_info.id`) is written to `litellm_routes.json` —
    the rendered desired state for the gateway's route table.
- **Apply** (`apply()` → `_reconcile_routes()`): after `docker compose up`, list
  the gateway's current models (`GET /v1/model/info`), diff against
  `litellm_routes.json` by `model_info.id`, then `POST /model/new` for the
  missing and `POST /model/delete` for the extra — no container restart.

Several dedicated deployments of one model share one public `model_name` but
distinct upstreams, so LiteLLM **load-balances** the alias across them — each on
its own GPU, while clients still ask for the one name. Clients keep the uniform
gateway `base_url`; nothing about the env-file descriptor changes.

### Why this is robust (the implementation honors these)

- **Deterministic ids ⇒ pure set-diff reconcile.** `_route_id(owner, alias)`
  (the owner is the deployment id, or `external` for an external endpoint, so
  redefining an external endpoint's server replaces one route) is stable, so the same route is added once and never churned, and a dropped
  route is deleted by exactly its id. Reconcile is idempotent (a redundant apply
  is a no-op), which is what makes the controller's *coalesced* apply correct for
  routes too.
- **Drift-healing.** Routes lost to a gateway/DB restart reappear in the diff and
  are re-added; stale routes from a prior run (still in the DB) are deleted
  because they are no longer desired. (DB persistence is the survivability that
  plain in-memory adds would lack — another reason Postgres is required, not
  incidental.)
- **Co-existence.** Only routes infer-stack created (id prefix `isr-`) are ever
  deleted, so a model added by hand through the UI/admin API is left alone.
- **Best-effort apply.** The gateway may still be starting (it waits on Postgres
  health), so the initial `/v1/model/info` listing is retried; a persistent
  failure is logged and left for the next converge rather than raised — apply
  stays non-fatal, like `docker compose up`.

### Consistency requirement

Every verb (`acquire`/`release`/`gc`/`evict`/`apply`) must agree on the mode, or
they render inconsistent service names. The **persisted setting**
(`config set dynamic_routing true`) is therefore the primary switch; the
`--dynamic-routing` flag is a per-invocation override.
