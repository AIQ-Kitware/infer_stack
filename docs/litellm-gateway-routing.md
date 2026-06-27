# LiteLLM gateway routing: today (static superset) and where we're headed (admin API)

This explains how the leasing stack exposes models through the LiteLLM gateway,
why the current design avoids restarting the gateway, and the direction we
intend to take it.

## How it works today: static superset route table

When a catalog is available, the gateway is rendered with a **static superset
route table**: one LiteLLM `model_list` entry per *catalog* endpoint, each
pointed at the endpoint's deterministic upstream host
(`vllm_service_name_for` / `ollama_service_name_for`). See
`infer_stack/leasing/compose.py::_litellm_model_list_from_catalog` and the
`catalog is not None` branch of `render_compose`.

The key property: **the gateway config depends only on the catalog, not on which
deployments are currently placed.** Acquiring or releasing a model does not
change the rendered `litellm_config.yaml`, so its `config_hash` label is stable,
so `docker compose up -d` leaves the gateway container untouched — **no blip.**
A route whose upstream vLLM/Ollama is not up yet simply cools down until the
container appears; `router_settings` make that warmup self-healing.

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
  cannot be routed without changing the config file — which means a recreation
  (a blip). The legacy fallback (`_litellm_model_list` from live deployments)
  routes only placed deployments and *does* rewrite-and-recreate per change.

## Direction we want to head: LiteLLM admin API for dynamic routes

> **For a future agent working on gateway routing:** the intended evolution is
> to add/remove routes at runtime via LiteLLM's admin API instead of (only)
> rendering a static file. Static superset is fine for now and should remain the
> default; treat the admin API as an additive capability, not a replacement.

LiteLLM proxy supports runtime model management with no container restart:

- **Admin API:** `POST /model/new` and `POST /model/delete`, authenticated with
  the master key (already set via `API_KEY_ENV`). Add or remove a single route
  live.
- **DB-backed model store:** `STORE_MODEL_IN_DB=true` + `DATABASE_URL`
  (Postgres). Models live in the DB and LiteLLM hot-loads them; this is what the
  admin UI uses.

### Why this is wanted

It removes the one thing static superset can't do: route **non-catalog /
interactive** acquires with zero blip. It also avoids first-bring-up and
catalog-change recreations.

### What to watch out for when implementing it

- **It is imperative, stateful mutation.** The gateway's in-memory route set can
  drift from the ledger's desired state, so it needs a reconcile (list current
  routes → diff against desired → add/remove), not fire-and-forget calls.
- **Partial failure:** `/model/new` succeeding while the deployment dies (or
  vice versa) must be handled so the gateway and ledger don't disagree.
- **Survivability:** plain `/model/new` state is lost on a gateway restart
  unless `STORE_MODEL_IN_DB` is used — which adds a Postgres dependency.
- **Render/apply fit:** the static config file *is* the rendered desired state,
  which matches the project's render/apply separation. Keep that model — render
  the desired route set, then apply it via the API as a diff — rather than
  scattering imperative calls through acquire/release.

### Recommended shape

Keep static superset as the default. Add admin-API routing as an **opt-in**
(e.g. a flag / setting) that, when enabled, reconciles the live route set on
each converge. That preserves today's behavior while unlocking non-catalog
dynamic routing for callers that need it.
