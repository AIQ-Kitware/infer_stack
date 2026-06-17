# Leasing model — tracked follow-ups

Durable list of known design follow-ups for the `infer_stack.leasing` path.
Not bugs (those are fixed on the branch / in CHANGELOG) — these are deliberate
"do this later" decisions with enough context to pick up cold.

---

## LiteLLM reload on model switch — keep the gateway alive

**Status:** interim fix shipped; **#1 chosen as the real fix (not yet done)**;
**#2 to retry in the future.**

### Background

The legacy (pre-leasing) stack deliberately kept LiteLLM **up across model
switches** so in-flight requests to *other* models weren't interrupted. Evidence
— the comment still in `infer_stack/templates/docker-compose.yml.j2`:

> "avoiding provider dependency edges prevents Compose from restarting LiteLLM on
> every vLLM model swap. Smoke tests and clients should retry until the selected
> upstream model is healthy."

It worked because legacy LiteLLM had a **static, comprehensive route table**
(model → `vllm-<model>:8000`) plus a `postgres-litellm` (`DATABASE_URL`). A
"switch" only started/stopped vLLM containers; LiteLLM's config never changed, so
it never restarted.

### What the leasing model does instead (and the interim fix)

The leasing Compose backend renders LiteLLM's `model_list` **dynamically, to
exactly the live deployment groups**. So adding/removing a served alias changes
the config file. LiteLLM reads that bind-mounted file only at startup, and
`docker compose up -d` won't recreate the litellm service if only the mounted
file changed — so a new route silently never appeared (coalescing readiness
timed out; see CHANGELOG + the `50_coalescing` e2e tier).

**Interim fix (shipped, commit ~6efad9b):** stamp an `infer-stack.config-hash`
label (sha256 of the rendered config) on the litellm service, so converge
recreates it **iff** the routing changes. Correct, and idempotent (no restart on
repeat-acquires of an existing alias, or on GPU reshuffles that keep the same
groups). **Cost / regression vs. legacy:** a brief (~1–3s) gateway-wide blip on
each *genuine* model add/remove. The vLLM model containers are NOT touched
(compose recreates only the `litellm` service), so GPU-resident models stay
loaded and serving — only the shared proxy hiccups momentarily.

### #1 — static superset route table (the chosen direction)

Render LiteLLM with **all catalog endpoints** routed to **stable upstream
addresses**, independent of the live set — like legacy. Then serving/releasing a
model is pure vLLM container churn; LiteLLM's config (and the gateway) never
changes. The config only changes when the *catalog* changes (rare, user edit).

Why it's a *medium* change, not a one-liner (scope notes for whoever picks it up):
- The Compose backend / controller only see the **live groups** at converge time
  — they do **not** have the catalog. A superset table needs the full
  endpoint→upstream map threaded CLI → controller → backend → `render_compose`.
- Upstream service names are currently `vllm-<group.id>` where `group.id` is
  **random per group**. A static table needs **deterministic** upstream targets
  (e.g. a compose network alias = the served name) so a route is valid before /
  across (re)creations. That ripples into the placement sidecar (keyed by group
  id) and `observe()`'s service→group mapping.
- Philosophical shift: `/v1/models` would then advertise the whole catalog,
  including models not currently deployed (legacy lived with this; clients retry
  until the upstream is up). Mitigation: LiteLLM's `/health` (and
  `/health/readiness`) probes each configured model's upstream, so "configured"
  (`/v1/models`) and "actually deployed/up" (`/health`) are separable — a client
  or UI can show undeployed models as down rather than guessing. Worth wiring
  into the demo/docs if we go superset, so the side effect is explicit, not a
  trap.

### Deterministic group ids / upstream service names (prereq for #1)

Today a `DeploymentGroup.id` is random (`grp-<hex>`) and the compose service /
LiteLLM upstream are derived from it (`vllm-<id>`, `http://vllm-<id>:8000`). For
#1 the upstream must be predictable. Recommendation, in order:

- **Preferred — derive the id from the compatibility key:** `grp-<short hash of
  compat_key>`. Coalescing already keys on the compat_key, so the id becomes a
  pure function of (model, runtime, served_name, …): same logical model → same
  id → same service name → stable upstream. Not a footgun — the compat_key *is*
  the identity that defines coalescing. Dedicated groups (which must not
  coalesce) get the same hash **plus an owner/nonce salt** so they stay unique.
- **Lower-risk alternative — keep ids opaque, add a deterministic network
  alias:** leave `group.id` random but give each vLLM service a compose network
  alias equal to its served name, and have LiteLLM route to the alias. Decouples
  container identity (random, collision-free) from network identity
  (deterministic) without touching the sidecar/observe keying. Pairs naturally
  with #1's static table.
- **Not recommended — per-user ids.** The leasing model coalesces *across* users
  (two users wanting the same model share one deployment — the whole point), so a
  per-user prefix either breaks coalescing or duplicates deployments. This is the
  footgun the maintainer worried about; avoid it. Per-user isolation is what the
  `--dedicated` flag is for, handled by the nonce salt above.

### Keep-warm vs. live demand on a GPU-constrained host

Surfaced by the GPU e2e suite: a `reclaim: keep-warm` group survives `release`
and keeps its GPU. On a host where few GPUs are usable (e.g. yardrat: only GPU 0,
GPU 1 is display), one idle keep-warm group starves later acquires — they can't
place and time out. The e2e harness now resets between tiers to isolate this, but
the product question stands: **should a new lease with real demand be able to
reclaim an idle keep-warm group's GPU** (evict-to-make-room) rather than fail to
place? Probably yes, with a policy knob. Until then, keep-warm on a 1–2 GPU box
is effectively "pin a GPU."

### #2 — DB-backed live model management (retry later)

Bring back `postgres-litellm` and use LiteLLM's admin API (`/model/new`,
`/model/delete`, `STORE_MODEL_IN_DB`) so converge updates routes on a **running**
gateway — zero blip, no restart at all.

**We tried this before and hit issues** (per the maintainer; details to recover
from the legacy stack's history). Not the current direction, but **worth
retrying in the future** — it's strictly nicer than #1 (no blip, and no
advertising of undeployed models) if the earlier problems can be resolved. When
revisiting, capture *what* broke last time before re-committing to it.

### Recommendation / sequencing

Do #1 as a focused change after the GPU e2e baseline is fully green (so the
rendering redesign doesn't destabilize tiers still being validated). Keep the
config-hash label as the correctness backstop even after #1 (it's harmless when
the config is static — same hash → no restart).

---

## Other noted smells (lower priority)

See `dev/leasing-demo.md` §"Ergonomic smells" for: no persisted default backend
(`--backend compose` repeated), storage location being env-only for the leasing
path, no endpoint-addressed teardown for standing `serve` leases, unmanaged Open
WebUI, and no managed `HF_TOKEN` slot.

Also: reclaim:stop deployment groups linger in the ledger as `state=stopped` and
still show up in `infer-stack leases` / its group list. Decide whether `leases`
should hide/prune stopped groups or keep them as history. (Surfaced by the
`50_coalescing` e2e tier needing to filter to `state==live`.)
