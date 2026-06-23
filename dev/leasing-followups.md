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
#1 the upstream must be predictable. **Decision (maintainer): use the compat-key
hash (option 1) — after the GPU e2e dashboards are green.** Recommendation, in
order:

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

### Keep-warm should yield its GPU to live demand (resolved policy)

Surfaced by the GPU e2e suite: a `reclaim: keep-warm` group survives `release`
and keeps its GPU. On a host where few GPUs are usable (e.g. yardrat: only GPU 0,
GPU 1 is display), an idle keep-warm group starves later acquires — they can't
place and time out.

**Resolved policy (maintainer):** the decision turns on what "idle" means, which
is unambiguous here because `demand` counts active, un-expired leases:

- **LIVE (demand ≥ 1, a lease is held): never reclaim.** A held lease protects
  the group whether or not requests are currently flowing — "I'm holding this
  GPU, don't touch it." Respect it.
- **IDLE (demand == 0: every lease released, or TTL-expired and not renewed):
  reclaimable.** No one is holding it; it's warm only as a courtesy. A new lease
  with real demand may **evict it to free the GPU** instead of failing to place.

So keep-warm means "stay resident *if there's room*; yield to real demand when
GPUs are contended" — not "pin a GPU forever." Implementation: when placement
fails for a LIVE-demand group, allow it to reclaim an IDLE group's GPU (tear the
idle one down, place the new one). The ledger already distinguishes LIVE vs IDLE
by demand, so this is a placement/reclaim policy change, not a new state. (The
e2e harness resets between tiers regardless, so this isn't blocking the suite.)

**Manual eviction landed** (the *explicit* half of this): `infer-stack evict
[NAME…|--all]` and `release --evict` mark IDLE groups `stopped` so the next
reconcile tears them down, freeing the GPU on demand. The *automatic*
yield-under-pressure above (placement evicting an idle group when a live acquire
can't otherwise place) is still open — `evict_idle` / `Controller.evict` are the
reusable primitive it should call once the placer detects contention.

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

See `docs/source/manual/leasing-demo.md` §"Ergonomic notes". RESOLVED by the CLI reorg
(`dev/cli-redesign.md`): persisted default backend (`config set backend`),
durable storage location (`config set data_dir`, honored by `data_root()`), and
a managed `HF_TOKEN` slot (`secret set HF_TOKEN=…`). ALSO RESOLVED: Open WebUI
is now managed — bundled into the leasing compose stack, on by default
(`--no-ui` / `config set ui false`), and rendered stable across model switches
so it isn't recreated when routing changes (see `infer_stack.leasing.compose
._open_webui_service`). STILL OPEN: no endpoint-addressed teardown for standing
`acquire` leases (want `release --endpoint <name>`).

Open WebUI sub-follow-ups: it currently runs with `WEBUI_AUTH=False` (single-user
workstation assumption) — expose an auth/port knob before anyone points it at a
shared host. And like LiteLLM, the UI's own recreation on key rotation isn't
specially handled (rare; the baked `OPENAI_API_KEY` changes the spec, so it
recreates — acceptable).

Also: reclaim:stop deployment groups linger in the ledger as `state=stopped` and
still show up in `infer-stack leases` / its group list. Decide whether `leases`
should hide/prune stopped groups or keep them as history. (Surfaced by the
`50_coalescing` e2e tier needing to filter to `state==live`.)

### Ledger accumulation / leaked-active-lease (real, found by GPU e2e)

The GPU e2e suite surfaced a cluster of related product issues (worked around in
the harness by wiping the ledger between tiers; worth fixing in the product):

- **A not-ready `acquire` keeps its lease ACTIVE with no auto-cleanup.** Unlike
  `run` (which releases in a `finally`), a plain `acquire` that times out / never
  becomes ready leaves the lease active, so its group stays LIVE and holds a GPU
  indefinitely. Decide: should `acquire --timeout` that fails auto-release? At
  minimum it's a footgun — a failed acquire silently pins a GPU.
- **Re-acquiring a model spawns a new group instead of reviving a compatible
  idle one.** Repeated acquire/release of the same model accumulates distinct
  group rows (different ids) rather than coalescing onto the existing idle group.
  Fixing this is largely the deterministic-group-id work above (a compat-key id
  makes "the qwen-small group" a single stable row).
- **Ledger vs. reality drift.** If a group's container is removed out-of-band
  (here: the harness `compose down` between tiers), the ledger still believes it
  is LIVE and the next reconcile re-realizes it. Reconcile trusts the ledger as
  desired-state and never prunes a LIVE group whose lease leaked. Combined with
  the first bullet, one leaked active lease re-spawns a GPU-hogging group on
  every subsequent reconcile. An `observe()`-driven reconciliation of ledger
  state (or just fixing the leak) would close this.
