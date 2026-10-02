# External endpoints, and access above leasing (design)

**Status:** implemented, 2026-09-27 (queue items 28-41 in
[../queue.md](../queue.md); decision 2 revised during implementation, see
there). Parent: the backend-parity campaign
([backend-parity-roadmap.md](backend-parity-roadmap.md)). Verified by
`tests/test_external_endpoints.py` and `dev/external_e2e.sh` (real LiteLLM,
static and dynamic routing).

## The workflow

```yaml
endpoints:
  qwen:
    external:
      api_base: http://gpu-box:8000/v1
      model: Qwen/Qwen3-32B
      api_key_env: REMOTE_QWEN_KEY
```

```bash
infer-stack env REMOTE_QWEN_KEY=sk-...       # the key, by name; never in the catalog
infer-stack access qwen --env-file qwen.env  # publish the route; no lease
source qwen.env && my-eval --model "$INFER_STACK_ENDPOINT_QWEN"
```

**`api_base` is resolved by the gateway, not by your shell.** LiteLLM runs
in a container (Compose, KubeAI's host gateway) or a pod (the in-cluster
gateway), so `http://localhost:8000/v1` means the gateway itself, not the
host you typed it on. A server on the same host is reached through the
host's address as the gateway's network sees it: on Compose, the Docker
bridge (`http://172.17.0.1:8000/v1` on a default install; `ip -4 addr show
docker0`), with the server listening on that interface, not only on
127.0.0.1; from the in-cluster gateway, a node address or a Service name.
A server elsewhere is reached by any name or address the gateway can
resolve. `infer-stack test <alias>` checks the whole path.

Switching `qwen` back to a managed runtime (`engine: vllm`, `model: ...`)
changes neither the alias nor these commands: `access` then takes a lease and
waits for the model, and the env-file gains `INFER_STACK_LEASE_ID`. A bundle
may mix both; it takes one lease, for its managed members. `run --endpoint
qwen -- CMD` does the same around a command. `acquire` stays the lease
operation and points an external endpoint at `access`.

An endpoint is the public name a workflow asks for (`qwen`). Today every
endpoint is a model infer-stack runs, so asking for one means taking a
lease. This change lets an endpoint name a model server that already runs
elsewhere, and uses that to separate three things that are currently one:

| Layer | What it is | Owner |
|---|---|---|
| Endpoint | the public contract: alias, protocol | the catalog |
| Fulfillment | a managed runtime (vLLM, Ollama) or an external OpenAI-compatible upstream | the catalog's target |
| Lease | protection of a runtime infer-stack runs | the ledger, only for managed targets |

## The chain, after the change

```text
catalog definitions (the invocation's, merged into the published union)
    |
    v
ResolvedEndpoint(alias, protocol, target)       one semantic key per endpoint
    |-- ManagedTarget  -> EndpointRequest -> ledger (lease, deployment)
    '-- ExternalTarget -> nothing in the ledger
    |
    v
GatewayRoute(alias, kind, model, api_base, key_env, route_id?)
    |-- static:  derived from the published union (+ live deployments)
    '-- dynamic: the same routes, reconciled through LiteLLM's admin API
    |
    v
front-door publication (render, approve, apply: one transaction)
    |
    v
AccessResult(connection info, alias -> request name, optional lease)
```

Each arrow has one implementation; each box one owner.

## Decisions

### 1. Published endpoint lifetime

Today two things persist endpoint meaning: the recovery profile's
`catalogs` (replaced wholesale when an acquire finds the stack quiescent,
`Controller._acquire_profile_candidate`) and the route registry (append-only
rows derived from those catalogs, `_merge_route_registry`). They agree only
because routes outlive the catalog that made them. An external endpoint has
no lease and no residency to pin it, so it would be kept by whichever copy
happened to remain.

**Decision.** The profile's `catalogs` is the published catalog union, the
one durable store of endpoint definitions, with its own lifetime:

- every acquire and access merges the invocation's catalogs into it (the
  existing merge path: identical definitions coexist; a different
  definition of an unpinned endpoint replaces it; of a pinned one, refuses);
- quiescence permits replacing the frozen settings (backend, ports, images,
  gateway placement) as today, but no longer replaces the catalog set: an
  unrelated runbook or a settings change does not unregister anything;
- `routes prune` (renamed in help to what it now does: unpublish) removes
  published definitions the invocation's catalogs do not have and no live
  deployment uses; it is the explicit removal, confirmed as today.

Visible behaviour for managed endpoints does not change: their routes
already outlived catalog replacement through the append-only registry.

### 2. Route ownership: derive, do not store

Three owners, never mixed in one map:

| Routes | Derived from | Stored |
|---|---|---|
| catalog routes | the published union | no: derived at render |
| deployment routes (static) | placed deployments | remembered in `litellm_registry.json` past release, only for an alias no published catalog defines |
| dynamic routes | placed deployments, and external targets | reconciled into LiteLLM's DB (as today) |
| older registry rows | a `litellm_registry.json` an older version wrote (catalog rows, seeded siblings) | read, lowest precedence; `routes prune` drops the ones nothing uses |

The registry file stops being written with catalog-derived rows: once the
union is durable, it carries nothing the union does not. It keeps one owner:
ad-hoc deployments (no catalog defines them), whose route must outlive their
release or releasing one recreates the gateway (a blip for every client).
*Revised in implementation:* the first draft derived those only while live;
`test_converge_to_empty_keeps_the_front_door` showed the blip. Existing files
are still read, so an upgrade loses no route; `routes list` marks their rows
`registry`. `routes seed FILE...` becomes "merge these catalogs into the
published union" (same user-visible effect: the gateway serves their
aliases; conflicts refuse unless `--replace`). `CatalogUnion` then decides
conflicts from endpoint semantics alone, and the import of
`_registry_incoming_from_catalog` goes.

### 3. One access transaction

`Controller.acquire` already runs one admission transaction: sync the
profile, preview (placement and render), commit the profile candidate, the
marker and the lease together, render, apply, wait. Access is the same
transaction with the managed request set possibly empty and the external
definitions carried by the profile candidate. One preview, one approval
digest, one marker, one render and apply, for managed-only, external-only and
mixed alike. `acquire` stays the public lease operation over it and refuses
an external-only request with a pointer to `access`.

### 4. Credential lifecycle

- **Reference, never value.** `external.api_key_env: NAME`; the route renders
  `api_key: os.environ/NAME`; the catalog, the published profile, routes,
  fingerprints and logs hold the name only.
- **Host gateway (Compose, KubeAI's host gateway).** The LiteLLM service's
  environment lists each referenced name (`NAME: ${NAME}`), so the existing
  per-service fingerprint (which hashes the values of the variables a stanza
  references) recreates LiteLLM when a value changes.
- **In-cluster gateway.** Referenced names go into its Secret; the key-hash
  annotation covers every value in it, so a change rolls the pods; values
  never appear in the manifest or a diff.
- **`infer-stack env NAME=value`** stays a file write. When NAME is referenced
  by a published external endpoint it says so: "the gateway uses it after
  `infer-stack apply`". No silent staleness, and `env` does not become a
  publication command.
- **Validation.** `NAME` must be a valid environment name and not one of
  infer-stack's own (`LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`,
  `LITELLM_DB_PASSWORD`, `WEBUI_SECRET_KEY`, `HF_TOKEN`): a catalog must not
  be able to send the gateway's own credentials to a server it names. A
  referenced name with no value fails before apply; provider keys are never
  generated. *Implemented at the render* (queue item 44): every publication,
  whichever command caused it, refuses while a published route's key has no
  value, so an unrelated operation cannot recreate the gateway without it.

### 5. Stable route identity

A dynamic route's managed id is its logical owner: `(deployment id, alias)`
for a managed deployment (unchanged: one alias can have several dedicated
deployments), `('external', alias)` for an external target. api_base, model
and key name are content; the semantic route compares content to decide a
replacement. Redefining `qwen` from server A to B replaces one route.

### 6. Front-door readiness

An external upstream's readiness is not infer-stack's; its own gateway's is.
Managed access already waits for a generation through the front door.
Access with external members also waits until the front door accepts the
master key (`gateway_accepts`), bounded, and fails if it never does. No
request is sent to the external upstream.

### 7. Failure semantics

- A managed member that fails and rolls back leaves the external
  definitions published (they are catalog state, not tied to the call).
- A process that dies after publication leaves the routes; a committed lease
  follows TTL and recovery as today. There is no access handle to clean up.
- Removing an external endpoint from the published union (`routes prune`)
  removes its route on the next publication.

### Also settled

- **Front door only.** An external endpoint without a LiteLLM front door
  fails clearly. Direct access is a later option, not this change.
- **Views.** `leases` stays leases and deployments. External endpoints show
  in an endpoint/routing section of `status`, in `catalog show`, `routes
  list` and the TUI's catalog, as `external`.
- **Names.** Endpoint alias (what clients request), upstream model (what the
  target server expects), deployment id. `public_name`/`served_name` in YAML
  stay as spellings of the upstream model name; new code uses the three
  terms.

## What goes away

- catalog-derived rows in `litellm_registry.json` (it keeps only ad-hoc
  deployments' routes);
- `CatalogUnion`'s route-row comparison and its import of
  `_registry_incoming_from_catalog`;
- the `upstream` pseudo-engine (`UPSTREAM_ROUTE`): a KubeAI or external route
  is a `GatewayRoute` of kind `openai` with an api_base;
- the registry row shapes `{engine, served, host}` and their second
  interpreter in `routes list`: one `GatewayRoute` and one renderer;
- `render_front_door`'s four route strategies: it takes a route table, or
  dynamic routes, from `front_door_routes`;
- the seed-from-`litellm_config.yaml` upgrade path (it would make every
  rendered route permanent once the registry stopped being written);
- `backend.access(endpoints) -> dict`: the front door gives connection info,
  endpoints give request names, an `AccessResult` combines them;
- a descriptor that needs a lease (`INFER_STACK_LEASE_ID` is written only
  when there is one);
- two unrelated types named `FrontDoor` (the capability becomes
  `FrontDoorControl`, the rendered artifact `RenderedFrontDoor`).

## Alternatives rejected

- **A fake lease for an external endpoint** keeps today's API shape but puts
  demand, TTL and reclaim state in the ledger for something with no runtime.
- **`engine: external`** mixes how a runtime is realized with who owns it.
- **Authoring external endpoints as route rows** makes the registry a second
  catalog, outside the conflict and pinning rules that protect meaning.
- **Direct access to an external server** gives external-only requests a
  second access mode where the alias stops being the model name.
