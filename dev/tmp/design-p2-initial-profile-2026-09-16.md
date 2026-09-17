# P2, last step: initial profile and explicit Compose environment

Status: proposal for review before implementation. Everything else in P2 has
landed on `dev/0.7.1` (see CHANGELOG). The plan (revision 7a, §4.2, §5 step 3)
says only: "initial profile persisted once, rendering only from it, explicit
Compose environment". Implementing that literally raises four user-visible
questions the plan does not answer. This note proposes answers.

## What renders read today (facts)

`_make_backend` (`cli/commands_leasing.py`) builds `ComposeBackend` on **every
command** from:

| input | source today | per-invocation flag? |
|---|---|---|
| backend kind | `--backend`, `INFER_STACK_BACKEND`, setting | yes |
| litellm, ui, dynamic_routing, skip_display_gpus | flag, else setting | yes |
| reverse proxy (enabled, port, config path) | setting block, flag overrides enabled | yes |
| allowed_gpus | `--allowed-gpus` | yes |
| images | `PINNED_IMAGES` from the installed package | no |
| ports, state paths | `DEFAULT_PORTS`, `default_state_paths()` | no |
| catalog (static-superset route table) | `--catalog`, `INFER_STACK_CATALOG`, default path; missing -> None | yes |
| secrets | managed `.env` (master key, DB password) | no |
| `HF_TOKEN` | **the caller's shell**, via `${HF_TOKEN:-}` in the YAML | n/a |

So two processes, or one process before and after an upgrade, can render
different projects from the same ledger. With serialised publication, recovery
re-renders, and then that difference becomes a real bug: recovery could
recreate unrelated services.

## Proposal

### 1. Profile contents

A single JSON row, `meta.profile`, with `version` and `digest`, containing:

- `backend`: `compose` or `kubeai`. The `null` backend never persists one.
- **Compose:** `project`, `litellm`, `ui`, `dynamic_routing`,
  `skip_display_gpus`, `reverse_proxy {enabled, port, config_path}`,
  `allowed_gpus`, `images` (resolved), `ports`, `state` paths, and `catalog`.
  `catalog` is the parsed catalog dict plus its digest, or `null`.
- **KubeAI:** `namespace`, `base_url`, `resource_profile`.

Secrets are **not** in the profile. They stay in the managed `.env`, which is
already persistent and not per-invocation.

### 2. When it is created

Implicitly, **on the first controller mutation against a ledger with no
profile**, resolved from that invocation's settings exactly as today, and
logged. A mutation that is not the first renders from the stored profile.

The plan says "one-time, explicit, on upgrade". An explicit step means every
command fails until someone runs it. Implicit-once gives the same determinism
without that trap.

### 3. How settings change before P4's `config publish`

This is the real question. Freezing without an exit makes changing any setting
impossible until P4.

**Proposal:** a minimal `infer-stack config publish` in P2 that refuses unless
the stack is **quiescent**: no ACTIVE lease, and strict residency shows no
deployment container. It shows the rendered diff, asks for confirmation
(`--yes`), replaces the profile, and publishes. That matches the design
boundary already documented ("catalog and configuration are fixed during a
leasing epoch"). P4 then adds publishing while leases are live, with
`approved_digest` and image pre-pull.

Per-invocation flags that differ from the profile (`--ui`, `--no-litellm`,
`--dynamic-routing`, `--reverse-proxy`, `--allowed-gpus`,
`--skip-display-gpus`, `--catalog`) are **ignored with a warning** naming
`config publish`. The alternative, refusing, would break existing pipelines that
pass `--catalog` on every call.

### 4. Catalog and acquire

The static-superset route table is rendered from the profile's catalog. For
consistency, **acquire resolves endpoint names from the profile's catalog too**.
An endpoint missing from it fails with "not in the published catalog; run
`infer-stack config publish`".

This is the plan's test 25. It changes behaviour for anyone who edits
`catalog.yaml` and immediately acquires the new endpoint. With the quiescent
publish above, they must publish first, or wait until the stack is idle.

Question: is that acceptable for current workflows? The incubilate and
entrypoint scripts pass `--catalog` with an overlay that can differ between
runbooks sharing one host. If they genuinely use **different catalogs
concurrently**, a single frozen catalog breaks them, and the profile would have
to hold the **union** instead (`routes seed` already builds unions). See open
question A.

### 5. Explicit Compose environment

`_compose` runs `docker compose` with `env=` built from:

- an allow-list from the caller: `PATH`, `HOME`, `DOCKER_HOST`,
  `DOCKER_CONTEXT`, `DOCKER_CONFIG`, `DOCKER_CERT_PATH`, `DOCKER_TLS_VERIFY`,
  `XDG_RUNTIME_DIR`, `LANG`, `LC_ALL`;
- `COMPOSE_PROJECT_NAME` unset, so `-p` alone decides the project;
- `--env-file` from the managed `.env`, unchanged.

`HF_TOKEN` currently comes from the shell. Under an explicit environment it
must come from the managed `.env`.

**Proposal:** on first render, if the `.env` has no `HF_TOKEN`, seed it once from
the invoking shell's `HF_TOKEN`, or else from `~/.cache/huggingface/token` if
readable. Log that it was captured, without the value. After that the shell
value is ignored, and `infer-stack config set-secret HF_TOKEN` (small) replaces
it. See open question B.

`_default_docker_run` needs an `env` parameter; injected test runners ignore it.

## Tests (plan 23, 24, 25, plus these)

- **Frozen after the first mutation.** Changing a setting or the catalog does
  not change the render of a later release (23), and neither does a changed
  `PINNED_IMAGES` (24).
- **Acquire.** An endpoint absent from the profile's catalog fails, naming
  `config publish` (25).
- **Publishing.** `config publish` refuses with an ACTIVE lease or a resident
  container, and with neither it replaces the profile and publishes.
- **Recovery.** A pending marker recovered by a process with different flags
  renders byte-identically.
- **Compose environment.** `_compose` passes no shell variable outside the
  allow-list, and `HF_TOKEN` from the shell does not override the `.env`.

## Open questions

A. **Concurrent catalogs.** Do runbooks sharing one host really run with
   different `--catalog` files at the same time? If yes, the profile holds a
   union, publish merges (like `routes seed`), and acquire accepts any endpoint
   in the union.

B. **HF_TOKEN.** Is seeding it once into the managed `.env` acceptable, given
   that the `.env` already holds the LiteLLM master key and DB password with
   restricted permissions? Or should gated-model tokens stay per-caller, in
   which case the "explicit environment" rule would allow-list `HF_TOKEN` only?

C. **Implicit creation.** Is profile creation on the first mutation acceptable
   instead of an explicit migration command?

D. **Scope.** Is a quiescent-only `config publish` in P2 acceptable, with live
   publish kept for P4?
