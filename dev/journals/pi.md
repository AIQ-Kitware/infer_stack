# pi (qwen3.8-27b-dbirks-hyperqwen-full, via LiteLLM)

## 2026-10-02 20:33:17 -0400

Summary of user intent: user wants infer-stack to "work with aivm" so that a
managed aivm VM can run `pi` against this stack's LiteLLM gateway — the right
TCP endpoints opened in the aivm firewall and the right settings written into
pi inside the VM — without an aivm-specific command in infer-stack (feature
creep). Asked for an exploration + design-spec planning pass, not an
implementation; a script in `dev/` is the candidate temporary home, with the
hard parts later delegated to a stronger agent. Also raised whether
infer-stack itself could serve a self-signed HTTPS endpoint instead of the
current allow-HTTP workaround.

Model and configuration: pi coding agent, model
`qwen3.8-27b-dbirks-hyperqwen-full` served through a self-hosted LiteLLM
proxy (the very integration being designed), no special weights/config
variant known beyond the identity.

What I did: discovery only, no code changes. Wrote the design spec at
`dev/aivm-pi-bridge.md`. Key verified facts:

- Gateway: litellm v1.82.3-stable, container 4000, host 14042, master key in
  the compose env file (`LITELLM_MASTER_KEY`, `sk-` prefix,
  `set_master_key`/`rotate_master_key` in `infer_stack/leasing/gateway.py`);
  `ports` binding is `0.0.0.0` (confirmed on the live host); `Gateway.urls()`
  advertises `127.0.0.1` only, so a VM client needs the aivm bridge gateway
  IP instead. No TLS anywhere in the stack today; the optional nginx proxy
  (443) is the natural TLS point.
- pi client: the provider is the npm extension `pi-provider-litellm` v3.3.0.
  A live VM on this very machine gave the ground-truth config shapes:
  `~/.pi/agent/auth.json` (`{litellm: {type: api_key, key, env:
  {LITELLM_BASE_URL}}}`, mode 0600 — the shape `/login` itself persists),
  `~/.pi/agent/settings.json` (`litellm.providers.litellm` with
  `baseUrl`/`apiKey`/`allowInsecureHttp` plus top-level
  `defaultProvider`/`defaultModel`), and `models-store.json` (discovery
  snapshot, `/model/info` → `/v1/models` → `/health`). `allowInsecureHttp`
  is the current plain-HTTP workaround; the extension has no CA-cert option,
  so self-signed TLS would ride on `NODE_EXTRA_CA_CERTS` (unverified against
  pi's Node 22) or an upstream feature request. Discovered that `pi install
  -l` is the *project*-scope install (needs project trust) — the TODO in
  aivm's pi installer should drop the `-l` for a VM-wide setup.
- aivm: machine-store `[[networks]]` carry `network.gateway_ip` +
  `firewall.allow_tcp_endpoints` (`IPv4:PORT`), VMs reference networks by
  name, mutation is first-class (`parse_store_toml` / `upsert_network` /
  render), application is `aivm vm update --yes` (live nftables reconcile,
  no restart). Because the forward filter matches the pre-DNAT conntrack
  original, one endpoint entry `<gateway_ip>:14042` covers the Docker-published
  port even after Docker DNATs it to the container's 4000 — the historical
  reason a broad `4000` used to be needed is gone.

Reflection / state of mind: the pleasing part of this task was that the
"hard" integration turned out to be three small, well-owned surfaces — no
new protocol is needed at all. infer-stack already publishes everything a
client needs (URL, key, model list via discovery), aivm already has narrow
endpoint exceptions plus a live-reconciling update, and pi's extension
persists exactly the credentials a script would write. The whole bridge is
data plumbing between two existing config stores, which is why a `dev/`
script is an honest home: it is composition, not capability. The design
decision worth keeping in the future: the *host-side* exposure (0.0.0.0
publish to whatever LAN the GPU box sits on) is the only risk the aivm
firewall doesn't cover, so the cheap win is binding the publish to the
bridge IP, not jumping to TLS. I kept the HTTPS question explicitly
deferred with a verification checklist rather than designing cert lifecycle
speculatively.

Uncertainties / risks: (1) non-interactive `aivm ssh <vm> <cmd>` — the CLI
treats ssh as a foreground session workflow, so scripted guest steps may
need the managed ssh key directly or a confirmed `--yes` path; (2)
`NODE_EXTRA_CA_CERTS` with pi's Node 22 is untested (blocks option B); (3)
store write-back must round-trip through aivm's own parse/render to survive
hand-edited machine stores — the spec requires `--dry-run` to show the
exact store diff before any write; (4) the litellm extension registers MCP
tools and Skills injection by default — fine for a self-owned proxy, but
the script's docs should carry the trust note.

Tradeoffs: chose to document the *observed working* config shapes (auth.json
0600 as the credential home, matching `/login`'s own persistence) over the
settings.json `apiKey` variant, since file-mode hygiene matters more than
one fewer file. Chose to keep the bridge host-side only on the infer-stack
knowledge and let aivm own all guest/firewall operations — the alternative
(aivm-side hook) would leak LiteLLM knowledge into aivm, which the user
explicitly wants to avoid.

What I'm confident about: the endpoint math and the full manual procedure —
this exact setup is running on the machine I'm executing on, so every
claimed config shape was read, not inferred. The spec is a verified map, not
a guess.

## 2026-10-02 21:33:08 -0400

Implemented the LiteLLM context-window metadata change (plan in
`dev/tmp/notes-litellm-context-metadata-2026-10-02.md`). The goal: pi (and
any client reading `/v1/model/info`) should learn the true context window
instead of defaulting to 128k for custom model names like
`qwen3.8-27b-dbirks-hyperqwen` whose vLLM process runs with
`--max-model-len 262144`.

What I did, in dependency order:

- `leasing/launch.py` — moved `VLLM_DEFAULTS` here from `compose.py` and
  added `effective_max_model_len(runtime)`: the one derivation of the
  window a vLLM deployment launches with (runtime's `max_model_len` after
  `translate_legacy`, else the default 8192; no coercion, so launch and
  advertisement can never diverge).
- `leasing/routes.py` — `GatewayRoute` gained a trailing
  `max_input_tokens: int | None = None` (participates in `__eq__`);
  `entry()` now builds `model_info` incrementally, adding
  `max_input_tokens` whenever known, including routes without a `route_id`.
- `leasing/gateway.py` — every route-producing path now advertises the
  effective window: static compose (catalog + deployment routes), dynamic
  `/model/new` (deployment routes carry it; `_route_semantics` includes it,
  so a window change is drift and the route is replaced under its id),
  `upstream_route` (KubeAI Models), and the remembered registry
  (`remembered_rows` writes it for vLLM rows and known upstream rows;
  `_remembered_context` guards hand-edited/legacy rows: positive whole
  number only, else silently unknown — never a wrong number).
- `backends/kubeai.py` — same propagation for the KubeAI frontend: catalog
  routes, the dynamic per-(deployment, endpoint) upstream routes, and the
  static per-alias model routes.
- `tests/test_leasing_context_metadata.py` (new, 24 tests) — serialization,
  static compose (superset + live), default consistency between
  `vllm_service_dict` and the advertisement, dynamic `/model/new` bodies
  and window-change drift, the remembered registry (new/legacy/junk rows),
  one deployment with two aliases, and the KubeAI static + dynamic paths.

A consequence worth remembering: context is an endpoint contract, while
`max_model_len` is also a deployment-capacity field. Capacity subsumption
means a 65K catalog alias can legitimately share a compatible 262K process.
The alias must still advertise 65K, both to keep its public contract stable
and to preserve the static gateway's byte-stability across acquire/release.
The catalog resolver therefore records `max_input_tokens` in each vLLM
endpoint's `served` payload; route rendering uses that per-alias value, with a
catalog fallback for deployments persisted before the field existed and the
deployment runtime only as the ad-hoc fallback.

LiteLLM can independently populate `max_input_tokens` from its bundled model
metadata for an external route even when infer-stack did not set a window.
Dynamic reconciliation must ignore such synthesized context when the desired
route has no infer-stack-owned value; otherwise a known external model is
deleted and re-added forever. When infer-stack does publish a window, drift
remains strict and still forces replacement.

Validation (CPU-only; this VM has no Docker, K8s, or a live LiteLLM):
`pytest --xdoctest infer_stack tests` → 1184 passed, 8 skipped; flake8
(E9,F63,F7,F82) clean; compileall clean. I also rendered a real config to
confirm the shape: the `model_list` entry carries
`model_info: {max_input_tokens: 262144}` and the dynamic `/model/new` body
carries `model_info: {id: isr-…, max_input_tokens: …}`.

What I could not test: that a live LiteLLM 1.82.3 actually returns
`max_input_tokens` in its `/v1/model/info` response, and that pi picks it
up as `contextWindow`. My reading of the LiteLLM source (v1.82.3: the
config `model_info` is merged first and wins over the cost map, and
`max_input_tokens` is a first-class `ModelInfo` field) says yes, but the
live end-to-end check belongs to the next GPU-VM session. That is also the
one operational wrinkle: the first converge after upgrading rewrites
`litellm_config.yaml` (new `model_info` key), so the `config-hash` label
changes and the gateway container is recreated once — expected, one-time.

Confident: the derivation is centralized (launch.py), the invariant is
enforced by construction on every path, and the guard on remembered rows
fails closed. Uncertain: whether any client other than pi treats
`max_input_tokens` differently from what the cost map would have supplied
— worth a spot-check of the gateway's `/v1/model/info` after the live test.

Also: `AGENTS.md` picked up aivm's commit-attribution rule (agent commits
carry a `Co-authored-by:` trailer naming the model that produced them);
infer_stack had no equivalent.
