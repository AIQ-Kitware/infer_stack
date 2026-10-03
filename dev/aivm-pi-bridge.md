# aivm × pi bridge: design notes (temporary home)

Status: **design only, not implemented.** This document codifies the bridging
logic that makes a user's aivm VM able to run `pi` against this stack's
LiteLLM gateway, so that whoever implements the script (possibly a stronger
agent) does not have to rediscover the ground truth. If this ever graduates
to a real feature, move it to `docs/` and pair it with an ADR; until then it
lives here, next to the temporary script it describes
(`dev/aivm_pi_bridge.sh` or `.py`, to be created).

## Purpose and ownership split

After `infer-stack acquire <model>` on a host, the user wants `pi` on a
managed aivm VM to work against that host's gateway with no manual
configuration. The tools each own one half of the truth:

- **infer-stack** knows the *endpoint*: the LiteLLM host port, the master
  key, the gateway health, and the acquired model ids.
- **aivm** knows the *VM network*: which bridge network a VM is on, its
  gateway IP, how to reach into the guest (`aivm ssh`), and the guest's
  firewall exceptions (`firewall.allow_tcp_endpoints`).
- **pi** is a passive consumer: it needs a provider registration, a
  credential, and a default model in its per-user config.

The bridge is a small script that composes the two tools' *public* surfaces.
Deliberately not: a first-class `infer-stack aivm` subcommand (feature creep
for a niche pairing — the stack must not know what aivm is), and not an
aivm-side hook that knows about LiteLLM (aivm must not know what
infer-stack is). Promotion criterion: a second consumer (another agent
client, multi-VM fleets, or a second target platform) is what would justify
a generic "clients/connectors" abstraction in infer-stack.

## Ground truth (verified 2026-10-02)

### The gateway (infer-stack side)

- Compose service `litellm`: image `ghcr.io/berriai/litellm:v1.82.3-stable`
  (`infer_stack/config.py::PINNED_IMAGES`), container port 4000, host port
  14042 (`DEFAULT_PORTS['litellm']`, user-configurable).
- The published port is a plain `f'{host_port}:4000'` mapping
  (`infer_stack/leasing/gateway.py`), so Docker binds **0.0.0.0** on the
  host (confirmed live: `0.0.0.0:14042->4000/tcp`).
- The master key is generated with an `sk-` prefix and stored in the
  compose project env file: `API_KEY_ENV = 'LITELLM_MASTER_KEY'`,
  `set_master_key()` / `master_key()` / `rotate_master_key()` in
  `gateway.py`. The rendered LiteLLM config references it as
  `os.environ/LITELLM_MASTER_KEY`.
- `Gateway.urls()` advertises `http://127.0.0.1:14042/v1` —
  host-local only. A VM client needs the *host's bridge-side IP* instead
  (the IP aivm calls the gateway IP).
- There is **no TLS/HTTPS anywhere** in the stack today. An optional nginx
  reverse proxy exists (ports 80/443, `DEFAULT_PORTS['reverse_proxy_*']`)
  and is the natural TLS-termination point if we ever want one.

### The client (pi side)

The provider is the npm extension **`pi-provider-litellm` v3.3.0**
("LiteLLM proxy provider extension for Pi", requires Pi 0.83+; MCP parts
need 0.99+). Verified against a live VM (`~/.pi/agent/`):

- **Install**: `pi install npm:pi-provider-litellm` — the *personal*
  (global) install, writing the package declaration to
  `~/.pi/agent/settings.json`. Note: `-l`/`--local` means the *project*
  `.pi/settings.json`, which loads only after project trust is granted —
  the wrong scope for a VM-wide setup. (The TODO in aivm's pi installer
  currently says `pi install -l …`; it should drop the `-l`.)
- **Credential/endpoint config surfaces** (any one works; the first is what
  a live, working VM shows):
  1. `~/.pi/agent/auth.json` (mode 0600):
     `{"litellm": {"type": "api_key", "key": "sk-…", "env": {"LITELLM_BASE_URL": "http://<gw>:<port>"}}}`
     — the exact shape the extension's own `/login` persists.
  2. `~/.pi/agent/settings.json`:
     `{"litellm": {"providers": {"litellm": {"baseUrl": "http://<gw>:<port>", "apiKey": "…", "allowInsecureHttp": true}}}}`
     (`apiKey` may be `$ENV` / `!command` / literal).
  3. Env: `LITELLM_BASE_URL` + `LITELLM_API_KEY` (lowest precedence).
- **`allowInsecureHttp`** (default `false`; loopback is exempt) is the
  current plain-HTTP workaround for non-loopback `http://` base URLs.
  There is **no CA-cert / self-signed option** in the extension's documented
  fields; trusting a self-signed cert would have to go through Node's
  `NODE_EXTRA_CA_CERTS` (unverified against pi's Node 22) or an upstream
  feature request.
- **Model discovery**: `/model/info` (admin) first, falling back to
  `/v1/models`, then `/health` + per-route probes. Results persist to
  `~/.pi/agent/models-store.json` (`litellmDiscoveryVersion` marker), so
  the guest keeps a working catalog even if the extension is later absent.
  `/litellm-refresh` re-discovers on demand. The extension also registers
  LiteLLM MCP tools and Skills Gateway injection *by default* — fine for a
  self-owned proxy, but the design should mention the trust note.
- **Default model/provider**: `settings.json` top-level
  `defaultProvider` / `defaultModel` (e.g. `litellm` /
  `qwen3.8-27b-dbirks-hyperqwen`).

### aivm side

- The machine store (host, `/var/lib/aivm` for fresh 0.6 installs) holds
  `[[networks]]` records: `[networks.network]` (`name`, `bridge`,
  `gateway_ip`, …) and `[networks.firewall]` (`enabled`, `block_cidrs`,
  `allow_tcp_ports`, `allow_tcp_endpoints`, …). `[[vms]]` records carry
  `network_name`, so the VM→network mapping is explicit
  (`aivm/config_store/models.py::VMEntry`).
- `firewall.allow_tcp_endpoints` takes `IPv4:PORT` strings and generates:
  an input-hook accept for traffic addressed to that host, and a
  forward-hook accept on the **pre-DNAT** conntrack original tuple
  (`aivm/firewall.py::_nft_script`). Because of that, one entry
  `<gateway_ip>:<litellm_host_port>` covers the Docker-published port even
  though Docker DNATs it to `<container_ip>:4000` (the pre-v0.6.0 failure
  that made people add the *container* port to `allow_tcp_ports`; see
  aivm `dev/journals/gpt56.md`, fix commit 42c8b47).
- Programmatic mutation is first-class:
  `aivm.config_store.parse.parse_store_toml` →
  `aivm.config_store.mutate.upsert_network` → render/write. Applying is
  `aivm vm update --yes <vm>` (reconciles the managed nftables table live,
  policy-fingerprint driven, no VM restart) or `aivm firewall apply`.
- Guest access is `aivm ssh <vm>` (managed access identity; verify the
  exact non-interactive command form before scripting it — the CLI treats
  it as a foreground session workflow). Pi itself is provisioned by
  `aivm vm provision pi` (managed install under `~/.pi/agent`, official
  installer, Node bootstrap).

### Concrete observed instance

This design was discovered from a live VM: default route `10.77.0.1`
(gateway), VM address `10.77.0.119`, `settings.json` with
`baseUrl: http://10.77.0.1:14042`, `allowInsecureHttp: true`, and 86
discovered models in `models-store.json` — i.e. the manual version of this
bridge already works; the script just automates and idempotently repeats it.

## The bridge algorithm

Runs on the **host**. Inputs: target VM name, `--model` (optional; default:
a sensible choice from the acquired endpoints), `--dry-run`, `--yes`.
Fail closed at each stage; never echo the master key.

**Host side**

1. Read infer-stack state: litellm enabled, host port, master key (env
   file in the generated dir), and probe the gateway
   (`/health` or `/v1/models` with the key). Abort with a clear message if
   the gateway is down.
2. Read the aivm machine store; find the target VM's `network_name`; take
   that network's `gateway_ip`.
3. Compute `endpoint = "<gateway_ip>:<litellm_host_port>"`; merge it into
   that network's `firewall.allow_tcp_endpoints` (deduplicated; leave
   `allow_tcp_ports` alone). Write the store back through aivm's own
   parse/mutate/render round-trip so formatting, schema version, and
   sibling records are preserved.
4. Apply: `aivm vm update --yes <vm>`; confirm the endpoint shows up
   (`aivm firewall status` or `nft list`), and optionally TCP-probe the
   endpoint from the guest.

**Guest side** (via `aivm ssh` or the managed ssh key)

5. Verify pi is provisioned (managed install present); if not, direct the
   user to `aivm vm provision pi` (its installer already carries the TODO
   for the plugin step).
6. `pi install npm:pi-provider-litellm` (global; idempotent).
7. Merge `~/.pi/agent/auth.json` (keep 0600): `litellm = {type: api_key,
   key: <master key>, env: {LITELLM_BASE_URL: http://<gw>:<port>}}`.
   Re-running after `rotate_master_key` is what resyncs the credential.
8. Merge `~/.pi/agent/settings.json`: `defaultProvider: "litellm"` and
   `defaultModel: <id>` — JSON-merge only; never rewrite unrelated keys,
   and skip the merge when values already match (idempotence).
9. End-to-end verification: one-shot non-interactive `pi` print of a trivial
   completion through the gateway. Report success/failure with the
   concrete commands used (auditability: the script logs what it ran).

**Shape**: ~150 lines of Python (aivm and infer-stack are both importable
on the host; prefer their Python APIs over parsing their output) or a shell
script if importability turns out to be an assumption. Secrets: write files
directly, log only redacted forms.

## HTTPS: options and recommendation

- **Option A — keep plain HTTP (status quo).** Inside the aivm sandbox this
  is defensible: the firewall endpoint is scoped to one VM network, every
  other private-range destination is blocked, and the traffic is between the
  user's own VM and their own host. Residual risk is only the *host-side*
  exposure of the 0.0.0.0 publish to whatever LAN the host sits on. Cheap
  mitigation without TLS: publish the port bound to the bridge IP instead of
  0.0.0.0 (compose `ports: ["10.77.0.1:14042:4000"]` — Docker supports
  `IP:port:port`), so the gateway is reachable only from VM networks.
- **Option B — self-signed TLS terminated at the stack's nginx.** The
  optional reverse proxy (443) generates a self-signed cert (SAN = the
  bridge IP(s)), terminates TLS, and proxies to `litellm:4000`. The guest
  trusts the cert via `NODE_EXTRA_CA_CERTS=<ca.pem>` on pi's Node runtime.
  **Unverified**: whether pi's Node 22 build honors `NODE_EXTRA_CA_CERTS`
  for the `fetch` used by the provider extension (undici historically lagged
  on this). Alternatively request a `caCert` field upstream in
  pi-provider-litellm.
- **Option C — native TLS in the LiteLLM container.** Check whether the
  pinned image passes uvicorn `ssl_certfile`/`ssl_keyfile`; if it does, this
  is simpler than B (no proxy hop) but gives the least control over cert
  lifecycle.

**Recommendation**: ship the bridge with Option A (+ the interface-scoped
bind if easy) and treat B as a follow-up once the Node trust-store behavior
is confirmed with a 10-minute experiment on one VM. Do not block the bridge
on TLS: the aivm firewall already does the actual confinement.

## Open questions / verification checklist

- Exact non-interactive form of `aivm ssh <vm> <cmd>` (does it need a TTY?
  does `--yes` cover the prompts?) — the script may fall back to plain
  `ssh` using aivm's managed identity.
- Fix the aivm pi-installer TODO: `pi install -l …` → `pi install …`
  (global, not project).
- `NODE_EXTRA_CA_CERTS` behavior with pi's bundled Node 22 (only for B).
- Docker `IP:port:port` publish on the target host (Option A mitigation).
- Multi-VM on one network: one endpoint entry covers *all* VMs on that
  bridge (the entry is network-scoped, not VM-scoped) — say so in output.
- KubeAI backend is out of scope for the bridge (no compose publish;
  ingress is a different story).
- What the script does when the user has *other* pi providers configured
  (default: only touch `default*` when `--make-default` is passed? decide
  during implementation — the observed live setup did set them).

## Confident / at risk

- **Confident**: the endpoint math (`gateway_ip:litellm_port`, one entry,
  pre-DNAT matching), the pi config shapes (verified against a working
  VM), and the aivm mutation/apply path (first-class APIs, live-reconciling
  update). The manual procedure demonstrably works today.
- **At risk**: non-interactive `aivm ssh` scripting, `NODE_EXTRA_CA_CERTS`
  (Option B), and the store write-back preserving every corner of a user's
  hand-edited machine store (mitigate by going through aivm's own
  parse/render and by `--dry-run` showing the exact store diff first).
