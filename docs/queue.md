# Work queue: backend parity

The executable order for [planning/backend-parity-roadmap.md](planning/backend-parity-roadmap.md),
limited to what can be done and verified without a GPU or a second physical
machine: a guest VM running k3s with CPU vLLM. Items that end in a hardware
check produce a script to hand over rather than a claim of done.

Work top to bottom. An item is done when its **Done when** holds and its
commit is pushed; mark it `[x]` with the date and commit. Update
[backend-parity.md](backend-parity.md) and the roadmap's status in the same
commit as the change they describe.

## Rules

- **Unforeseen work gets added here, not done silently.** When an item turns
  out to need something the plan did not foresee, add it as a new item in
  its place in the order, with a *Why* line naming what was found. A blocker
  goes directly above the item it blocks.
- **Duplicate authorities** found on the way: refactor if small, otherwise
  add to the roadmap's *Duplicate authorities* table as deferred. Never make
  one worse. A blocker is fixed whatever its size.
- **Verified on k3s, not on fakes.** Every item that changes KubeAI
  behaviour adds or extends a step in `dev/kubeai_e2e.sh` and passes it.
- **The queue is not done until item 9 passes.** Do not stop at the last
  feature item.

## Items

### 1. [x] P1a: KubeAI acquires go through admission

Done 2026-09-25, `d6a19c4`.

### 2. [x] P1b: delete the legacy acquire branch

Done 2026-09-26, `184289c`.

`MemoryBackend`, `NullBackend` and the test fakes (queue, lock, serialised
publication) get a trivial `residency` and `preview`. Then the
non-admission branches, `_render`'s one-shot `converge(desired)` fallback
and `_admission_mode()` go.

**Done when:** `_admission_mode` does not exist; the full suite passes;
queue-semantics tests still assert the same behaviour.

### 3. [x] P2: day-2 commands and the TUI through the backend

Done 2026-09-26, `c2d393b`.

`instances()` and `stream_logs(target, *, follow, tail)` on both backends;
`ps`, `logs`, `stack up` / `stack down` use them, the raw compose form
moves under `stack compose …`; the TUI's docker pane, log follower and
Up / Down go through the backend; `measure` works on KubeAI if vLLM's
memory line reaches the pod log.

**Done when:** on k3s, `infer-stack ps` and `infer-stack logs <alias>` work
with the same output shape as on Compose, and the TUI follows a Model's log
and lists its pod (checked in a real terminal, not only in tests).

### 4. [x] P3: gateway feature parity

Done 2026-09-26, `7a322c1`.

`routes` and `secrets rotate` resolve the backend's gateway instead of
checking its kind; the KubeAI gateway project honours `ui`,
`reverse_proxy` and `dynamic_routing`. Move the KubeAI gateway's approval
into the admission preview (deferred duplicate).

**Done when:** `routes list` works on KubeAI; Open WebUI answers in front of
the cluster; an e2e step acquires the same model `--dedicated` twice under
dynamic routing and gets two Models and two routes.

### 5. [x] P6: one test surface

Done 2026-09-26, `1a4e4b2`.

Parametrize the controller's acquire scenarios over the Memory,
fake-Compose and fake-KubeAI backends; `tests/test_parity.py` runs each
*same* row of the parity matrix on both fakes.

**Done when:** every *same* row has a parity test, and the CI suite runs it.

### 6. [x] P4: placement from node labels (verified with faked labels)

Done 2026-09-26, `94c263d`; the GPU run is `dev/handover/p4_gpu_labels.sh`.
*Why reordered:* two GPU sizes need two nodes (a node has one
`nvidia.com/gpu.memory` label), so item 7's simulated second node
(`dev/k3s_agent_container.sh`) was built first, here.

`catalog suggest` on KubeAI proposes `resourceProfiles` from
`nvidia.com/gpu.product` / `.memory` node labels; `min_vram_gib` picks the
smallest fitting profile when an endpoint names none.

`measure` on KubeAI (moved here from P2): it already runs, but CPU vLLM prints
no GPU memory-profiling lines, and `--record` writes Compose's measurements
overlay, which KubeAI does not read. Decide where a cluster measurement is
recorded, and verify `measure` in the GPU handover.

**Done when:** on k3s with hand-set labels for two GPU sizes (CPU-backed
profiles), a catalog with `min_vram_gib` and no `resource_profile` lands on
the right profile; `dev/handover/p4_gpu_labels.sh` exists for one run on a
real GPU node, and runs `measure` there.

### 7. [x] P5: in-cluster gateway and a second node

Done 2026-09-26, `49e70dd`; the two-machine run is `dev/handover/p5_two_hosts.sh`.

Render the gateway as a Deployment + Service behind an ingress; `secrets
rotate` becomes a Secret update and a rollout; `doctor` checks the ingress.
Try a second k3s agent in a Docker container on the guest; write the
"add a workstation" runbook into `kubeai-backend.md`.

**Done when:** on one node, a card reaches a Model through the in-cluster
gateway and `secrets rotate` works. If the simulated second node fits in the
VM, a Model pinned to it by node selector is served through the same
gateway. `dev/handover/p5_two_hosts.sh` exists for one run across two real
machines.

### 8. [x] The README's Compose sections, and the other stale docs

Done 2026-09-26, `e1c68f3`.

About 33 references to verbs that no longer exist (`setup`, `up -d`,
`switch`, `describe-profile`, `smoke-test`, `wait-ready`, `diagnose`), and
top-level `restart` / `stop` / `start` / `pull` (they live under `stack`).
*Why widened (2026-09-26):* the same verbs appear in
`docs/persistent-caches-and-warm-restarts.md` and
`docs/stack-graph-profiles.md`.

**Done when:** every command in the README and `docs/` runs as written
against the current CLI, and `grep` finds none of those verbs.

### 8a. [x] Decide: rewrite or delete the pre-leasing recipes

*Why added (2026-09-26):* `docs/demos/`, `recipies/` and `examples/` (about
2,000 lines) were hardware recipes for the removed profile CLI.

*Decided (2026-09-26): keep the tuning, delete the pages.* The only content
the current docs lacked was the 4 x 96 GB vLLM tuning. It is now three gated
variants in the suggestion pool, which `catalog suggest` offers on a host
with four GPUs of 80 GiB or more: `qwen3.5-122b-a10b-tp4-128k`,
`qwen3.5-122b-a10b-fp8-tp4-262k`, `qwen3.6-35b-a3b-tp2-262k` (TP, context,
batch limits, `--language-model-only`, the qwen3 reasoning parser). They are
marked as not re-run since leasing. The rest was covered already: the
completions-only Pythia recipes by the README's protocol modes, the TLS/LDAP
example by the README's statement that those settings are gone, and the
tutorials by the manual. The pages are deleted; git history keeps them.
Test: `test_a_4x96_host_gets_the_former_recipes_as_variants`.

### 8b. [x] Decide: `/dev/shm` for vLLM containers

*Why added (2026-09-26):* the renderer set neither `ipc: host` nor
`shm_size`, so a vLLM container got Docker's 64 MiB `/dev/shm`.

*Decided (2026-09-26): opt-in, with a nudge.* `runtime.shm_size` (e.g.
`16g`) renders `shm_size` on the Compose service; unset, the service is
byte-identical to before, so no running engine is recreated by the upgrade.
A tensor- or pipeline-parallel engine without it logs one warning naming the
key, and `catalog suggest` sets `16g` on the multi-GPU entries it adds.
KubeAI needs nothing: its vLLM pods mount a memory-backed `/dev/shm`
(checked on the dev cluster, KubeAI v0.23.4). The TP=2 failure itself is
still unobserved here (no GPUs); the key is how an operator fixes it when
seen. Tests: `tests/test_shm_size.py`.

### 9. [x] UX audit loop: do not stop without a passing audit

Polishing the UX can take hours; that is expected, and this item is the
reason the queue exists. Loop:

1. Audit (below) on **both** backends, from a fresh data root.
2. Write every finding as a sub-item here, with the exact command and
   output.
3. Fix them, one commit each, tests added where the behaviour is testable.
4. Repeat from 1. The audit passes when a full pass finds nothing new.

The audit covers:

- **First run.** Following the README and `kubeai-backend.md` literally
  from nothing reaches a working request on each backend.
- **Every verb** (`acquire`, `release`, `wait`, `evict`, `gc`, `clean`,
  `renew`, `run`, `test`, `status`, `leases`, `ps`, `logs`, `routes`,
  `doctor`, `config`, `catalog`, `secrets`, `stack`): `--help` is accurate
  and its examples run; output has the same shape on both backends.
- **Every failure.** Each error names the cause and the next command to run;
  none mentions Docker or Compose on KubeAI, or `kubectl` on Compose; no
  traceback reaches the user for an expected condition (port in use, no
  cluster, unrenderable endpoint, refused approval).
- **The TUI**, in a real terminal at 80x24 and wide: every pane, tab, key
  and button works on both backends; every action logs its CLI equivalent;
  nothing is empty without saying why.
- **Consistency.** The same thing has the same name in the CLI, the TUI and
  the docs.

Seed findings, already known:

- [x] `r` does not reload catalogs, and edits to a catalog file are not
      picked up until restart. Fixed: each refresh rereads a changed catalog
      (a broken save is reported once), and `r` rereads it always. Checked
      live: `catalog endpoint add` in another shell shows in the TUI.
- [x] `infer-stack logs` should accept a container name, a prefixed name or
      an endpoint name. Done with P2: an instance name, a container id
      prefix, a deployment id or an endpoint alias.
- [x] `logs --no-color` silently kept color: kwconf reads a leading `no-` as
      negation, so a flag *named* `no_color` never saw it. Now `color`, whose
      `--no-color` works. Found by the README rewrite 2026-09-26.
- [x] `infer-stack logs -f qwen` followed every instance: a kwconf flag took
      the next word as its value. Fixed for every command (P2).
- [x] `status` shows STALE during an apply instead of "apply in progress".
      Fixed: `pending` while the publication marker says the change is not
      applied yet; STALE only when nothing is pending.
- [x] Open WebUI writes its data directory as root: removing a data root
      (the e2e's cleanup, or an operator's `rm -rf`) fails with permission
      denied. Found by `dev/kubeai_e2e.sh` 2026-09-26; fixed: it runs as the
      directory's owner (a directory root already wrote keeps root, and the
      log says how to `chown` it). Engine caches (vLLM as root) are the same
      shape; not yet looked at.
- [x] A refused acquire always ends with "free a GPU first — …", also when
      the reason is a render refusal (a served-name collision) or the
      backend is kubeai, where no GPU is ours to free. Found 2026-09-26;
      fixed: `PlacementError.capacity` says whether room would have helped.

Audit pass 1 (2026-09-26): every command's `--help` collected
(`infer-stack help tree`, 65 leaves) and every example in them run on a
dry-run root.

- [x] Five one-line summaries ended mid-sentence in `help tree` (`measure`,
      `routes prune`, `status`, `tui`, `wait`); `leases` said "deployment
      deployments".
- [x] `--litellm`, `--ui` and `--yes` said "(compose backend)", and
      `acquire`, `apply`, `--apply` and `render` described only a compose
      project; `render` on kubeai printed "(backend has no on-disk project)".
      `apply`'s help still called `stack up` the raw hatch.
- [x] Every `.env` write printed `Write .env to …` on stdout, into `--json`
      output too.
- [x] The render step logged "(not applied; `infer-stack apply` …)" in
      every acquire, right before the apply it said had not happened.
- [x] A runtime refusal (the gateway's port taken, a daemon down) ended in
      a Python traceback of `CalledProcessError`. Now the command, docker's
      own last lines, and a hint (a port in use names the port); docker's
      stderr is still shown live, through a pipe that keeps its tail.
- [x] `evict <unknown>` printed "no idle deployment for" then "nothing to
      evict"; it now says whether each name is held by a lease or unknown.
      `test` on a stopped gateway printed a urllib3 dump; now "nothing is
      listening there". `logs` said "running: nothing".
- [x] On kubeai with no cluster, `acquire` refused with "the runtime did
      not answer" and no cause; `ps` and `leases` printed kubectl's klog
      retries (hundreds of characters). Now kubectl's own last sentence
      ("The connection to the server … was refused"), in the refusal too,
      with `infer-stack doctor` as the next step.
- [x] TUI at 80x24: opening the runtime pane took every row (the lease and
      deployment tables vanished), and each log line spent ~20 of its ~33
      columns on the instance-name prefix. The pane is now capped near half
      the screen (three log lines at 24 rows), pane descriptions hide below
      32 rows, and one followed instance gets no prefix.
- [x] The TUI's API tab showed the gateway's master key in clear text in
      its example curl. The pane now reads it at run time
      (`$(infer-stack env LITELLM_MASTER_KEY)`); Copy curl copies the
      literal key. The UI tab said "docker observe interval".
- [x] TUI: a refused runtime command showed as a `CalledProcessError`
      repr full of paths (same fix as the CLI's), and colored engine output
      (vLLM's `(APIServer pid=1)`) was garbled in the logs pane. Checked on
      kubeai at 80x24: acquire, the pod's log, release-all, from the TUI.
- [x] First run, README literally, on a host with no GPU: `catalog suggest`
      found nothing and offered only simulated hardware (endpoints that
      cannot run here), so step 3 had nothing to acquire; `config init`
      pointed at `catalog init`, not the README's next step. Now `catalog
      suggest --simulator` adds a simulator endpoint (named in suggest's
      message when there is no GPU, and in the README), and `config init`
      points at `suggest`. Checked: init, suggest --simulator --apply,
      acquire, test, release, from nothing.
- [x] Consistency: the TUI's "Clean up" (`x`) runs `gc --forget`, which
      only forgets finished rows, while `infer-stack clean` releases and
      tears everything down. It is "Clear finished" now. The KubeAI setup
      example's resource profile had no GPU request or runtime class,
      which the README warns makes a pod start without CUDA.

Audit pass 2 (2026-09-26): the day-2 commands swept on both backends with a
live model each (`leases status ps env doctor clean gc renew routes wait
test logs`), plus pass 1 again.

- [x] `infer-stack test` failed on kubeai with HTTP 401, and `env
      LITELLM_MASTER_KEY` read nothing there (and so did the TUI's curl):
      both read a hard-coded compose `.env`. They now ask the configured
      backend's gateway for its `.env` and base URL (the in-cluster
      gateway's NodePort included).
- [x] `leases` showed a lease's TTL as `ttl=@1790450980`; now `ttl=1h59m`
      (JSON keeps the timestamp). `clean`'s dry run said `gpus=[]` where
      `leases` says `cpu`, and converge logged "on GPU(s) (cpu)".
- [x] `logs` into a pipe or file kept the engines' color codes; with color
      off they are stripped.

Audit pass 3 (2026-09-26): `dev/ux_audit.sh compose` and `kubeai` (help,
mistakes, the day-2 sweep; 0 flags on both), and the reports read by eye.

- [x] Every kubeai acquire logged "Converging 0 deployment(s): (none)" right
      after "Converging 1 deployment(s) onto kubeai": the gateway-only
      project narrating itself. It says "Converging the gateway project".
- [x] `doctor` on kubeai called KubeAI's own API "gateway", beside
      infer-stack's LiteLLM gateway; it is "KubeAI's API" now, here and in
      the docs.

Audit pass 4 (2026-09-26): `dev/ux_audit.sh` on both backends (0 flags;
the two reports differ only in backend facts), the reports read by eye.

- [x] `doctor` on kubeai did not check Docker, though the gateway on this
      host is a Compose project: a stopped daemon passed preflight and
      failed the first acquire. The gateway project's checks (Docker,
      compose, its images; no engine image or GPU check) now follow
      KubeAI's.

Audit pass 5 (2026-09-26): `dev/ux_audit.sh` on both backends.

- [x] `ps` (and the TUI's Instances table) printed the runtime's UTC start
      time with no zone, 4 hours off every other time on screen here. Now
      local time; `ps --json` keeps the runtime's UTC stamp.
- [x] The audit script itself: without `infer-stack` on PATH every command
      returned 127 and the pass reported 0 flags. It now refuses to start,
      and flags a timeout or a command that did not run. Doctor's `gateway:`
      lines on kubeai no longer trip the Compose-wording check: that gateway
      is a Compose project and needs it.

Pass 5, the TUI by eye (compose, 80x24 and 200x50):

- [x] At 200 columns the catalog sidebar stayed 38 wide and cut its gpu
      column to "aut" beside an empty 160-column table. It now follows the
      terminal (38 up to 64 columns) until resized by hand.
- [x] API, UI and Settings were reachable only by mouse or by tabbing into
      the tab bar; the command palette found none of them. Keys 1-5 and
      "Go to <tab>" palette entries now reach every top tab.
- [x] The API tab showed `http://localhost:14042/v1` while `env` and `test`
      say `127.0.0.1`: the TUI derived the gateway and Open WebUI URLs from
      the backend's ports itself, which is wrong for the in-cluster gateway
      (a node's NodePort) and found no Open WebUI on kubeai. A duplicate
      authority; now `Gateway.urls()` is the one derivation, and the env
      file, `env`, `test` and the TUI all read it through the front door.
- [x] The tab named "UI" held the TUI's own refresh settings, beside "open
      webui" everywhere else. Renamed "TUI settings", like "TUI log".

Pass 5, the TUI by eye on kubeai (80x24, then enlarged to 200x50):

- [x] Enlarging the terminal left the runtime pane at its small-screen
      height: one log line under a 26-row leases pane. The resize handler
      read the app's size before it updated; it now uses the event's.
      Instances, Control and Deployments read the same as on compose.

Pass 5, the first run from the README (no-GPU path, empty roots): config
init, `catalog suggest --simulator --apply`, acquire, test, the env exports
and a raw request all worked as written.

- [x] After `release --all` tore down the simulator (`reclaim: stop`),
      `leases` warned `NOT-RUNNING` and `clean` offered to tear it down
      again: both read every IDLE row as meant to run. The ledger keeps a
      released `stop` deployment IDLE by design. `Controller.keeps_up` is
      now the one reading of the reclaim policy (admission, health,
      `clean`); such a row is `reclaimed`, and `clean` says the stack is
      already clean ("no active leases, no models up").

Pass 5, consistency: product names (Open WebUI, LiteLLM, KubeAI, front
door, keep-warm) are spelled one way in user-facing text; the TUI's panes
and buttons use the CLI's words (leases, deployments, instances, apply,
down, `gc --forget` as "Clear finished").

- [x] `release`'s summary read "deployments idle/teardown per their reclaim
      policy"; now "its models stay warm or stop, per their reclaim policy".
- [x] Decided, no change: argparse accepts abbreviated flags, so
      `leases --all` fails as "ambiguous option: --all could match
      --allowed_gpus". kwconf's `__allow_abbrev__ = False` would give
      "unrecognized arguments" instead, and would break any script that
      abbreviates a flag today. Backwards compatibility wins; revisit with a
      deprecation warning if it bites.

Audit pass 6 (2026-09-26): `dev/ux_audit.sh` on both backends. Compose 0
flags; kubeai 2.

- [x] `release --env-file lease.env` and `renew --env-file ...` on a file the
      acquire never wrote (it failed and rolled back): a FileNotFoundError
      traceback, which is what a cleanup trap hits. Now "no env-file at ...;
      the acquire that writes it did not finish", and a file without a lease
      id says so.
- [x] The acquire failed because the dev cluster's node was under disk
      pressure and evicted KubeAI: ten audit data roots (0.9 GB each, Open
      WebUI's files and the weights) were still in /tmp. The audit now
      removes its data root and keeps the report, whose long outputs keep
      their tail (where a failure says why). `doctor` named the dead API.

Audit pass 7 (2026-09-26): `dev/ux_audit.sh` on both backends, 0 flags;
both reports read whole.

- [x] For the same GPU-less mock, `leases` and `clean` said `gpus=cpu` and
      `ps` said `-`. `ps` now uses the same words: `cpu` for a Docker engine
      with no GPU, `all` for one that requests every GPU, `-` for a pod
      (the cluster picks) or a front-door service.

Audit pass 8 (2026-09-26): **passed.** `dev/ux_audit.sh` on both backends
(0 flags, reports read whole, the only non-zero exits are the deliberate
mistakes); the README's no-GPU first run from empty roots; the TUI on both
backends at 80x24, enlarged to 200x50, every top tab by key. Nothing new.

### 10. [x] Handover

Summarize for the operator: what was verified here, and the two handover
scripts (items 6 and 7) with what each run proves.

**Verified here** (a VM with no GPU; k3s with a second node in a container):

- Both backends through one admission path, day-2 commands and the TUI
  through the backend's instances (items 1-5): the unit suite, the 21-row
  parity table on both backends, and `dev/kubeai_e2e.sh` with every phase
  (CPU vLLM: acquire, env-file client, gateway routes and Open WebUI,
  release, refusal, `min_vram_gib` sizing on fake GPU labels, dynamic
  routing, the gateway in the cluster, a Model on the second node).
- The UX audit (item 9): eight passes, the last with nothing new.

**Needs GPUs or a second machine** (run on the operator's hosts):

- `dev/handover/p4_gpu_labels.sh` on a GPU k3s node. Proves GPU Feature
  Discovery's real labels feed sizing: `catalog suggest --backend kubeai`
  proposes a profile per GPU product, an endpoint with only `min_vram_gib`
  lands on the right one and answers, and `measure --record` reads vLLM's
  memory profile from the pod log. Here those labels were faked.
- `dev/handover/p5_two_hosts.sh` on the first node once a second GPU
  workstation has joined. Proves pod-to-pod traffic across the real
  network: a Model on the second node's GPU answers through the in-cluster
  gateway, the NodePort answers on the second node's own address (one env
  file for cards on either machine), and `secrets rotate` holds with the
  Model served. Here the second node shared the host's network.

Both keep their own config and data roots and print PASS/FAIL per step.

**Decided without GPUs, worth confirming on them:** 8a's recipe variants
were not re-run since leasing, and 8b's `shm_size` fixes a TP failure not yet
observed here.

## Reopened 2026-09-26: outside review

Found by a two-part review of the finished campaign. Part 1: items 11-13.
Part 2: items 14-22, in the order the review recommends; item 11 also
takes part 2's finding on replicated startup diagnosis. The campaign's
completion rule changes with them: not "every *same* row has a test" but
"every shared lifecycle invariant holds when combined with the
backend-specific features that stress it" (item 13 and the poison cases in
each item). Item 9's UX audit is not the closing gate for these: a terminal
audit cannot show transaction safety under partial failure.

Rules for this section: correctness before cleanup; no broad CLI polish; a
poison test that fails on the old code with every fix; no new `hasattr`
probes; no second KubeAI lifecycle path; backwards compatibility for cards,
catalogs and CLI invocations that work today.

### 11. [x] Residency: replicas are not duplicates (reopened by the re-review, closed)

*Why added:* `Residency.ambiguous()` meant "more than one unit" and
`resident()` returned a unit only when there was exactly one. On Compose two
containers for one deployment is a conflict and must fail closed. On KubeAI
`runtime.min_replicas: 2` is a supported, documented setting, and its two
pods made the deployment `ambiguous` and not resident: releasing the last
lease of a keep-warm replicated Model pruned it, `leases` showed it
AMBIGUOUS, and `_pinned_endpoints` stopped protecting its definition. Part 2
adds: `diagnose_startup()` returns nothing unless there is exactly one unit,
so a replicated Model whose every pod crash-loops waits out the timeout
instead of failing fast.

**Do:** model residency at the deployment level (resident, conflicted,
units, warm units) and keep "one physical unit" only where GPU adoption on
Compose needs it. Audit every "the one resident" caller: keep-warm
admission, `_pinned_endpoints`, status, served-model rows, TUI/runtime
views, readiness, logs, startup diagnosis. Diagnosis of a replica set: fatal
when every unit is fatal, nothing while one is healthy. Compose keeps
failing closed on duplicate containers; "could not inspect" stays distinct
from "nothing running"; KubeAI gains no host-GPU accounting.

**Done when:** tests show a two-replica KubeAI Model is resident and not
conflicted; releasing its last keep-warm lease keeps it desired; `gc` keeps
it; `leases` and `status` show it running; its idle definition stays pinned;
a rollout snapshot (old pod leaving, new pod up) prunes nothing; every
replica crash-looping fails fast with the engine's error, one healthy
replica does not; two Compose containers for one deployment still fail
closed; the k3s e2e has a two-replica keep-warm phase and passes.

*Done 2026-09-26 (35be92c):* `Residency.replicated` (true for pods) with
`is_resident`, `is_conflicted`, `warm_units`, `resident_gpus`, and
`unique_unit` kept for GPU adoption only; every caller audited;
`diagnose_startup(..., replicated=)` fails fast only when every replica has
crashed. Unit and parity tests; the k3s e2e's two-replica keep-warm phase
passed (released and gc-ed: still desired, both replicas up, shown running).

### 12. [x] One backend protocol, the one the controller uses

*Why added:* `Controller` took a `Backend` (the old `realize/teardown/
observe` protocol) and cast it to `AdmissionBackend`, which inherited that
protocol and added Compose internals: `run` (a docker command; on KubeAI the
same name runs kubectl), `_load_sidecar()` whose schema the controller
read, `network`, `adopted`, probed with `hasattr`. KubeAI kept a `realize`
that was deliberately `pass`, and would be unsafe if it did anything.

**Do:** type `Controller` against the protocol it uses, with no cast. Move
`realize/teardown` to a small protocol behind `SimpleAdmission`. Replace the
Compose internals with semantic capabilities (stable network, legacy
adoption, orphan removal, placement notes) whose mechanics stay inside
Compose. No new `hasattr` probes.

**Done when:** no production backend has a method that exists only to
satisfy a protocol; the controller builds no `docker` command and reads no
sidecar; `ty` checks each backend against the protocol, and fails when one
drifts.

*Done 2026-09-26 (35be92c):* `ServingBackend` is the controller's protocol
(no cast); `Realizer` holds `realize/teardown` for `SimpleAdmission` only;
KubeAI's no-op methods are gone; `HostRuntime` (Compose:
`network_table/configure_network`, `subnet_clashes`, `rendered_services`,
`set_adopted`, `remove_containers`) behind `backend.host_runtime`;
`placement_notes()` replaces the sidecar read. `backend._conforms()` makes
`ty` check each backend against the protocol (verified: removing KubeAI's
`host_runtime` fails the check). `Backend` stays exported as an alias.

### 13. [x] Parity tests: cross-feature invariants

*Why added:* the parity suite's KubeAI fake ran one pod per Model, so
"replicas render" and "release keeps keep-warm" each passed while their
combination was broken. Rows of the matrix are not independent.

**Do:** a small poison suite, not a matrix: replica cardinality (multi-pod
keep-warm, overlapping pods in a rollout, replicated crash-loop), and the
cases items 14-19 add. The fake keeps "one Model, one pod" as its default
and gains a replica and a scheduler mode. Record the completion rule in the
roadmap.

**Done when:** the suite is in `tests/test_parity.py` (or beside it), each
case fails on the code before its fix, and the roadmap states the rule.

*Done 2026-09-26:* the KubeAI fake runs `minReplicas` pods, injects rollout
pods, and has a scheduler mode (nodes, pending pods with a message). Poison
cases: replicated keep-warm, rollout snapshot, replicated crash-loop,
Compose duplicates (`tests/test_parity.py`); scheduler selector, taint,
wrong-node victim, reclaimable pressure (`tests/test_parity.py`); partial
apply with renderer drift and the KubeAI gateway result
(`test_leasing_admission.py`, `test_leasing_kubeai.py`); rotation at both
boundaries (`test_leasing_secrets.py`); route removal failure on Compose and
KubeAI, and a replacement gap (`test_leasing_dynamic_routing.py`);
LIVE-over-IDLE coalescing (`test_leasing_controller.py`); seed conflicts
(`test_cli_leasing.py`). Each failed on the code before its fix, except
the positive control for reclaimable pressure, which passes on both.
The rule is in the roadmap's Principles.

### 14. [x] Scheduler-aware reclaim, not "Unschedulable means evict" (reopened by the re-review, closed)

*Why added:* KubeAI's readiness sets `needs_room` whenever a pod waits with
reason `Unschedulable`, and `wait_ready` answers by evicting the
longest-idle deployment every 30 s. `Unschedulable` also covers an
impossible node selector, an untolerated taint, affinity, a missing
profile, and pressure no idle Model relieves; and on several nodes an idle
Model on node A frees nothing for a pod that fits only node B. A permanently
impossible request can empty the warm set one Model at a time.

**Do:** replace the boolean with a structured signal, or a backend
operation that names reclaim candidates for a blocked deployment. Read the
scheduler's condition message, not only its reason; evict only for a
capacity shortage; only a resident idle victim in a compatible scheduling
domain (its node could host the blocked pod, by profile or node selector)
is a candidate. Keep: a leased deployment is never a victim; the policy
"leased demand outranks idle keep-warm" stays in the controller.

**Done when:** poison tests: an impossible node selector evicts nothing; an
untolerated taint evicts nothing; with two profiles on two nodes the idle
Model on the wrong node is not chosen; real reclaimable pressure evicts a
compatible idle Model and the leased one proceeds.

*Done 2026-09-26:* KubeAI's probe sets `needs_room` only when the
scheduler's message names an `Insufficient` resource (the message is now in
residency); `ServingBackend.reclaim_candidates(blocked, idle)` names idle
Models with a pod on a node the blocked pod could use (node selector,
taints; required affinity is not evaluated, so nothing is evicted) that
request the short resource. Compose returns none, the in-process backends
all. `_make_room` keeps the policy. Four parity poison tests; three fail on
the old code.

### 15. [x] Publication phases: the approved digest outlives a partial apply (reopened by the re-review, closed)

*Why added:* `_apply_pending()` clears the approved-render digest right
after `apply()` returns, before checking it returned `False`. Compose
returns `False` when the runtime changed but dynamic routes did not verify:
the publication stays pending, but the record of which render was approved
is gone, so a renderer that changes before the retry (an upgrade) applies
an unapproved render.

**Do:** keep the digest until the publication completes, unless an explicit
re-approval replaces it. Make the apply result say what happened (runtime
applied, routes verified, complete) instead of `True / False / raise`, and
only as far as the phases that differ today.

**Done when:** a test: preview approves D1, the first apply changes the
runtime and returns partial; the publication is pending with D1; the
renderer changes to D2; an ordinary retry refuses on the digest mismatch;
`infer-stack apply` re-approves and applies.

*Done 2026-09-26:* `apply()` returns an `ApplyResult` (`runtime`, `routes`,
`detail`; `None`/`True`/`False` still map onto it). The controller clears
the approved digest and the marker only on a complete result. Found on the
way: KubeAI's `apply` dropped its host gateway's result, so unverified
routes behind KubeAI cleared the marker; it now returns both. Tests: the
partial-apply/renderer-drift case (fails on the old controller: the digest
was gone) and the KubeAI propagation.

### 16. [x] Secret rotation is a transaction (reopened by the re-review, closed)

*Why added:* `rotate_gateway_key()` writes the new key into `.env`, then
publishes, and restores the old key only on `ConvergeAborted`. A strict
residency or render failure before apply leaves `.env` on the new key while
LiteLLM runs the old one, so clients are handed a key that does not work.
The tests' fake gateway reads `.env` per request, which hides it.

**Do:** restore the old key on any failure before apply begins; after apply
may have begun, keep a recoverable pending publication instead (do not
revert the file blindly). Uses item 15's phase information.

**Done when:** a gateway fake that captures its key when it (re)starts;
tests for a declined approval, a render failure before apply (both: old
file, old running key), and a failure after the gateway restarted (pending,
converges on retry).

*Done 2026-09-26:* the controller records when an apply begins; a rotation
whose publication fails before that restores the old key (any failure, not
only a decline), and after it keeps the new key with the publication
pending and interrupted. Tests use a gateway that accepts the key its
container started with: a residency failure before apply (fails on the old
code), a decline, and a failure after apply began that converges on
`infer-stack apply`.

### 17. [x] Dynamic routes never point at a torn-down upstream

*Why added:* Compose `apply()` removes departing engines, then reconciles
routes; KubeAI deletes stale Models, then applies the gateway. If route
removal fails, the publication is pending but LiteLLM still routes to a
dead upstream, which with several dedicated deployments behind one alias
fails some requests. Replacing a drifted route deletes before it adds, so a
failed add leaves a gap.

**Do:** order a removal as: remove and verify the routes that would dangle,
tear down the departing upstreams, converge the desired ones, add and
verify new routes. Make the replace-route gap an explicit, reported state.

**Done when:** failure injected into route deletion while releasing one of
two dedicated deployments for one alias leaves the departing upstream
running (the invariant, not just `publication_pending`), on Compose and on
KubeAI with the host gateway.

*Done 2026-09-26:* an apply's first phase, `retire_routes()`, deletes and
verifies the routes the render drops (only with dynamic routing and a
running gateway); if that fails the apply returns "runtime not reached" and
tears nothing down. KubeAI retires before pruning stale Models; its
in-cluster gateway has static routes to one upstream and retires nothing.
A replacement whose add fails after its delete is logged by route. Tests on
Compose and on KubeAI with a host gateway (both fail on the old code): the
departing engine or Model keeps running while its route stays, and goes
once the gateway recovers.

### 18. [x] Coalescing prefers a LIVE deployment over reviving an IDLE one

*Why added (older than this campaign):* `plan_acquire()` takes the first
compatible deployment by creation time among LIVE and IDLE, so an older
IDLE deployment is revived even when a LIVE one already satisfies the
request. 8k, then 32k, then 4k on one GPU: the 4k request revives the idle
8k deployment and queues, although the live 32k one serves it now.

**Do:** rank adequate candidates LIVE, then resident IDLE, then other IDLE,
then create; the ledger stays the authority on eligibility and takes a
residency hint from the controller.

**Done when:** the 8k/32k/4k sequence on a one-slot backend coalesces the
4k request onto the live 32k deployment: nothing revived, no placement, no
queue, no runtime change.

*Done 2026-09-26:* `Ledger.plan_acquire(requests, resident=)` ranks
adequate candidates LIVE, then created in this plan, then resident IDLE,
then other IDLE, creation order within each; the controller passes
`residency.is_resident`. Tests: the 8k/32k/4k sequence coalesces onto the
live 32k deployment with nothing revived or started, and a resident idle
deployment beats an older one that is gone (both fail on the old code).

### 19. [x] `routes seed` fails closed on a redefinition; one route API

*Why added:* the registry merge is documented as additive, but a same-name
row with a different definition wins with a warning, and `routes seed`
skips confirmation because "seeding is additive". That silently redirects a
public alias. `routes prune` in the CLI reaches into
`controller._admission_view`, `gateway._load_route_registry`,
`backend._converge_lock`, `_atomic_write` and `gateway._registry_file`.

**Do:** a public route-registry operation that plans a seed or prune
(added, unchanged, updated, conflicted) and commits it under the normal
locks; conflicts refuse by default, an explicit override shows the change
and asks. The CLI presents and confirms only.

**Done when:** tests: seed A; seed identical A is a no-op; seed a
conflicting A refuses and leaves the registry unchanged; the override
reports A updated. No private names from the CLI.

*Done 2026-09-26:* `leasing/routes.py` plans a seed (added, unchanged,
conflicted) or a prune (dropped); `Controller.plan_route_seed/prune` and
`commit_route_seed/prune` recheck under the lock and publish through
`publish_change`; the gateway exposes `route_registry()`,
`route_entries()`, `replace_route_entries()`. `routes seed` refuses a
redefinition and leaves the registry unchanged unless `--replace` (which
lists each change and asks on a terminal); its JSON says added, unchanged,
updated, conflicted. The CLI uses no private name. The render's own merge
keeps "incoming wins" (an edited catalog updates its route) and its
docstring now says so. Also fixed: under `--json` the conflict list went to
stdout and broke the JSON. Test: seed, identical seed, refused conflict,
`--replace` (fails on the old code).

### 20. [x] Preview does not write secrets

*Why added:* rendering calls `master_key()`, `db_password()`,
`webui_secret()`, which generate and write missing secrets, so a preview
that is then refused has still changed `.env`.

**Do:** initialize secrets before a pure render, or commit generated
secrets with the publication. If kept, document the narrower guarantee.

**Done when:** a refused acquire on a fresh data root leaves no `.env`, or
the docs say exactly what a preview may write.

*Done 2026-09-26:* the gateway's secrets go through one `_managed_secret`;
inside `staging_secrets()` (Compose `preview`, the in-cluster gateway's
`preview`) a missing secret is generated in memory and not written, and the
next writing call persists that same value. Fingerprints read
`managed_env()` (file plus staged), so the commit's render matches the
preview's digest. Test: a refused acquire on a fresh root leaves no key in
`.env` (fails on the old code: it wrote the master key and the Open WebUI
secret), and the next admitted acquire needs no second approval.

### 21. [x] Controller decomposition, where the authorities now show it (reopened by the re-review, closed)

*Why added:* the controller holds admission, residency interpretation,
placement, publication markers, recovery snapshots, network migration,
secret rotation, adoption, orphans, reclaim, leases and rollback. Split
only along state machines (publication/recovery, admission/reclaim,
profile snapshots, backend capabilities), and only after 14-20.

**Done when:** each extracted part needs no private cross-layer call, or
the item records why nothing was extracted.

*Decided 2026-09-26: no class extracted; the cross-layer calls removed
instead.* After items 11-20 the controller calls no private backend member
and builds no runtime command; the CLI's two private reaches
(`controller._global_lock`, `_invocation_profile`) are now
`publication_lock()` and `invocation_profile()`; the publication marker's
approval is written through `Ledger` (`clear_approved_digest`,
`publish_profile`, `migrate_network`), not `ledger.store`. A split would not
remove a call today: the publication coordinator needs `_render` and
`_admission_view`, and admission needs the profile snapshot, so each part
would call back into the others. The boundaries the review names
(publication/recovery, admission/reclaim, profile snapshots, backend
capabilities) are where to cut when one of them grows. Left as it was: 15
optional-hook `getattr(self.backend, ...)` reads that predate this section
(`use_profile`, `render_profile`, `settle_snapshot`, the `last_*`
attributes, ...). They could become protocol members with in-process
defaults, as `front_door` and `route_rows` did here.

### 22. [x] Docs and full verification

*Why added:* the parity matrix, roadmap and `known-limitations.md` describe
the semantics these items change; `known-limitations.md` still says KubeAI
has no strict residency.

**Do:** update them from the resulting semantics; set the roadmap status to
"features complete, review hardening" until this section is done; run the
full suite, `ty`, flake8, and the full k3s e2e.

**Done when:** those pass and the docs match the code.

*Done 2026-09-26:* the parity matrix, the roadmap (principle, authorities,
status) and `known-limitations.md` describe the new semantics; CHANGELOG
entry. Verified: the full suite, `ty` and flake8 pass; the full k3s e2e
(`E2E_SIZED=1 E2E_REMOTE_NODE=1`, dynamic routing and the in-cluster
gateway on by default, the new replica phase) passed from a frozen copy;
`dev/ux_audit.sh` reports 0 flags on both backends, and the only non-zero
exits are its deliberate mistakes.

Also observed while working on these: `test_an_edit_made_outside_the_tui_appears_on_the_next_refresh`
failed in two of five full-suite runs and never alone. It counted every
`_refuse` call after two refreshes, and a refresh's background worker can
refuse something unrelated meanwhile; it now counts catalog-reload
refusals only (the likely cause, not reproduced on demand).

## Re-review (2026-09-26): second-order cases

The re-review closed most of items 11-22 and reopened five, for states the
new abstractions exposed but did not carry through. Order: 21's settlement
bug first (a safety bug), then 16, 15, 11, 14, 23, then the rest of 21.

- **21, settlement.** `_wait_for_settled_runtime` reads
  `getattr(backend, 'settle_snapshot')`; KubeAI has none, but its host
  gateway is a Compose runtime whose Docker work can outlive a killed client.
  After a `BackendTimeout` in the gateway's apply, the next apply starts
  another Compose operation unsettled. **Do:** a required
  `settle_snapshot() -> object | None`; KubeAI delegates to a host gateway.
  **Done when:** a KubeAI backend whose host gateway times out mid-apply
  makes the next apply settle first (or raise `RuntimeUnsettled`).
- **16.** An apply that returns `ApplyResult(runtime=False)` cleanly leaves
  `.env` on the new key and the gateway on the old one: the boundary was
  "apply was called", not "the runtime changed". **Done when:** a backend
  returning `runtime=False` without recreating the gateway leaves no key in
  `.env` the gateway rejects.
- **15.** `infer-stack apply` accepting D2 over an approved D1 does not
  record D2, so a partial D2 apply leaves D1 approved and every retry asks
  again. **Done when:** after an explicit D2 apply that is partial, the
  marker holds D2 and an ordinary retry of D2 proceeds; a D3 still refuses.
- **11, diagnosis.** A replica set is judged from the concatenated log, so
  one fatal and one transient replica read as fatal. **Done when:** replicas
  are classified one by one; fatal only when each is fatal; one transient
  replica with retries left suppresses it.
- **11, health.** `status` equated residency with serving: every replica in
  CrashLoopBackOff is "resident" (restarting is warm) and showed `up`.
  **Done when:** one deployment health summary (up / starting / restarting /
  conflicted) feeds `status` and `observe_state`; all replicas looping is not
  up, one healthy is up, one starting plus one looping is not up.
- **14.** `eligible_nodes()` models the default scheduler; a pod with
  `schedulerName` (the catalog's `scheduler_name`) or required pod
  (anti-)affinity may be filtered differently. **Done when:** those evict
  nothing (test through `scheduler_name`).
- **23 (new), route seed race.** A conflict found under the lock raises after
  `publish_change` set the marker, so a clean refusal leaves a publication
  pending. **Done when:** a race-time conflict leaves the marker as it was.
  (The stale `--replace` display is noted, not fixed.)
- **21, optional hooks.** Classified by the re-review: the `last_*`
  attributes and a new `last_planned_digest` and `allocates_gpus` are
  required state read directly; `plan_on_idle_host` and `validate_requests`
  become common operations with in-process defaults; `last_displaced` /
  `last_degraded` go (`placement_notes()` is the authority);
  `render_profile` / `use_profile` become a nullable recovery-profile
  capability; `rotate_master_key` / `restore_env` / `litellm` belong on a
  typed `front_door() -> FrontDoor | None`. Rule: a `getattr` default may
  serve display, never a correctness decision.

### 23. [x] A race-time `routes seed` conflict leaves no marker

*Done 2026-09-26 (30a62bb, 122f1b4):* `publish_change(..., preflight=)` runs
a check under the lock before the marker; route seed rechecks there. Test
injects the other process's write when the lock is taken (the old code left
the marker). The stale `--replace` display is not fixed.

### Re-review closure (2026-09-26)

Each with a test that fails on the code before it:

- **21, settlement (4b8783b):** `settle_snapshot()` is required; KubeAI
  delegates to its host gateway, `None` for an in-cluster one and in-process.
- **16 (4476573):** an apply returning `runtime=False` restores the old key
  and refuses with the apply's reason (`ReconcileResult.apply_detail`).
- **15 (768d924):** `infer-stack apply` over a changed render records it as
  approved (in place, same marker version) before applying.
- **11 (b8b3542):** replicas classified from their own logs (`_pod_logs`),
  fatal only when each is; `deployment_health()` (up / starting / restarting
  / conflicted) is what `status` shows; `observe_state` takes only
  `restarting` from it and `leases` flags RESTARTING (item 26).
- **14 (55c1f96):** `eligible_nodes()` returns unknown (evict nothing) for a
  non-default `schedulerName`, required pod (anti-)affinity, or a hard
  topology spread; tested through a profile's `scheduler_name`.
- **21, hooks (2f9adc4):** as classified above. No `getattr(self.backend,
  ...)` remains in the controller; `ty` checks `FrontDoor` and
  `RecoveryProfile` conformance, and removing Compose's `allocates_gpus`
  fails it.

## Final closure pass (2026-09-26)

A third review found one cross-feature blocker and three small items. All
done; verified by the suite, `ty`, flake8, both UX audits and the full k3s
e2e (which also exposed, and 5337815 fixed, a race in `secrets rotate`'s
old-key check behind an in-cluster rollout).

### 24. [x] `secrets rotate` under a running dynamic-routing gateway

*Why added:* rotation writes K1 to `.env` before publishing; the dynamic
apply first retires routes on the still-running gateway, whose admin API
accepts only K0, but authenticates with `.env`'s K1. Retirement cannot
list routes, the apply returns `runtime=False`, and rotation restores K0
and refuses: with dynamic routing on (Compose, or KubeAI's host gateway)
rotation can never complete. The e2e rotated only after switching dynamic
routing off, and the unit fake ran without it.

**Do:** before the gateway is recreated, admin calls use the credential the
running gateway holds; after, the desired one. Keep route retirement.
**Done when:** with dynamic routing and a gateway started on K0, rotation
succeeds: `.env` is K1, the gateway accepts K1 and rejects K0, routes
verify, no marker; on Compose and KubeAI's host gateway; and the e2e rotates
once while the host dynamic gateway is the front door.

*Done 2026-09-26 (45e0e1c):* route retirement (the phase before the gateway
is recreated) authenticates with the key the running LiteLLM container
holds, read from its environment by `docker inspect`, through
`Gateway.live_credential()`; everything after recreation uses the `.env`
key. No new durable state, so a crash mid-rotation recovers the same way.
Unit tests on Compose and on KubeAI with a host gateway use a fake whose
admin API answers only the key its container started with (both fail on
the old code); the e2e rotates inside the dynamic-routing phase with one
dedicated lease live, and the alias answers with the new key.

### 25. [x] `SimpleAdmission` keeps `last_planned_digest`; the guard fails closed

*Why added:* the in-process backends never set `last_planned_digest`, and
the approval guard skips when it is empty, so an approved digest was not
checked there. **Done when:** converge sets it from the same digest as
preview, and an approved digest with no rendered digest refuses.

*Done 2026-09-26 (3abb2ff):* `SimpleAdmission._digest(plan, rendered)` for
both; the guard is `approved and rendered != approved`. Six test backends
that override `converge` had not kept the contract; they do now.

### 26. [x] Say what `leases` health reports

*Why added:* `observe_state` takes only `restarting` from
`deployment_health`, while the docs say the summary "feeds" it. **Done
when:** code and docs agree.

*Done 2026-09-26:* the behaviour stays: `leases` flags a crash loop, the
state that needs an operator; a deployment still starting reads `running`
there, and `status` shows `starting`. The docstrings of `observe_state` and
`deployment_health` and the item 11 note now say so.

### 27. [x] Record the `routes seed --replace` confirmation race as deferred

**Done when:** it is in `known-limitations.md`.

*Done 2026-09-26:* recorded there as current, deferred.


## Still open outside the numbered items (logged 2026-09-27)

Everything known to be open that is not a numbered item above or in
campaign 2. None blocks campaign 2.

- [ ] **GPU handover runs (operator).** `dev/handover/p4_gpu_labels.sh` on a
  GPU k3s node (GFD labels feed sizing, `measure --record`) and
  `dev/handover/p5_two_hosts.sh` across two real machines (pod-to-pod over
  the real network, the NodePort on the second node, rotation with a Model
  served). Verified here only with fake labels and a node in a container.
- [ ] **Confirm 8a and 8b on GPUs.** The three 4 x 96 GB suggestion variants
  were not re-run since leasing; `runtime.shm_size` fixes a TP > 1 failure
  never observed here. Serve one variant and a TP=2 endpoint with and
  without `shm_size`.
- [x] **vLLM caches written as root.** The Open WebUI fix (seed item under 9)
  noted that engine caches under the data root are the same shape (vLLM runs
  as root, so an operator's `rm -rf` of a data root fails).
  *Done 2026-09-27 (VM part):* `infer-stack paths` now lists every state
  directory (its docstring promised the caches; it showed only the data
  root), marks one holding another user's files `foreign-owned`, and prints
  the `docker run --rm -v DIR:/d busybox chown -R uid:gid /d` that takes it
  back (verified: a root-owned model directory blocked `rm -rf`; after the
  printed command it did not). Running vLLM itself as the owner would change
  a GPU runtime this VM cannot verify; not done.
- [ ] **In-cluster gateway: deferred features (P5).** Dynamic routing and
  Open WebUI with `kubeai_gateway cluster` (they need Postgres and a UI in the
  cluster), and an Ingress (`kubeai_gateway_url` accepts one; none tested).
- [ ] **`routes seed --replace` compare-and-swap.** Deferred; see
  `known-limitations.md`. Carry the expected old row in the plan and ask
  again on a mismatch under the lock.
- [ ] **CLI capability probes that decide behaviour.** The controller has no
  optional-hook `getattr` left, but the CLI still branches on
  `getattr(backend, ...)` for `down`, `doctor`, `compose_project`,
  `rendered_file`/`compose_file`, `deployment_logs` and `access` (the last is
  campaign 2's cleanup D). By the re-review's rule a `getattr` default may
  serve display, never a decision: make the deciding ones protocol members
  or typed capabilities.
- [ ] **TUI catalog-reload flake.** The fix counts only catalog refusals; the
  cause (a background worker refusing meanwhile) was inferred, not
  reproduced. Reopen if it recurs.
- [ ] **Commit history before merge (maintainer's call).** `30a62bb` has one
  failing test, fixed by `122f1b4`; squash them if every commit must be green.
- [ ] **Guest VM state to remember.** `/etc/rancher/k3s/config.yaml` sets a
  1Gi eviction minimum reclaim for the dev cluster (undo: remove it, restart
  k3s); the second node is recreated by `dev/k3s_agent_container.sh` with the
  same setting.

---

# Campaign 2: external endpoints, and access above leasing

**Start only after items 11-27 are committed, green (suite, `ty`, flake8,
full k3s e2e, UX audits) and the docs coherent. Do not interleave.** A
separately reviewable campaign, requested 2026-09-26.

**Goal.** First-class catalog endpoints whose model server already exists
outside infer-stack, used as the occasion to remove conceptual duplication
in endpoint -> lease -> route -> access. User stories: (1) a workflow asks
for `qwen`; today infer-stack leases a local vLLM for it, tomorrow `qwen`
points at a running OpenAI-compatible server, and the workflow does not
change; (2) register a hardcoded OpenAI-compatible server as an endpoint
without a fake lease.

**Target data flow** (fewer transformations than today):
catalog definition -> resolved endpoint meaning -> (managed:
`EndpointRequest` -> ledger) / (external: target) -> semantic gateway
route(s) -> gateway publication -> access descriptor.

**Stop rule.** If the diff ends with more conceptual branches than today,
stop and reconsider the abstraction. Do not bolt `external` onto every
`if engine == ...`. If the code contradicts a conclusion below, stop and
document it instead of forcing it.

## Conclusions to preserve

- **Endpoint identity is not lease identity.** An external endpoint creates
  no `Lease`, `Deployment`, demand, GPU allocation, TTL or reclaim state.
  The ledger holds only what infer-stack owns.
- **"External" is not an engine.** No `engine: external`. Ownership is its
  own axis: `Endpoint(alias, protocol, target: ManagedTarget(vllm|ollama) |
  ExternalOpenAITarget(api_base, upstream model, api_key_env?))`. Existing
  managed YAML is unchanged. Additive YAML:
  `endpoints: {qwen-remote: {external: {api_base: ..., model: ...,
  api_key_env: ...}, protocol: chat}}`. An external endpoint rejects
  `runtime`, `placement`, `sharing`, `reclaim`, `host` (unless one proves a
  meaning) and needs no `models:` entry.
- **The catalog is the authoring authority.** Routes are derived
  publication state; `routes seed` stays for multi-runbook pre-seeding, but
  registering an external endpoint is a catalog operation, protected by the
  profile/catalog conflict machinery: identical definitions coexist,
  different ones conflict, a pinned managed endpoint cannot be silently
  turned external, and once it is unpinned the change is allowed.
- **Access through the front door only (first version).** One base URL,
  the alias as model name, mixed bundles work, upstream credentials stay
  behind the gateway. No implicit direct-to-external fallback. An external
  endpoint without a LiteLLM front door fails clearly ("external catalog
  targets currently require the front door").
- **Externally owned means not controlled.** No startup, restart,
  keep-warm, reclaim, placement for external targets. Publishing the route
  is enough for `access` (no generation probe); an optional reachability
  probe may report, never own. Views say `external`, not stopped / idle /
  unplaced.

## Work items (in order, small commits)

### 28. [x] Read, then write the design down first
Read catalog, profile, gateway, controller and access code together. Write
the data flow and invariants into a planning doc, and name the concepts
that become unnecessary (e.g. `UPSTREAM_ROUTE`, unreachable
`render_front_door` branches, lease-bound descriptors, route-derived
catalog equality). Gate for the rest.

*Added by the campaign-2 design review (2026-09-27): decide these in the
doc before item 29.*

1. **Published endpoint lifetime.** What keeps an external endpoint
   registered with zero leases; how it is explicitly removed; an unrelated
   runbook or a global-setting change must not silently unregister it.
   Today the recovery profile replaces its catalog set wholesale when
   quiescent, and the route registry is append-only: two authorities that
   external endpoints would make visibly disagree. Proposed: separate the
   published catalog set's lifetime from the recovery settings' (may stay in
   one persisted profile); quiescence permits changing frozen settings but
   does not prune unrelated published definitions; an explicit
   prune/unpublish removes them.
2. **Route ownership.** Catalog-derived routes (exactly from the published
   catalogs), manual/seeded routes (if still needed, owned explicitly), and
   dynamic deployment routes (runtime-derived) are three owners, not one
   alias -> row map. Decide whether catalog rows stay in the persistent
   registry at all; if the registry carries nothing the published catalogs
   do not, remove the duplicate instead of teaching it external targets.
   `routes seed FILE...` may become "merge these catalogs into the
   published union". `CatalogUnion` importing
   `_registry_incoming_from_catalog` is the symptom.
3. **One access transaction.** Mixed managed + external access is one
   plan, one preview, one approval digest, one commit (profile/catalog,
   lease, marker), one render/apply; not "acquire, then publish routes".
   `acquire()` stays the lease-specific public operation over the same
   transaction machinery.
4. **Credential lifecycle.** Host Compose gateway: the LiteLLM service's
   environment names each referenced variable (`KEY: ${KEY}`), so the
   existing fingerprint recreates it on a value change without the value in
   any route. In-cluster gateway: referenced variables go into its Secret
   (`envFrom`), the pod-template hash covers their values, never in a
   manifest or diff. `infer-stack env KEY=new`: choose publish-on-write or
   stage-and-say-so ("gateway uses it after `infer-stack apply`"), never
   silent. Reject reserved names (`LITELLM_MASTER_KEY`, salt, DB password,
   Open WebUI secret, `HF_TOKEN`) and invalid variable syntax as
   `api_key_env`.
5. **Stable route identity.** An external route's id is its logical owner
   (e.g. `external-route:<alias>`); api_base / model / credential name are
   its content, compared by the semantic route to decide replacement.
   Managed dedicated routes keep `deployment id + alias`.
6. **Front-door readiness.** External upstream readiness is not
   infer-stack's; its own gateway's is. `access` returns only after the
   front door answers with the expected master key (no generation to the
   upstream). Poison test: LiteLLM starts slowly.
7. **Failure semantics.** A managed member of a mixed bundle that fails and
   rolls back leaves the external definition and route published; a process
   that dies after publication leaves the route (a committed lease follows
   normal TTL/recovery; no access handle); removing an external endpoint
   from the catalog unpublishes it by the rule decided in 1.

*Done 2026-09-27:* [planning/external-endpoints.md](planning/external-endpoints.md)
decides all seven. The chosen rules: the profile's `catalogs` is the durable
published union (merged on every acquire/access; quiescence replaces
settings, not catalogs; `routes prune` unpublishes); routes are derived at
render, the registry becomes read-only legacy input; access is the acquire
transaction with a possibly empty managed set; credentials by reference
with `env` saying when an apply is needed; external route id =
`('external', alias)`; access waits for the front door's key.

### 29. [x] `ResolvedEndpoint`: endpoint meaning, not a lease request
One normalized object (alias, protocol, target) with a canonical semantic
key used for catalog-union conflicts, profile drift, and "did this endpoint
change". A managed target derives an `EndpointRequest`; an external one has
none. `CatalogUnion` / profile code stop importing gateway or Compose route
conversion (cleanup A). Transitional names may stay for compatibility.

*Review:* the managed target's key must cover everything behaviourally
relevant, resolved (model source/revision, quantization/dtype, engine,
upstream served name, launch/runtime structural knobs, capacity, meaningful
placement, default sharing, reclaim policy, protocol), never the raw
`models:` key: two catalogs naming one source differently stay equal. A
request-time `--dedicated` is not part of the key. API: make
`Catalog.resolve_endpoint(name)` return the `ResolvedEndpoint`, with
`resolved.to_request(sharing_override=...)` (or `resolve_requests`) for the
ledger; no parallel `resolve_endpoint_meaning` / `resolve_access_endpoint`.

*Done 2026-09-27:* `leasing/endpoints.py`: `ResolvedEndpoint(alias,
protocol, target)`, `ManagedTarget(request)`, `ExternalTarget(api_base,
model, api_key_env)`, `semantic_key()` from the resolved request.
`Catalog.resolve_endpoint` returns it; `resolve(names)` gives meanings,
`resolve_requests(names, sharing=)` ledger requests (`resolve_names` kept
as its alias). `CatalogUnion` compares semantic keys and no longer imports
`_registry_incoming_from_catalog` (cleanup A). A managed `ResolvedEndpoint`
still answers the `EndpointRequest` fields: eval_audit reads them off
`resolve_endpoint` (its test failures are the same with and without this).

### 30. [x] The external target in the catalog and its CLI
YAML `external:` block, validation of illegal mixtures, round-trip.
`catalog endpoint add` gains mutually exclusive external options (e.g.
`--external-api-base`, `--external-model`, `--external-api-key-env`), never
an `--engine` choice; reject `--engine vllm` + external, `--gpu`/`--reclaim`
+ external. `catalog show` makes ownership obvious.

*Done 2026-09-27:* `EndpointSpec.external`; `endpoints.external_errors`
(managed-only keys, URL, model, env-name syntax, reserved names);
`catalog endpoint add --external-*`; `resolve_requests` on an external
member raises "... does not require a lease; use `infer-stack access
NAME`". `catalog show` prints the `external:` block. Tests 1-2 and the CLI in
`tests/test_external_endpoints.py`.

### 31. [x] One terminology
Endpoint alias (what users request through the front door), upstream model
(what the target server expects as `model`), deployment id (one managed
realization). Stop calling `served_model_name` a "public name"; keep old
YAML spellings as compatibility aliases; all new code uses the three terms.

*Done 2026-09-27:* the three terms are defined once, in
`leasing/endpoints.py`'s docstring (the naming module and
`EndpointSpec.served_name` point at it); `--public-name`'s help no longer
calls the upstream model the public name. YAML `public_name`/`served_name`
unchanged. New code (endpoints, routes, access) uses alias / upstream model /
deployment id.

### 32. [x] One semantic `GatewayRoute`

*Done 2026-09-27.* `GatewayRoute` (`leasing/routes.py`) with one renderer,
`entry()`; `front_door_routes` derives the static table (registry < catalog
< deployment < upstream) or the dynamic set (one per managed id) for Compose,
KubeAI's host gateway and the in-cluster gateway alike. `render_front_door`
takes routes; its catalog-superset and legacy branches and
`UPSTREAM_ROUTE` are gone (older registry rows spelled `upstream` still
read). The registry stores only ad-hoc deployments' routes (decision 2,
revised: see the design doc). Backend protocol: `routes()` /
`catalog_routes()` replace `route_rows()` / `catalog_route_rows()`. `routes
seed` merges into the published union (refuses conflicts unless
`--replace`, never a pinned one), `routes prune` unpublishes, `routes list`
shows origins. External routes (item 33's static half, and the dynamic id)
came with it: they route to their own server on every backend.

*Step 1 done 2026-09-27 (decision 1):* `adopt_catalog_sources` merges the
invocation's catalogs into the published union on every acquire: a changed
definition replaces an unpinned published one, a pinned one refuses,
everything else stays; quiescence adopts settings only. The live-epoch path
used to drop every unpinned definition on any conflict; it now drops only the
conflicting names. ADR 0001, known-limitations and `profile.py` updated.
Alias, upstream adapter (OpenAI-compatible / Ollama), upstream model,
api_base, optional credential-env name, optional managed dynamic route id.
Produced by Compose (managed service name), KubeAI (cluster gateway),
external targets (fixed URL), Ollama (its adapter); one renderer to
LiteLLM's model entry; `routes list` uses the same object. Remove
`UPSTREAM_ROUTE` if unneeded; persist a versioned serialization of the
route in the registry if it removes branches. Same route type, different
publication mechanisms (static registry vs dynamic reconciliation), not one
storage for everything. Delete `render_front_door` branches that only tests
reach (cleanup B), never a supported mode.

### 33. [x] External routes in static and dynamic routing
*Done 2026-09-27* (with 32): static and dynamic routes on every backend, the
`('external', alias)` id, the key name in `model_info` for reconcile. Tests:
the route survives dedicated deployments coming and going, zero deployments,
release and gc; redefinition replaces one route in place; unpublishing
deletes it (`tests/test_external_endpoints.py`).
Dynamic mode: an external route is standing desired state with a stable
managed id from its logical owner, the endpoint alias (decision 28.5), not
from target semantics; the semantic route decides whether it is replaced. It survives zero leases,
releases, other acquires, gc, a dedicated deployment going away. Route
retirement (item 17) stays intact; infer-stack never tears down an external
upstream, only its route.

### 34. [x] External credentials by env reference
*Done 2026-09-27.* Host gateway: `NAME: ${NAME}` in LiteLLM's environment,
so the fingerprint recreates it when the value changes. In-cluster: the
Secret carries each routed key, and the pod's key-hash covers them (the hash
is unchanged when there are none). `env NAME=...` says the gateway uses it
after `infer-stack apply`. `routes seed` refuses a key with no value, and a
render warns. Real LiteLLM: `dev/external_e2e.sh` (static and dynamic, key
rotation against a mock that accepts one key) passes on the guest.
Also fixed: `profile_drift` compared catalog sources by bytes, so after a
seed every command warned "catalogs" drifted; it now compares meaning.
`api_key_env` only; never a literal key in the catalog, registry, rendered
route or logs. Render as a LiteLLM env reference; the gateway container
receives the variable from the managed `.env` (`infer-stack env KEY=...`).
Fail before apply if a declared key has no value; never generate provider
keys. Per decision 28.4: host Compose service environment and fingerprint;
in-cluster gateway Secret and pod-template hash; the chosen `env KEY=new`
semantics; reserved-name validation. Real test against the pinned LiteLLM:
add a model through the admin API with `api_key = os.environ/TEST_KEY`,
request, change `TEST_KEY`, recreate/roll as designed, request again. Test
both gateway placements. Dynamic routing: LiteLLM redacts credentials when listing, so do not
read keys back; the route identity/fingerprint includes the env NAME (a
changed reference forces replacement), never the value; a changed value
triggers what the gateway needs to see it, without leaking it.

### 35. [x] Publication without a ledger mutation (one transaction, 28.3)
*Done 2026-09-27.* `Controller.publish_endpoints`: under the lock, the
profile candidate (invocation catalogs merged into the union), a preview
with approval, `ledger.publish_profile` (profile + approved marker at once),
`_publish`. Mixed access is `acquire`, whose transaction already carries the
candidate. No CLI writes gateway state.
External-only access may change nothing in the ledger but still needs:
check/incorporate the invocation catalog/profile, render gateway state,
persist routes, apply under the publication lock with the pending /
approved-render machinery. Factor it so managed acquire and external-only
access reach the same publication coordinator; no second ad-hoc gateway
apply path; no private gateway writes from the CLI.

### 36. [x] Access above leasing
*Done 2026-09-27.* `ConnectionInfo` + `backend.connection_info()` /
`request_names()` (typed, in the protocol) replace the `access` hook;
`Controller.access` -> `AccessResult` (endpoints, external, request names,
connection, lease or `None`, the acquire's outcome, front-door readiness);
a front door that never accepts its key releases the lease the call took.
`infer-stack access NAMES [--env-file]`; `run` uses it; the descriptor
omits `INFER_STACK_LEASE_ID` without a lease; `release --env-file` of such
a file says there is nothing to release (exit 0); `acquire` of an external
endpoint points at `access`.
*Review:* split the front door's part from the endpoints': a typed
`FrontDoorControl.connection_info()` (base URL, front-door credential,
optional UI URL), plus the resolved endpoints / routes (with LiteLLM the
request name is the alias), make an `AccessResult`; the untyped
`backend.access(endpoints) -> dict` goes (renamed, not a second "access").
Mixed access is one transaction (28.3); `access` waits for front-door
readiness (28.6); failure semantics per 28.7.

An access result: requested endpoints, base URL, credential info,
alias -> request-model mapping, optional real lease, optional GPU
reservation metadata (external-only: `lease=None`; mixed: one real lease
for the managed subset; the lease never lists external names). The env
format omits `INFER_STACK_LEASE_ID` when there is no lease (no sentinel);
managed-only descriptors stay byte-identical where practical. A high-level
`prepare_access`/`acquire_access` partitions managed/external, leases only
managed, ensures publication, waits for managed runtime as today, returns
one result. `infer-stack access ... [--env-file]` for workflows without
`run`; `run` uses it. `acquire` stays lease-only: for external-only it
says "'qwen' is externally provided and does not require a lease; use
`infer-stack access qwen` or `infer-stack run ...`"; mixed bundles go
through access. `release` of external-only access has nothing to release.
Access is a typed capability, not a `getattr(backend, 'access')` hook
(cleanup D).

### 37. [x] Views: `external`
*Done 2026-09-27.* `status` lists published external endpoints in their own
section (alias, upstream model, server; "no lease"); `leases` is unchanged;
`routes list` shows origin `external`; the TUI's catalog shows engine
`external` with the upstream model, and its endpoint editor refuses an
external endpoint (saving would have turned it into a vLLM one). `catalog
show` prints the `external:` block as written.
Status, TUI and catalog views show external targets as `external`.
*Review:* no pseudo-deployment rows: `leases` stays leases and
deployments. Show external endpoints in an endpoint/routing section
(`status`, the TUI's catalog or routes view, `catalog show`).

### 38. [x] `FrontDoor` naming collision (cleanup C)
*Done 2026-09-27:* `FrontDoorControl` (the capability, now with
`connection_info`) and `RenderedFrontDoor` (the render).
If both the controller capability and the rendered artifact are still
called `FrontDoor`, rename (e.g. `FrontDoorControl`, `RenderedFrontDoor`).

### 39. [x] Tests (not only parser tests)
*Done 2026-09-27:* all 22 in `tests/test_external_endpoints.py` (plus the
existing managed `run` tests for 7), KubeAI parity for routes, and the mixed
bundle through the CLI (`access pair`, `run --endpoint pair`).
1 external parses and round-trips semantically; 2 invalid mixtures fail
clearly; 3 no lease/deployment for external; 4 external-only access
publishes a route with zero ledger demand; 5 mixed bundle creates only
managed demand; 6 `run` works external-only and mixed; 7 managed-only `run`
unchanged; 8 external needs a usable front door; 9 static routing keeps
external routes across unrelated acquire/release/gc; 10 same for dynamic;
11 two catalogs, same alias and target, union cleanly; 12 same alias,
different target, conflict; 13 an active managed `qwen` cannot be silently
redefined external; 14 once unpinned the same edit succeeds; 15 external ->
managed likewise; 16 upstream key referenced by name, never serialized as
value; 17 missing required key fails before apply; 18 a changed key
reference forces route reconciliation; 19 external-only descriptor has no
fake lease id; 20 releasing a mixed access result affects only its real
lease; 21 `routes list` renders the same target as the gateway renderer;
22 external routes recover after a process restart from durable state.
Parity: Compose and KubeAI-with-gateway; on KubeAI an external route must
not be rewritten to point at the cluster gateway.
Mixed bundle acceptance (`access pair` / `run --endpoint pair`, pair =
managed `local-model` + external `remote-model`): one real lease for
`local-model`, none for `remote-model`, wait for the local one, external
route published, one base URL, both aliases in the descriptor, only the
managed lease released at the end.

### 40. [x] Real e2e with a local fake OpenAI server
*Done 2026-09-27:* `dev/external_e2e.sh`, on the guest with no GPU: real
LiteLLM, the mock OpenAI server as the external upstream (accepts one key),
llm-d-inference-sim as the managed endpoint. Static and dynamic routing:
key refusal, publish, key rotation, prune, external-only access, managed
access, mixed bundle, managed alias moved to external. Passes, from a
frozen copy. It found three bugs the unit tests missed (catalog drift
warnings, orphaned bundles on a redefinition, lowercase `authorization`).
Not run: the same on KubeAI (unit-tested only).
No internet provider. Phases: external only; managed only; mixed bundle;
managed alias -> external after release; external under dynamic routing if
practical. Run from a frozen copy.

### 41. [ ] Docs, journal, verification
Design doc: Endpoint (public contract) / Fulfillment (managed runtime or
external upstream) / Lease (only when infer-stack owns runtime), with the
workflow (`external:` in the catalog; `infer-stack access qwen --env-file
./qwen.env`; `infer-stack env REMOTE_QWEN_API_KEY=...`) and that switching
back to managed changes neither alias nor workflow. Update architecture
docs, not only CLI help. Journal: fake lease vs access above leasing;
`engine: external` vs target ownership; registry as authoring vs derived
state; direct remote vs stable front door. Full suite, `ty`, flake8, UX
audits, the relevant real e2e.

## Non-goals
Direct external access (a second access mode where the alias stops being
the model name) is a possible later optimization, not campaign 2.
Arbitrary providers; Anthropic/Bedrock/Azure schemas; direct multi-base-URL
descriptors; health ownership or lifecycle of external services; fake
leases; a service mesh; rewriting existing catalog entries. The first
external target is an already-running OpenAI-compatible upstream; the
target union allows more later.
