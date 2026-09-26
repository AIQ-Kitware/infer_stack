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

### 8a. [ ] Decide: rewrite or delete the pre-leasing recipes

*Why added (2026-09-26):* `docs/demos/`, `recipies/` and `examples/` (about
2,000 lines) are hardware recipes for the removed profile CLI; the Makefile
targets that used them are gone. Each now opens with a "Pre-leasing" banner
saying its commands no longer run. Rewriting them as catalog entries needs
the hardware they describe; deleting them loses tuning notes. **The
operator's call.**

### 8b. [ ] Decide: `/dev/shm` for vLLM containers

*Why added (2026-09-26):* the leasing renderer sets neither `ipc: host` nor
`shm_size`, so a vLLM container gets Docker's 64 MiB `/dev/shm`; the
pre-leasing template had `ipc: host`, and vLLM's own Docker instructions use
it for tensor parallelism. Adding it changes every vLLM service's
fingerprint, so the next apply recreates running engines. Needs a TP=2 run
on a GPU host to confirm the symptom first. **The operator's call.**

### 9. [ ] UX audit loop: do not stop without a passing audit

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

### 10. [ ] Handover

Summarize for the operator: what was verified here, and the two handover
scripts (items 6 and 7) with what each run proves.
