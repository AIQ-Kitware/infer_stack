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

Done 2026-09-26; the two-machine run is `dev/handover/p5_two_hosts.sh`.

Render the gateway as a Deployment + Service behind an ingress; `secrets
rotate` becomes a Secret update and a rollout; `doctor` checks the ingress.
Try a second k3s agent in a Docker container on the guest; write the
"add a workstation" runbook into `kubeai-backend.md`.

**Done when:** on one node, a card reaches a Model through the in-cluster
gateway and `secrets rotate` works. If the simulated second node fits in the
VM, a Model pinned to it by node selector is served through the same
gateway. `dev/handover/p5_two_hosts.sh` exists for one run across two real
machines.

### 8. [ ] The README's Compose sections, and the other stale docs

About 33 references to verbs that no longer exist (`setup`, `up -d`,
`switch`, `describe-profile`, `smoke-test`, `wait-ready`, `diagnose`), and
top-level `restart` / `stop` / `start` / `pull` (they live under `stack`).
*Why widened (2026-09-26):* the same verbs appear in
`docs/persistent-caches-and-warm-restarts.md` and
`docs/stack-graph-profiles.md`.

**Done when:** every command in the README and `docs/` runs as written
against the current CLI, and `grep` finds none of those verbs.

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

- [ ] `r` does not reload catalogs, and edits to a catalog file are not
      picked up until restart.
- [x] `infer-stack logs` should accept a container name, a prefixed name or
      an endpoint name. Done with P2: an instance name, a container id
      prefix, a deployment id or an endpoint alias.
- [x] `infer-stack logs -f qwen` followed every instance: a kwconf flag took
      the next word as its value. Fixed for every command (P2).
- [ ] `status` shows STALE during an apply instead of "apply in progress".
- [ ] Open WebUI writes its data directory as root: removing a data root
      (the e2e's cleanup, or an operator's `rm -rf`) fails with permission
      denied. Found by `dev/kubeai_e2e.sh` 2026-09-26.
- [ ] A refused acquire always ends with "free a GPU first — …", also when
      the reason is a render refusal (a served-name collision) or the
      backend is kubeai, where no GPU is ours to free. Found 2026-09-26.

### 10. [ ] Handover

Summarize for the operator: what was verified here, and the two handover
scripts (items 6 and 7) with what each run proves.
