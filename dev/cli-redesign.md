# infer-stack CLI redesign — submodals, a catalog editor, and `help tree`

Status: **implemented** (decisions: noun-verb grammar, full phased reorg).
Shipped across phases: `catalog` editor + `help tree`; `config` submodal +
settings store (honors `backend`/`data_dir`); `secret` get/set/list; `stack`
day-2 group; the pre-leasing profile world grouped under `legacy`. Top-level is
now the leasing loop + submodals. Companion to the leasing redesign
(`dev/leasing-followups.md`, `dev/infer-stack-redesign-critique.md`).

The tree and rationale below are kept as the design record; `infer-stack help
tree` prints the live surface.

## What we have

`infer-stack` is one flat `scfg.ModalCLI` (`infer_stack/cli/__init__.py`) with
~38 top-level subcommands spanning three eras mashed together:

- **Leasing (the new primary model):** `acquire release renew run serve leases
  secrets`
- **Day-2 ops on the running stack:** `logs ps restart pull start stop`
- **Meta:** `version config(+paths) paths status env`
- **Legacy profile/active-profile world:** `setup init resolve validate lock
  render switch list-models list-profiles explain describe-profile
  verify-profile kubeai-sync-resource-profiles up down purge deploy diagnose
  wait-ready smoke-test benchmark ollama-pull ollama-list ollama-ps`

Problems:
- **Sprawl** — 38 flat verbs; the hot path (acquire/run) is buried among
  legacy profile machinery.
- **No ergonomic catalog editing** — adding a model/endpoint means hand-editing
  `~/.config/infer_stack/catalog.yaml`. (The demo literally `cat >` a heredoc.)
- **`init` is mis-placed** — it's config bootstrap, not a top-level verb.
- **No way to see the whole surface** — nested help requires walking each group.

Good news: `config` is already a nested `ModalCLI` (`config paths`), and
scriptconfig 0.9.1 supports **3-level** nesting (verified: `infer-stack catalog
model add NAME` works, and group-level `--help` renders). So everything below is
feasible with the framework we already use.

## Principles

1. **Hot paths stay at the top level.** The kwdagger/AIQ pipeline-node loop —
   `acquire`, `run`, `release`, `leases`, `status` — and the two most
   common day-2 verbs (`logs`, `ps`) remain one token deep.
2. **Everything else is grouped into noun submodals** (`catalog`, `config`,
   `secret`, `stack`) so the surface reads as a small set of areas.
3. **`legacy` is a holding pen,** not a graveyard: the pre-leasing profile world
   moves under `infer-stack legacy …` verbatim. We promote commands out as we
   give them leasing-native behavior, and delete `legacy` wholesale once empty.
4. **`help tree` shows the totality** (ported from `aivm help tree`).

## Proposed CLI tree

```
infer-stack
│  # ── hot path: the leasing loop (top-level) ──
├── acquire <ep…>            lease endpoints, block until ready
├── run -- <cmd>             acquire, run cmd with endpoint env, release
├── serve <ep…>              standing service (infinite lease)
├── release <id|--env-file>  release a lease
├── renew <id>               extend a lease's TTL
├── leases                   show leases + deployment groups
├── status                   overall stack status (+ leasing summary)
├── logs / ps                day-2 convenience aliases (-> stack logs/ps)
├── secrets [KEY]            convenience alias (-> secret get)
├── version
├── help [cmd…] | help tree  per-command help; `tree` prints the whole surface
│
│  # ── catalog: edit the user catalog without raw YAML (NEW) ──
├── catalog
│   ├── init                       write a starter catalog.yaml
│   ├── path                       print the catalog path
│   ├── show [name]                pretty-print the catalog (or one entry)
│   ├── validate                   parse + cross-reference check
│   ├── edit                       open $EDITOR (escape hatch)
│   ├── model    add|list|show|rm  models:        source/revision/quantization
│   ├── endpoint add|list|show|rm  endpoints:     engine/model/runtime/reclaim
│   ├── host     add|list|rm       runtime_hosts: ollama daemons / placement
│   └── bundle   add|list|rm       bundles:       named endpoint groups
│
│  # ── config: where things live + durable settings ──
├── config
│   ├── init                       was top-level `init`
│   ├── paths                      (existing) resolved file/dir locations
│   ├── show                       effective config
│   ├── set <key> <value>          persist defaults — e.g. backend=compose,
│   │                              data_dir=/data/infer-stack  (kills two smells)
│   └── edit                       open $EDITOR
│
│  # ── secret: the managed LiteLLM key + friends ──
├── secret
│   ├── get [KEY]                  was `secrets`; $(infer-stack secret get …)
│   ├── set KEY=VALUE              e.g. HF_TOKEN -> the managed .env
│   └── list                       export-style lines
│
│  # ── stack: day-2 ops on the running (leased) compose stack ──
├── stack
│   ├── logs ├── ps ├── restart ├── start ├── stop ├── pull └── down
│
│  # ── legacy: pre-leasing profile/active-profile world (promote out, then delete) ──
└── legacy
    ├── setup ├── render ├── switch ├── resolve ├── lock ├── validate
    ├── list-models ├── list-profiles ├── explain ├── describe-profile
    ├── verify-profile ├── kubeai-sync-resource-profiles
    ├── up ├── down ├── purge ├── deploy
    ├── env ├── diagnose ├── wait-ready ├── smoke-test ├── benchmark
    └── ollama-pull ├── ollama-list └── ollama-ps
```

### The `catalog` submodal (the headline new feature)

A `CatalogEditor` loads the YAML, mutates the dict, validates via
`Catalog.from_dict`, and writes it back (atomic temp+rename). Flags map 1:1 to
the schema so the demo's heredoc becomes:

```bash
infer-stack catalog init
infer-stack catalog model add smol17b --source hf://HuggingFaceTB/SmolLM2-1.7B-Instruct
infer-stack catalog endpoint add chat --engine vllm --model smol17b \
    --max-model-len 8192 --gpu-mem 0.4 --extra-arg --dtype=half \
    --reclaim keep-warm
infer-stack catalog endpoint add chat-fast --engine vllm --model smol135 --reclaim stop
infer-stack catalog host add local-ollama --engine ollama --gpu 0 --keep-alive 5m
infer-stack catalog bundle add pair chat chat-fast
infer-stack catalog validate
```

Notes / decisions for this group:
- **Round-tripping comments:** plain `yaml` drops comments. For a
  tool-generated catalog that's acceptable; if we want to preserve hand-written
  comments, use `ruamel.yaml`. Proposed: start with plain yaml, revisit.
- **`add` is idempotent-ish:** `add` errors if the name exists unless
  `--force`/`--update`; `rm` is by name with a friendly "not found".
- **`--dry-run`** prints the resulting YAML without writing.
- Validation reuses the existing `Catalog` parser, so the editor can never write
  a catalog the leasing path would reject.

### `help tree`

Walk `ManageCLI`'s registered subconfigs recursively (each is a `DataConfig` or
nested `ModalCLI`), printing an indented tree with the one-line description from
each class docstring. `infer-stack help <cmd…>` dispatches to that command's
`--help`. Pure introspection over the modal registry — no per-command wiring.

## Migration plan (phased; no hard back-compat required)

The user has said back-compat isn't required, but we minimize churn to the
already-green e2e harness + demo, which only use top-level
`acquire/run/serve/release/renew/leases/secrets/status/logs/ps/paths`.

1. **Add `catalog` + `help tree`** (purely additive). Unblocks the demo's
   "add a model without YAML" and gives the totality view. Lowest risk.
2. **Add `config` subcommands** (`init`→`config init` with a thin top-level
   `init` shim that warns; `config set/show/edit`). Wire `config set` to a real
   settings store so `--backend compose` / `data_dir` stop being repeated
   (leasing-followups smells #1/#2).
3. **Add `secret` group** (`secret get/set/list`); keep `secrets` as a top-level
   alias for `secret get` (script compatibility for `$(infer-stack secrets …)`).
4. **Add `stack` group**; alias `logs`/`ps` at top level; the other day-2 verbs
   move under `stack`.
5. **Introduce `legacy`**; move the profile-world commands under it. Keep
   top-level shims for any that the e2e/consumers still call, emitting a
   one-line "moved to `legacy …`" deprecation note, for one release.
6. **Prune:** once nothing calls the shims, delete them; promote any `legacy`
   command we've made leasing-native; delete `legacy` when empty.

Each phase is independently shippable and testable. Phases 1–3 are the high
ergonomic payoff; 4–6 are cleanup.

## Open questions for review

- `catalog model add` (noun-verb, 3-level) vs `catalog add-model` (verb-noun,
  2-level)? Proposal: **noun-verb** — scales better, reads like `aivm`/`kubectl`.
- Keep top-level `secrets`/`logs`/`ps` aliases, or force the grouped form?
  Proposal: **keep the few hottest as aliases.**
- `config set` backing store: a new `settings.yaml` in the config dir vs folding
  into the existing `config.yaml`. Proposal: a small `settings.yaml` owned by the
  leasing world (config.yaml stays the legacy profile artifact).
