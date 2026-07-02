# Follow-up work packages from the 2026-07-02 audit

Companion to `dev/audit-2026-07-02.md` (full findings + failure scenarios).
This file turns the *unfixed* items into self-contained work packages an agent
can execute one at a time. Line numbers were correct at commit `6b20b3f`;
re-locate by symbol name if the file has drifted.

Rules for whoever picks these up:
- One work package per commit (conventional style: `fix(area): ...`). Commit as
  you land each one; do not batch.
- Every fix needs a regression test that fails before the fix and passes after.
  Run the full suite (`python -m pytest tests/ infer_stack --xdoctest`) before
  each commit; all 274+ tests must stay green.
- Threat model (per Jon): trusted multi-user host — local users are NOT
  adversaries and group-level file permissions (0660/0640) are the target, not
  0600. What matters is keeping plaintext secrets out of *logs, tracebacks,
  process argv, and generated/copyable commands*; prefer env-var indirection
  like `$(infer-stack env LITELLM_MASTER_KEY)`.

## Already fixed (do not redo)

Commits `627644f`, `e2cb0fd`, `d749569`, `67b0e05`, `6b20b3f`: renew
resurrection; release-of-unknown-id; apply_now generation snapshot; acquire
rollback re-render + never-ran eviction; evict_idle demand guard; evict --json
purity; pp/dp GPU counting + dp structural; serving-knob rendering
(revision/quantization/dtype/pp/chat_template/trust_remote_code/image);
compose service-name collision reporting.

---

## WP1 — `docker_rm_dirs` unquoted `rm -rf` (data loss) [P1]

**File:** `infer_stack/docker_utils.py` (`docker_rm_dirs`, ~line 255-272).
**Bug:** directory basenames are joined unquoted into `sh -c 'rm -rf /mnt/a /mnt/b'`.
A name with a space deletes sibling dirs of the bind-mounted parent; a glob char
expands. **Fix:** drop `sh -c` entirely — pass argv directly:
`[docker_cmd, 'run', '--rm', '-v', f'{parent}:/mnt', 'alpine', 'rm', '-rf', *(f'/mnt/{n}' for n in names)]`.
**Test:** unit-test the constructed argv (inject a fake `run`); include a name
with a space and assert it stays one argument.

## WP2 — `nvidia-smi` robustness [P1]

**File:** `infer_stack/hardware.py`.
1. `subprocess.check_output([...])` at ~line 11 has no `timeout` — a wedged
   driver hangs every CLI forever. Add `timeout=20` and convert
   `TimeoutExpired` into the same "no inventory" fallback as a missing binary.
2. ~Line 62-63: `int(float(mem))` crashes on `[N/A]` / `[Insufficient
   Permissions]` fields (MIG, fallen-off-the-bus GPUs). Skip (or zero out, with
   a warning) unparsable GPUs instead of crashing `detect_inventory()`.
**Test:** fake `check_output` returning a `[N/A]` row → inventory still returned
without that GPU; fake raising `TimeoutExpired` → graceful fallback.

## WP3 — `our_published_ports` crashes on empty compose ps [P1]

**File:** `infer_stack/docker_utils.py` ~139-162.
**Bug:** `docker compose ps --format json` output `[]` falls through to
line-parsing; `json.loads('[]')` is a list and `_collect_published_ports`
calls `.get` on it → `AttributeError`. **Fix:** treat a parsed top-level list
as "parsed" (iterate its dicts) regardless of emptiness; ignore non-dict rows.
**Test:** feed `'[]'` and `'[{"Publishers": null}]'` through the parser.

## WP4 — `_reconcile_routes` blocks apply for ~3 min when the gateway is down [P1]

**File:** `infer_stack/leasing/compose.py` (`_reconcile_routes`, retry loop
~90 × 2 s). **Fix:** cut the budget to ~15-30 s total, and/or probe
liveness once (fast connect timeout) before entering the loop; keep the
existing "leaving it for the next converge" warning path. Do not remove the
retry entirely — the gateway legitimately takes a few seconds to come up after
first render. **Test:** fake http that always refuses → converge returns within
the budget; assert the warning fired.

## WP5 — secrets out of argv/logs (kubeai helm token) [P1, security posture]

**File:** `infer_stack/kubeai_ops.py` ~14-19, 62-65.
1. `--set secrets.huggingface.token=<token>` puts the HF token on the helm
   argv (visible in `ps`) — pass via `--set-file` with a temp file (0600,
   deleted after) or values-on-stdin.
2. `run()`'s `CommandError` message embeds the full command; redact any argv
   element matching `token=`/`key=` patterns before formatting the error.
**Test:** simulate a failing helm call; assert the raised message does not
contain the token string.

## WP6 — TUI "Copy curl" bakes the literal master key [P1, security posture]

**File:** `infer_stack/tui.py` (`_curl_for`, ~2151-2165).
**Fix:** emit `-H "Authorization: Bearer $LITELLM_MASTER_KEY"` (double-quoted
so the shell expands it) instead of the raw key, and add a leading comment line
`# export LITELLM_MASTER_KEY="$(infer-stack env LITELLM_MASTER_KEY)"` so the
paste works in a fresh shell. Also JSON-escape/quote the prompt correctly —
an apostrophe in the prompt currently breaks the single-quoted `-d '...'`
(use `shlex.quote` on the body). **Test:** `_curl_for` output contains no key
material and survives `shlex.split` with an apostrophe prompt.

## WP7 — escape `$` in rendered compose values [P1, security posture]

**File:** `infer_stack/leasing/compose.py`. Compose interpolates `${VAR}` at
parse time using the `--env-file` (which holds the master key). Catalog-
controlled strings (extra_args, served names, chat_template...) must have `$`
escaped to `$$` when rendered into the compose dict (service `command`,
`environment` values). Add a small `_compose_escape(value)` applied at the
render boundary. **Test:** render an endpoint whose extra_args contains
`${LITELLM_MASTER_KEY}`; assert the compose file contains `$${LITELLM...}`.

## WP8 — `.env` handling: group perms + stderr [P2]

**File:** `infer_stack/env_utils.py` ~85-97.
1. Write the `.env` 0660 (group-scoped; NOT 0600 — see threat model): open via
   `os.open(path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o660)`.
2. Move `print(f'Write .env to {path}')` to stderr (it currently pollutes
   machine-read stdout).
3. `ensure_secret` (~36-38): strip surrounding quotes from the parsed value
   before the `startswith(prefix)` check so a hand-quoted key is not silently
   rotated (rotation invalidates distributed credentials).
**Test:** tmp-path round trip asserting mode, stderr, and no rotation of a
quoted existing secret.

## WP9 — `config init` destroys dict-valued `reverse_proxy` [P1]

**File:** `infer_stack/cli/commands_meta.py` (`ConfigInitCLI`, `_as_bool`
~353, 445, 463-465). A `reverse_proxy: {enabled: true, port: ..., config_path:
...}` block set via `config edit` is normalized through `_as_bool(dict)` →
always False; re-running `config init` rewrites it to `reverse_proxy: false`.
**Fix:** if the existing value is a dict, preserve the dict — seed the prompt
default from `dict.get('enabled')`, and on save merge `{**old_dict, 'enabled':
answer}` instead of a bare bool. **Test:** seed settings with the dict form,
run `ConfigInitCLI.main(argv=['--yes'])`, assert port/config_path survive.

## WP10 — `--ttl 0` means infinite (undocumented trap) [P2]

**File:** `infer_stack/cli/commands_leasing.py` (`_parse_duration` ~74-77).
**Fix (choose one, document either way):** make `'0'` an error ("ttl must be
positive; use 'none' for infinite") — preferred — or keep it and document in
the `--ttl` help. Update the help string regardless. **Test:** CLI-level
`_parse_duration('0')` behavior + help text mentions the rule.

## WP11 — unify `--allowed-gpus` / `INFER_STACK_ALLOWED_GPUS` [P2]

**Files:** `infer_stack/cli/commands_leasing.py` (`_parse_gpus` ~94-97, used at
~232) vs `infer_stack/cli/context.py` (`effective_inventory` ~42-93).
`catalog suggest` honors the env var and errors nicely; `acquire/release/
evict/gc` ignore the env var and traceback on `--allowed-gpus x,`.
**Fix:** make `_make_backend` use the same resolution helper as context.py
(extract a shared function; env var honored, friendly SystemExit on parse
error). **Test:** set `INFER_STACK_ALLOWED_GPUS=1`, acquire on a fake 2-GPU
inventory, assert placement lands on GPU 1.

## WP12 — exit-code / JSON-schema consistency [P2]

**File:** `infer_stack/cli/commands_leasing.py`.
1. `run` readiness timeout: exit 2 (match `acquire`) and format pending pairs
   like acquire does (~1227-1231), not a raw list repr.
2. `acquire --json` and `wait --json` disagree on `pending` encoding
   (arrays vs objects, ~404-411 vs ~1088-1094): converge on objects
   `{"deployment": ..., "endpoint": ...}` in both.
3. `evict <live-name>` and `catalog suggest` no-fit: exit nonzero (or add
   `--strict` if backward compat matters — check e2e scripts first).
**Test:** JSON-schema assertions for both verbs; exit-code tests.

## WP13 — traceback-instead-of-error paths [P2]

One commit sweeping these to one-line SystemExit errors + tests:
- `release/renew --env-file <missing>`: FileNotFoundError
  (`envfile.read_lease_id` ~128-131 — check existence first).
- non-mapping catalog.yaml: AttributeError in `commands_catalog._load_raw`
  (~70-74) — raise "catalog.yaml must be a mapping".
- `$EDITOR` with spaces / missing binary (`commands_catalog.py` ~400,
  `commands_meta.py` ~563): `shlex.split` the value, catch FileNotFoundError.
- `logs --follow` Ctrl-C traceback (`commands_runtime.py` ~300): catch
  KeyboardInterrupt, return 130.
- `diff_prompt.Confirm.ask` EOFError on non-interactive stdin (~118): treat
  as declined with a message suggesting `--yes`.

## WP14 — catalog `--dry-run` honesty [P2]

**File:** `infer_stack/cli/commands_catalog.py` (`_save_raw` ~85-96, `_rm`
~147-149). Dry-run skips `_validate` (preview can show a catalog the real
write would reject) and `rm --dry-run` prints `removed '<name>'`.
**Fix:** validate before the dry-run return; guard the success line with
`if not config.dry_run`. **Test:** dry-run an invalid add → error; dry-run rm
→ no "removed" line.

## WP15 — TUI: dead log stream never reattaches [P1]

**File:** `infer_stack/tui.py` (`_restart_logs`/`_stream_logs`/
`_terminate_logs` ~1447-1477, `_sync_log_services` ~1409).
**Fix:** when the log subprocess exits (stream loop ends) schedule a reattach
retry (e.g. every poll tick while the Logs pane is expanded); also reattach
when `_sync_log_services` first discovers services and the current proc is
dead/None. Keep it simple: a `_log_stream_dead` flag checked from the poll
timer. **Test:** fake proc factory whose first proc dies immediately, second
lives; assert a second spawn happens after services appear.

## WP16 — TUI: cursor restore by row id, not index [P1]

**File:** `infer_stack/tui.py` (`_restore_view` ~1189-1195, `_diff_fill`
~1284-1289, `_selected` ~1530-1532). A rebuild between highlight and keypress
shifts rows → Release/Evict hit the wrong lease. **Fix:** capture the *id* of
the cursored row before rebuild; after rebuild move the cursor to the row with
that id (fall back to same index if gone). **Test:** pilot test — cursor on
row 2, rebuild with a new row inserted above, assert cursor follows the id.

## WP17 — TUI: worker cancellation + docker timeout [P1]

**File:** `infer_stack/tui.py` (`_refresh_bg` ~1136-1139, `_collect`
~1049-1053, `quiet_run` ~870-881).
1. In `_collect`/`_refresh_bg`, check `get_current_worker().is_cancelled`
   before `call_from_thread(self._render, ...)` so a superseded poll cannot
   render stale data over fresher state.
2. `quiet_run`: add `timeout=` (e.g. 60 s) to `subprocess.run`; on timeout
   return nonempty stderr so the status line shows it.
3. Wrap the `call_from_thread(self._render, ...)` marshal in the same
   try/except as `_collect` so one transient render exception cannot kill the
   app (`exit_on_error`).
**Test:** existing tui test harness; simulate a slow collect superseded by a
mutation refresh.

## WP18 — TUI: log-proc switch race + zombies [P2]

**File:** `infer_stack/tui.py` (`_DockerLogProc.terminate` ~111-117,
`_stream_logs` ~1456-1472). Assign `self._log_proc` *before* streaming starts
(or guard with a generation counter so `_terminate_logs` can kill a proc
spawned after it ran); after `kill()` call `wait()` again and close stdout;
move the 2 s `wait` off the UI thread (thread worker).

## WP19 — TUI small items (one commit, cherry-pick freely) [P2]

`infer_stack/tui.py`: double-click gate keys on (name, button==1) not row
index (~1701-1711); `_fill_catalog` cursor/scroll restore (~942-964);
`_after_mutation` sets `self._observed_at = None` (~2364-2366); ctrl/shift
click outside rows ignored (~1728-1731); `_lease_sel`/`_dep_sel` mutations
marshalled via `call_from_thread` (~2305, 2332, 2347); `_copy` in a thread
worker (~1508-1528); restore `backend.run` + re-enable loguru in
`on_unmount`/after `app.run()` (~864-883, 2402-2407); endpoint-edit wizard
preserves explicit `enable_prefix_caching: false` (~279-280, 2001-2002).

## WP20 — store/ledger secondary items [P2]

- `store.py` `_ensure_schema` fresh-DB race: `INSERT OR IGNORE` for
  schema_version (~140-149) + a two-process open test.
- `store.py` `prune`: don't delete claims of non-terminal leases via the
  deployment-state arm (restrict the deployment-claims delete to claims whose
  lease is also terminal) (~297-335).
- `ledger.acquire`: return data captured inside the transaction instead of
  re-reading after COMMIT (~154-159); guard empty `requests` with a
  ValueError (~118-159).
- `controller.py`: `:memory:` ledger should still take `_tlock` in
  `_global_lock` (~337-339); make `_ensure_applied`'s while-loop shape honest
  (plain `if`, or remove the unconditional final `break`) (~459-489).
- `models.py`: delete dead `VLLM_STRUCTURAL_FIELDS`/`OLLAMA_STRUCTURAL_FIELDS`
  constants or wire them as validation (~69-94); make `capacity_satisfies`
  fail closed on mixed types (~214-218).
- `backend.py`: document/extend the Protocol with the optional converge-style
  surface (`converge(desired, apply=...)`, `apply`, `last_unplaced`,
  `last_errors`, `last_assignments`) (~58-82).

## WP21 — migration-artifact cleanup (branch hygiene) [P2]

- Delete dead mixins in `cli/options.py` (~33-148) after grep-confirming zero
  references; decide the fate of the podman `--compose-cmd` override (either
  re-expose on the day-2 wrappers + ComposeBackend or remove the plumbing).
- Remove `--require-generation` from `acquire`/`wait` epilogs + `WaitCLI`
  docstring; stop plumbing it into ComposeBackend (`_make_backend` ~239).
- Fix stale docstrings: `commands_leasing.py` ~11-16 ("Until the
  Compose/KubeAI backends land..."), "deployment deployments" phrasing
  (~11, ~1366); `cli/__init__.py` ~8-16 module list + dead `requests` seam
  (~137-144).
- `commands_meta.py` `config paths`: import `COMPOSE_FILENAME` /
  `STATE_FILENAME` / `default_ledger_path()` instead of hardcoding (~254-268).
- `commands_runtime.py`: drop dead `command` param (~33); either render the
  built `leases`/`deployments` rows or stop building them (~96-104).
- CLI stores: close `SqliteStore`s (context-manage or try/finally) in
  `commands_runtime.py` ~91 and `commands_leasing.py` ~256 (kills the sqlite
  ResourceWarnings in the test suite).

## WP22 — misc core [P2]

- `config.py`: pin the ollama image (~29); `load_yaml` rejects non-mapping
  top level (~216-217); atomic writes (temp + `os.replace`) for
  `save_yaml`/`save_settings`/`write_env_file` (~229-231).
- `paths.py`: cache `load_settings()` per path+mtime; resolve relative
  `data_dir` against the settings file's directory, not CWD (~148-150);
  `encoding='utf-8'` on all read/write_text (~75, 84, 109, 118).
- `_log.py`: remove only the sink infer-stack added (keep the handler id)
  instead of `logger.remove()` (~41).
- `docker_utils.py`: distinguish EACCES from EADDRINUSE in
  `check_ports_available` (~188-190).
- `hardware.py`: `int(placement.get('gpu_count') or default)` for explicit
  nulls (~121-126).
- `envfile.py`: `shlex.quote` exported values; warn on `_endpoint_var`
  collisions (~35-37, 122-125).
- `suggest.py`: make `fits_on` respect the 0.92 utilization cap (~156-159 vs
  ~214); tighten the `'rtx 20'` pre-Ampere hint (word-boundary match on
  `rtx 20xx` models, exclude "RTX 2000 Ada") (~139, 147-149).
- `catalog.py`: None-guard model specs + CatalogError for missing `source`
  (~176-184); validate `sharing.mode` / `reclaim.policy` against the known
  sets (~119-136).
- `compose.py`: unify the served-name fallback (`sorted(served)[0]`
  everywhere, ~314/436 vs ~142-144); log skipped catalog endpoints in
  `_litellm_model_list_from_catalog` (~367-370); merge partial `state` dicts
  with `default_state_paths()` (~1034).
- `kubeai_renderer.py`: skip writing empty models.yaml (or write a comment
  doc); list the ingress.yaml deletion in the confirm prompt; error on empty
  `resource_profile`.

## WP23 — experimental modules (lowest priority; standalone CLIs)

See audit items 56-58 for full detail: memory-estimator sliding-window KV cap
(dead correct code exists at ~147-186 — call it), batch double-count in
`_steady_state_capacity`, safetensors-dataclass-vs-dict branch, dual-format
double count; catalog-discover author derivation (`repo_id.split('/')[0]`),
memory-hints field names, `'vl'` substring, slug collisions, 429 handling;
stress-test corpus fit, small `--num-facts`, None content handling.
