# CI smoke test

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../../README.md).

Walks through the canonical `infer-stack` workflow end-to-end: pick a
profile, render the deployment, inspect the rendered artifacts, and
exercise a few read-only inspection commands.

This file is **double-runnable**: read it as a tutorial, or replay
every code block in order with

```text
pytest --codeblocks docs/demos/ci_smoke_test.md
```

to verify each step still works (the test suite does this in CI). Each
fenced ``bash`` block is a self-contained shell script — copy-paste any
one of them into a terminal and it runs on its own.

All blocks share a single working directory so they build on each
other within one demo run:

```text
$INFER_STACK_CONFIG_DIR  = /tmp/infer-stack-demo/config
$INFER_STACK_DATA_DIR    = /tmp/infer-stack-demo/data
```

Each block re-exports those variables at the top — that's required by
``pytest-codeblocks`` (each block runs as its own subprocess) and it
also makes the blocks usable as standalone shell snippets.

## 1. Clean slate

Wipe any previous demo state so this run starts fresh.

If a previous run left containers running, use ``infer-stack purge`` so that
Docker-owned (root-written) directories are removed correctly:

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack purge --yes --delete-cache 2>/dev/null || true
```

Then recreate the empty directories:

```bash
rm -rf /tmp/infer-stack-demo
mkdir -p /tmp/infer-stack-demo/config /tmp/infer-stack-demo/data
```

## 2. Set up a profile

Pick the smallest vLLM test profile (``test-single-11gb`` runs SmolLM2 135M
on a single GPU) and write a config for it. This demo exercises the classic
`Open WebUI -> LiteLLM -> vLLM` path; use
``docs/demos/ollama_direct_quickstart.md`` for the no-LiteLLM Ollama path:

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack setup --backend compose --profile test-single-11gb
```

You should see ``Wrote …/config.yaml`` and a summary of the configured
backend / active profile. After this, ``config.yaml`` exists under
``$INFER_STACK_CONFIG_DIR``. A ``models.yaml`` file is only needed when you
add local ``vllm_models``, ``ollama_models``, or custom stack profiles.

## 3. List the catalog

What profiles are available?

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack list-profiles | head -20
```

## 4. Inspect a single profile

``describe-profile`` shows what a profile actually resolves to —
served model name, endpoint shape, transport details — without
rendering anything yet:

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack describe-profile test-single-11gb --format yaml --simulate-hardware 1x24 | head -30
```

The ``--simulate-hardware`` flag pretends this host has the requested
number of GPUs, so describe-profile can plan placement even on a
machine without an actual GPU. Real hosts can omit it.

## 5. Validate before rendering

``validate`` runs the resolver and the policy checks; it exits
non-zero if the resolved deployment has errors. It writes ``plan.yaml``
as a side effect so subsequent ``render`` calls see a consistent plan.

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack validate --simulate-hardware 1x24
```

A clean run prints ``"ok": true`` with empty ``errors`` and
``warnings`` arrays.

## 6. Render the compose stack

``render`` writes the actual ``docker-compose.yml``, ``.env``, and
mounted-runtime files. ``--yes`` skips the interactive per-file
confirmation diff (which you want for scripted use).

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack render --yes --simulate-hardware 1x24
```

You can confirm the artifacts landed where you expect:

```bash
test -f /tmp/infer-stack-demo/data/generated/docker-compose.yml
test -f /tmp/infer-stack-demo/data/generated/.env
test -f /tmp/infer-stack-demo/data/generated/plan.yaml
```

## 7. Constrain placement to specific GPU indices

If GPU 0 is already in use, you can ask the planner to consider only
other GPUs without editing any profile:

```bash
export INFER_STACK_CONFIG_DIR=/tmp/infer-stack-demo/config
export INFER_STACK_DATA_DIR=/tmp/infer-stack-demo/data
infer-stack render --yes --simulate-hardware 4x24 --allowed-gpus 1
grep -A1 device_ids /tmp/infer-stack-demo/data/generated/docker-compose.yml | head -3
```

The rendered ``device_ids`` for the vLLM container now pins to GPU 1.
``INFER_STACK_ALLOWED_GPUS=1,3`` works as an env-var equivalent.

## 8. Programmatic API

Every subcommand is also importable as a Python class — useful for
notebooks, integration tests, and anything that wants the CLI's
semantics without spawning a subprocess:

```python
import os, tempfile
os.environ["INFER_STACK_CONFIG_DIR"] = "/tmp/infer-stack-demo/config"
os.environ["INFER_STACK_DATA_DIR"] = "/tmp/infer-stack-demo/data"

from infer_stack.cli import ListProfilesCLI, ValidateCLI

# ListProfilesCLI prints to stdout; capture or just verify it runs.
ListProfilesCLI.main(argv=False)

# ValidateCLI returns 0 on a clean plan, 2 on validation errors.
rv = ValidateCLI.main(argv=False, simulate_hardware="1x24")
assert rv == 0, f"validate returned {rv}"
```

## 9. Tear down

Stop the stack and delete all Docker-written state (postgres data, caches, etc.):

```bash
infer-stack purge --yes
```

Model weights in hf-cache and vllm-cache are preserved by default so a
subsequent run doesn't need to re-download them. Pass ``--delete-cache`` to
also wipe those directories.

Afterwards the ``/tmp/infer-stack-demo`` directory can be removed normally:

```bash
rm -rf /tmp/infer-stack-demo
```

## Where to go next

- ``infer-stack up`` actually starts the compose stack (needs Docker
  and real GPU access). Try it with one of the ``*-single`` profiles
  on a workstation.
- ``infer-stack switch <profile> --apply`` swaps the active profile
  on a running stack.
- The hardware-shape integration profiles are named ``test-*``: pick
  ``test-single-11gb`` for a workstation card or ``test-multi-gpu``
  with ``--allowed-gpus 1,3`` to exercise tensor-parallel rendering.


## Readiness vs Docker health

Use `infer-stack wait-ready` when a test must wait until the active model can
actually serve a request. Compose health can become true before a routed vLLM
model is ready through LiteLLM. `infer-stack smoke-test` runs this readiness
probe by default; pass `--no-wait` only when you intentionally want to test the
currently listening endpoint without waiting for model readiness.
