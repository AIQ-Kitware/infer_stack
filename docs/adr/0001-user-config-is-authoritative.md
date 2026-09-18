# ADR 0001: User config is authoritative; recovery snapshots are internal

- Status: Accepted
- Date: 2026-09-17

## Context

The primary infer-stack leasing workflow is deliberately small:

```bash
infer-stack config init
infer-stack catalog suggest --apply
infer-stack acquire <endpoint>
```

The same rule applies to hand edits and the TUI: once the user changes the
normal config/catalog, the next ordinary operation should use that state. A
separate mandatory "publish" step creates a third configuration surface and
breaks this workflow.

Leasing still needs a frozen render context for crash-safe publication. Without
one, a process can commit desired state, die, then recover under different
image pins, gateway/UI settings, or catalog definitions. The recovery render
would no longer be the render that was admitted. This race is real, but the
snapshot that closes it is an implementation detail, not user-authored config.

## Decision

`config.yaml` and the user catalog remain the authoritative configuration.
The persisted leasing profile is renamed conceptually to the **recovery
snapshot** and is maintained automatically.

### Normal acquire

On acquire, under the publication lock:

1. Resolve the requested endpoint from the user's current catalog.
2. Compare current user settings/catalog with the recovery snapshot.
3. If the stack is quiescent (no active lease and no managed deployment
   container), replace the snapshot wholesale with current user config.
4. If workloads are resident, keep global render settings frozen for that
   leasing epoch, but merge compatible catalog additions into the snapshot.
   Existing endpoint/bundle/route definitions may not be redefined live.
5. Preview/admit/render using that exact candidate snapshot, then persist the
   snapshot before committing the acquire's desired-state mutation.

Compatible catalog additions therefore support the common pattern of adding a
new model while another model is already serving. A conflicting edit is a real
live-state conflict, not a request to run another configuration command: the
user quiesces the managed stack (release/evict resident deployments), then
retries acquire. The now-quiescent acquire adopts current config automatically.

### Explicit `config publish`

`infer-stack config publish` remains an advanced operation for cases where an
operator intentionally wants to pre-seed a union of multiple runbook catalogs,
preview/pre-pull a future profile, or exercise publication explicitly. It is
not part of the getting-started or ordinary edit/suggest/acquire workflow.

An explicitly seeded multi-catalog union is preserved when a later runbook sees
only one already-published subset. Automatic synchronization only advances the
snapshot on actual drift.

### Long-lived TUI

The TUI updates the controller's invocation-catalog snapshot whenever it writes
or reloads `catalog.yaml`. It does not write the recovery snapshot directly.
The next Acquire uses the same locked auto-adoption path as the CLI.

## Race-safety argument

The recovery snapshot still supplies every non-ledger render input during a
pending publication or recovery. No acquire renders directly from a catalog
that can change underneath it: the command reads a catalog snapshot, enters the
publication lock, derives a compatible recovery snapshot, previews with it, and
then commits against that frozen value.

For a live leasing epoch, catalog synchronization is monotonic: previously
published definitions are retained and incoming definitions are appended only
if the semantic `CatalogUnion` has no conflicts. Thus an unrelated catalog edit
cannot silently change an existing deployment's recovery render.

A wholesale snapshot replacement is allowed only at the same quiescence
boundary used by explicit publication. This preserves the original motivation
for the frozen profile while removing it from the normal UX.

## Consequences

- The documented first-run path stays `config init -> catalog suggest --apply -> acquire`.
- `catalog suggest --apply` needs no follow-up publication command.
- Adding a compatible endpoint while other endpoints are live works directly.
- Global settings changed during a live epoch are deferred until quiescence;
  existing workloads continue with their frozen settings.
- Conflicting redefinitions cannot be adopted live; quiesce the managed stack and retry.
- Multi-runbook pre-seeding remains available through `config publish`, but it
  is an advanced optimization/control surface rather than required state.
