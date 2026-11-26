## 2026-11-25 17:49:00 -0500

Model: ChatGPT (GPT-6.2); exact serving/thinking configuration not exposed.

The user wanted the CLI equivalent of the TUI's engines-only log selection,
excluding LiteLLM and Open WebUI. Based on snapshot 0b637511df60. The existing
`Instance.is_engine` classification already expresses this distinction across
Compose and KubeAI; no container-name heuristics or new logging machinery are
needed. Add a positive kwconf boolean `--engine` to LogsCLI, default false.
Filter instances before resolving explicit targets and on each follower poll,
so a gateway target is rejected and a follow started before an acquire can
still observe later engines. This retains engine crash/restart history and all
existing log formatting and tail settings. An empty engines-only follow emits a
waiting notice on stderr rather than silently attaching to infrastructure.
The same class serves the `stack logs` alias. Keep the overlay independent of
recent benchmark and registry edits.

Validation: syntax/diff checks; six isolated scenarios executing the actual
LogsCLI class with injected config and fake subprocess/follower (engine and
unfiltered history, alias, empty view, excluded gateway, later deployments).
Three existing runtime/resolve tests passed with an import-only ubelt placeholder.
New end-to-end CLI-level regression tests cover both runtimes and flag parsing,
but could not execute with real kwconf/ubelt here (installation unavailable).
No real Docker or Kubernetes log streams were exercised. The key reusable
principle is to filter by backend-provided deployment identity, not service
names, and apply it to dynamic discovery as well as the initial snapshot.
