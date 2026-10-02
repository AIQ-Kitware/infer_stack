#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. The implementation lives in infer-stack now so the
# CLI, tests, and docs share one setup authority.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(kube k3s bootstrap)
if [[ -n "${INSTALL_K3S_VERSION:-}" ]]; then
  ARGS+=(--version="${INSTALL_K3S_VERSION}")
fi
exec python3 "$ROOT/manage.py" "${ARGS[@]}" "$@"
