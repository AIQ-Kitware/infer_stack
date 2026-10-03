#!/usr/bin/env bash
set -euo pipefail
# Compatibility with the old optional VALUES_FILE NAMESPACE positional arguments.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(kube install --apply)
if [[ $# -gt 0 && "$1" != --* ]]; then
  ARGS+=(--values "$1")
  shift
  if [[ $# -gt 0 && "$1" != --* ]]; then
    ARGS+=(--namespace "$1")
    shift
  fi
fi
exec python3 "$ROOT/manage.py" "${ARGS[@]}" "$@"
