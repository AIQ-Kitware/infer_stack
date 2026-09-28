#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the historical positional interface. New callers
# should prefer:
#   infer-stack kube k3s join --server=... --token-file=... [--node-name=...]
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <server-url> <node-token> [node-name]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_URL="$1"
NODE_TOKEN="$2"
NODE_NAME="${3:-}"
TOKEN_FILE="$(mktemp)"
trap 'rm -f "$TOKEN_FILE"' EXIT
chmod 600 "$TOKEN_FILE"
printf '%s\n' "$NODE_TOKEN" > "$TOKEN_FILE"

ARGS=(kube k3s join --server="$SERVER_URL" --token-file="$TOKEN_FILE")
if [[ -n "$NODE_NAME" ]]; then
  ARGS+=(--node-name="$NODE_NAME")
fi
if [[ -n "${INSTALL_K3S_VERSION:-}" ]]; then
  ARGS+=(--version="${INSTALL_K3S_VERSION}")
fi
python "$ROOT/manage.py" "${ARGS[@]}"
