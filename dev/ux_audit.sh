#!/usr/bin/env bash
# The automated part of the UX audit (docs/queue.md, item 9): every command's
# help, the examples in them, common mistakes, and the day-2 commands with a
# model up, on one backend. It prints what it ran and flags the shapes that
# have been bugs before: a traceback, compose/docker wording on kubeai, a raw
# timestamp, ANSI codes in piped output, stray "Write .env" lines.
#
#   dev/ux_audit.sh compose                    # the simulator catalog, no GPU
#   dev/ux_audit.sh kubeai                     # CPU vLLM on the dev cluster
#
# The TUI, the first run from the README and wording consistency stay manual:
# they need eyes (docs/queue.md says what to look at). Isolated config and
# data roots; everything is released and taken down at the end.
set -uo pipefail

BACKEND="${1:-compose}"
# A missing command must not read as a clean pass.
command -v infer-stack >/dev/null || { echo 'infer-stack is not on PATH' >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/infer-stack-ux.XXXXXX")"
export INFER_STACK_CONFIG_DIR="$WORK/config" INFER_STACK_DATA_DIR="$WORK/data"
mkdir -p "$WORK/config"
REPORT="$WORK/report.txt"
FLAGS=0

flag() { echo "!! $*" | tee -a "$REPORT"; FLAGS=$((FLAGS + 1)); }
run() {   # run a command, keep its output, and check it for known bad shapes
  local out rc
  out=$(timeout 300 infer-stack "$@" 2>&1); rc=$?
  {   # the head, and the tail where a failure says why
    echo "\$ infer-stack $*  (rc=$rc)"
    if [ "$(echo "$out" | wc -l)" -gt 26 ]; then
      echo "$out" | head -18; echo '  [...]'; echo "$out" | tail -6
    else
      echo "$out"
    fi
    echo
  } >> "$REPORT"
  [ "$rc" -ge 124 ] && flag "rc=$rc (timeout or not run): infer-stack $*"
  echo "$out" | grep -q 'Traceback' && flag "traceback: infer-stack $*"
  echo "$out" | grep -q 'Write .env' && flag "stray 'Write .env': infer-stack $*"
  echo "$out" | grep -qE 'ttl=@[0-9]' && flag "raw timestamp: infer-stack $*"
  echo "$out" | grep -q $'\x1b\[' && flag "ANSI codes in piped output: infer-stack $*"
  if [ "$BACKEND" = kubeai ]; then
    # The host-side gateway really is a compose project: doctor's "gateway:"
    # checks name docker compose because that gateway needs it.
    echo "$out" | grep -v '^\s*\(INFO\|[0-9:]* INFO\)' | grep -v '^\[[a-z ]*\] gateway: ' \
      | grep -iqE '\bcompose (project|backend)\b|docker compose' \
      && flag "compose wording on kubeai: infer-stack $*"
  fi
  return 0
}

cleanup() {
  infer-stack clean -f >/dev/null 2>&1
  infer-stack stack down >/dev/null 2>&1
  # The data root holds Open WebUI's files and model weights, about 1 GB a
  # run: kept, a day of passes put the dev cluster's node under disk pressure.
  rm -rf "$WORK/data"
  echo "report: $REPORT"
  echo "flags: $FLAGS"
}
trap cleanup EXIT

case "$BACKEND" in
  compose)
    cp "$HERE/dev/e2e_tests/catalog-mock.yaml" "$WORK/config/catalog.yaml"
    ENDPOINT=mock-smol
    ;;
  kubeai)
    cat > "$WORK/config/catalog.yaml" <<EOF
models:
  tiny: {source: hf://Qwen/Qwen2.5-0.5B-Instruct}
endpoints:
  qwen-tiny: {engine: vllm, model: tiny, reclaim: {policy: stop},
              runtime: {resource_profile: ${E2E_RESOURCE_PROFILE:-cpu}, max_model_len: 2048}}
EOF
    ENDPOINT=qwen-tiny
    ;;
  *) echo "usage: $0 compose|kubeai" >&2; exit 2 ;;
esac
infer-stack config set backend "$BACKEND" >/dev/null

echo '== help: every one-line summary is a sentence'
COLUMNS=250 infer-stack help tree 2>/dev/null | sed -E 's/^[│ ├└─]+//' \
  | awk '{$1=""; print substr($0,2)}' | grep -vE '[.)?]$|^$' \
  | while read -r line; do flag "summary ends mid-sentence: $line"; done

echo '== mistakes: each names its cause, none is a traceback'
run acquire nope --yes
run release lease-nope --yes
run wait nope --timeout 5
run logs nope
run evict grp-nope
run renew lease-nope --ttl 1h
run test "$ENDPOINT" --timeout 5

echo "== day-2, with $ENDPOINT up"
run acquire "$ENDPOINT" --yes --ttl 1h --timeout 600 --env-file "$WORK/lease.env"
for cmd in leases status ps env "env OPENAI_BASE_URL" doctor clean gc \
           "renew --env-file $WORK/lease.env --ttl 2h" "routes list" \
           "wait $ENDPOINT" "test $ENDPOINT" "logs $ENDPOINT --tail 3"; do
  # shellcheck disable=SC2086
  run $cmd
done
run release --env-file "$WORK/lease.env" --yes
