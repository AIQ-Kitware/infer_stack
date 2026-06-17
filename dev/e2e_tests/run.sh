#!/usr/bin/env bash
# infer-stack leasing — developer e2e harness (run this on yardrat).
#
# Exercises the new leasing/compose CLI surface end to end, captures every
# command's exit code / duration / output, asserts the wiring, and writes a
# single self-contained results dir (report.md + logs + rendered artifacts) you
# can rsync back for review.
#
#   ./run.sh                  # non-serving tiers only (no GPU/docker pulls)
#   ./run.sh --gpu            # + the real serving tiers (vLLM/Ollama on GPU 0)
#   ./run.sh --gpu --only '40 50'   # just single-vllm + coalescing
#   ./run.sh --gpu --keep-running   # leave the compose stack up afterward
#
# Tiers WITHOUT --gpu: environment, dry-run (null backend), ergonomics,
# negative cases. They need infer-stack + docker-compose-v2 present but stand up
# nothing. Tiers WITH --gpu actually serve models and need GPU 0 free + images
# pre-pulled (see README).
set -uo pipefail

E2E_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export E2E_ROOT

GPU=0
ONLY=''
KEEP_RUNNING=0
KEEP_DATA=0
RESULTS=''
DATA_DIR=''
CATALOG="$E2E_ROOT/catalog.yaml"

usage() { sed -n '2,30p' "$E2E_ROOT/run.sh"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=1 ;;
        --only) ONLY="$2"; shift ;;
        --results) RESULTS="$2"; shift ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --catalog) CATALOG="$2"; shift ;;
        --keep-running) KEEP_RUNNING=1 ;;
        --keep-data) KEEP_DATA=1 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
    shift
done

command -v infer-stack >/dev/null 2>&1 || {
    echo 'error: infer-stack not on PATH (activate the venv first)' >&2; exit 1
}
command -v python3 >/dev/null 2>&1 || {
    echo 'error: python3 not on PATH' >&2; exit 1
}

TS="$(date +%Y%m%dT%H%M%S)"
[ -n "$RESULTS" ] || RESULTS="$E2E_ROOT/results/$TS"
mkdir -p "$RESULTS"
[ -n "$DATA_DIR" ] || DATA_DIR="$RESULTS/infer-stack-data"
mkdir -p "$DATA_DIR"

# Exported context every test script + lib.sh relies on.
export E2E_RESULTS="$RESULTS"
export E2E_CAT="$CATALOG"
export E2E_ENABLE_GPU="$GPU"
export E2E_KEEP_RUNNING="$KEEP_RUNNING"
export INFER_STACK_DATA_DIR="$DATA_DIR"

: > "$RESULTS/results.jsonl"

# ---- run metadata + environment capture -------------------------------------
VERSION="$(infer-stack version 2>/dev/null | head -n1)"
GITREV="$(git -C "$E2E_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITDIRTY="$(git -C "$E2E_ROOT" status --short 2>/dev/null | wc -l | tr -d ' ')"
[ "$GITDIRTY" != '0' ] && GITREV="$GITREV+$GITDIRTY dirty"
HOST="$(hostname 2>/dev/null || echo '?')"
WHEN="$(date -Iseconds)"
TIERS=$([ "$GPU" = 1 ] && echo 'non-serving + GPU serving' || echo 'non-serving only')

WHEN="$WHEN" HOST="$HOST" VERSION="$VERSION" GITREV="$GITREV" \
TIERS="$TIERS" GPU="$GPU" DATA_DIR="$DATA_DIR" \
python3 - "$RESULTS/run_meta.json" <<'PY'
import json, os, sys
json.dump({
    'host': os.environ['HOST'], 'when': os.environ['WHEN'],
    'version': os.environ['VERSION'], 'git': os.environ['GITREV'],
    'tiers': os.environ['TIERS'], 'gpu': os.environ['GPU'] == '1',
    'data_dir': os.environ['DATA_DIR'],
}, open(sys.argv[1], 'w'), indent=2)
PY

{
    echo "host:        $HOST"
    echo "when:        $WHEN"
    echo "infer-stack: $VERSION"
    echo "git:         $GITREV"
    echo "data dir:    $DATA_DIR"
    echo "catalog:     $CATALOG"
    echo
    echo '== uname =='; uname -a 2>/dev/null
    echo; echo '== python =='; python3 --version 2>&1
    echo; echo '== docker =='; docker version --format '{{.Server.Version}}' 2>&1
    echo; echo '== docker compose =='; docker compose version 2>&1
    echo; echo '== nvidia-smi =='
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,display_active \
        --format=csv 2>&1 || echo '(no nvidia-smi)'
    echo; echo '== mem/disk =='; free -g 2>/dev/null; df -h "$DATA_DIR" 2>/dev/null
} > "$RESULTS/environment.txt" 2>&1

# ---- select + run test scripts ----------------------------------------------
echo
echo "infer-stack leasing e2e — $TIERS"
echo "results: $RESULTS"
echo

for f in "$E2E_ROOT"/tests/*.sh; do
    name="$(basename "$f")"
    prefix="${name%%_*}"
    if [ -n "$ONLY" ]; then
        # 99_cleanup always runs so we never leak containers
        case " $ONLY " in
            *" $prefix "*) ;;
            *) [ "$prefix" = '99' ] || continue ;;
        esac
    fi
    export E2E_SECTION="${name%.sh}"
    echo "── $E2E_SECTION ──────────────────────────────────────────"
    bash "$f"
done

# ---- assemble the report ----------------------------------------------------
echo
python3 "$E2E_ROOT/render_report.py" --assemble --results "$RESULTS"

if [ "$KEEP_DATA" = 0 ] && [ "$GPU" = 0 ]; then
    : # non-GPU run leaves only a tiny sqlite ledger; harmless, keep it for the report
fi

# ---- summary + rsync hint ---------------------------------------------------
SUMMARY="$(python3 - "$RESULTS/results.jsonl" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in recs if r.get('kind') == 'step']
p = sum(r['verdict'] == 'pass' for r in recs)
f = sum(r['verdict'] == 'fail' for r in recs)
s = sum(r['verdict'] == 'skip' for r in recs)
print(f'{p} passed, {f} failed, {s} skipped')
sys.exit(1 if f else 0)
PY
)"
RC=$?

echo
echo "================================================================"
echo "  $SUMMARY"
echo "  report:  $RESULTS/report.md"
echo
echo "  rsync back with:"
echo "    rsync -av $HOST:$RESULTS/ ./e2e-report-$TS/"
echo "================================================================"
exit $RC
