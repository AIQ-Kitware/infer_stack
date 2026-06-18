# shellcheck shell=bash
# Helper library for the infer-stack leasing e2e harness.
#
# A test script sources this file and then describes its work as a sequence of
# steps:
#
#   step single-vllm 'acquire a vllm model and serve a chat completion'
#   run  'infer-stack acquire smol-135 --backend compose --catalog "$E2E_CAT"'
#   expect_rc 0
#   expect_out smol-135
#   end_step
#
# Every step records a structured record (id, title, verdict, duration, the
# individual assertions, and a path to the captured combined-output log) as one
# JSON line in "$E2E_RESULTS/results.jsonl". render_report.py turns those lines
# into report.md. The harness never uses `set -e`: a failing assertion marks the
# step failed and the run keeps going so one report covers everything.
#
# Required exported env (set by run.sh): E2E_RESULTS, E2E_SECTION, E2E_CAT,
# INFER_STACK_DATA_DIR, E2E_ENABLE_GPU.

: "${E2E_RESULTS:?lib.sh: E2E_RESULTS must be exported by run.sh}"
: "${E2E_SECTION:=misc}"

E2E_LOGDIR="$E2E_RESULTS/logs"
# Transient per-step files (captured stdout, assertion/note tallies) live in a
# scratch dir, NOT in logs/, so the rsync'd report's logs/ holds only real
# .log files. rsync excludes .scratch/ (see run.sh).
E2E_SCRATCH="$E2E_RESULTS/.scratch"
mkdir -p "$E2E_LOGDIR" "$E2E_SCRATCH"

# ANSI colors (only when stdout is a tty)
if [ -t 1 ]; then
    C_PASS=$'\033[32m'; C_FAIL=$'\033[31m'; C_SKIP=$'\033[33m'
    C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
    C_PASS=''; C_FAIL=''; C_SKIP=''; C_DIM=''; C_RST=''
fi

_now() { date +%s.%N; }

# --- heartbeat lifecycle ----------------------------------------------------
# The per-step progress heartbeat is a backgrounded subshell. In a
# non-interactive script bash makes async commands IGNORE SIGINT, so a Ctrl-C
# would otherwise leave the heartbeat looping and printing long after the run.
# We track its pid and kill it (and its in-flight `sleep` child) on every exit
# path of this test-script process — normal, error, INT, or TERM.
_HB_PID=''
_kill_hb() {
    [ -n "$_HB_PID" ] || return 0
    kill "$_HB_PID" $(pgrep -P "$_HB_PID" 2>/dev/null) 2>/dev/null
    wait "$_HB_PID" 2>/dev/null
    _HB_PID=''
}
trap '_kill_hb' EXIT
trap '_kill_hb; exit 130' INT TERM

# step <id> <title...>
step() {
    CUR_ID="$1"; shift
    CUR_TITLE="$*"
    CUR_VERDICT='pass'          # flips to 'fail' on any failed assertion
    CUR_SKIP=''
    CUR_START="$(_now)"
    CUR_LOG="$E2E_LOGDIR/${E2E_SECTION}.${CUR_ID}.log"
    LAST_OUT_FILE="$E2E_SCRATCH/${E2E_SECTION}.${CUR_ID}.lastout"
    CUR_ASSERT_FILE="$E2E_SCRATCH/${E2E_SECTION}.${CUR_ID}.asserts"
    CUR_NOTE_FILE="$E2E_SCRATCH/${E2E_SECTION}.${CUR_ID}.notes"
    : > "$CUR_LOG"; : > "$LAST_OUT_FILE"
    : > "$CUR_ASSERT_FILE"; : > "$CUR_NOTE_FILE"
    {
        echo "### step: $CUR_ID — $CUR_TITLE"
        echo "section: $E2E_SECTION   started: $(date -Iseconds)"
        echo "--------------------------------------------------------------"
    } >> "$CUR_LOG"
    printf '%s» %s%s  %s\n' "$C_DIM" "$CUR_ID" "$C_RST" "$CUR_TITLE"
}

# run '<shell command string>'   — executes via `bash -c`, captures combined
# output to the step log and to LAST_OUT_FILE, records RC and the duration.
# A heartbeat prints elapsed seconds every 20s so a long acquire/serve doesn't
# look hung (it writes to the console, not to the captured output).
run() {
    local start end
    printf '\n$ %s\n' "$1" >> "$CUR_LOG"
    start="$(_now)"
    # Heartbeat also self-terminates if this test-script process dies (kill -0
    # $$ — in a subshell $$ is the parent shell's pid), covering even SIGKILL or
    # a missed trap. Belt-and-suspenders with the EXIT/INT/TERM trap above.
    ( while true; do
        sleep 20
        kill -0 "$$" 2>/dev/null || exit 0
        printf '    %s… still running (%ds)%s\n' "$C_DIM" \
            "$(awk -v a="$start" -v b="$(_now)" 'BEGIN{printf "%d", b-a}')" \
            "$C_RST"
      done ) &
    _HB_PID=$!
    bash -c "$1" > "$LAST_OUT_FILE" 2>&1
    RC=$?
    _kill_hb
    end="$(_now)"
    DUR="$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')"
    cat "$LAST_OUT_FILE" >> "$CUR_LOG"
    printf '[exit %s in %ss]\n' "$RC" "$DUR" >> "$CUR_LOG"
    return "$RC"
}

_assert() {  # _assert <pass:0|1> <description>
    local ok="$1"; shift
    if [ "$ok" -eq 0 ]; then
        printf 'PASS|%s\n' "$*" >> "$CUR_ASSERT_FILE"
    else
        printf 'FAIL|%s\n' "$*" >> "$CUR_ASSERT_FILE"
        CUR_VERDICT='fail'
    fi
}

expect_rc()      { [ "${RC:-1}" -eq "$1" ]; _assert $? "exit code == $1 (was ${RC:-?})"; }
expect_rc_not()  { [ "${RC:-1}" -ne "$1" ]; _assert $? "exit code != $1 (was ${RC:-?})"; }
expect_out()     { grep -qF -- "$1" "$LAST_OUT_FILE"; _assert $? "output contains: $1"; }
expect_no_out()  { ! grep -qF -- "$1" "$LAST_OUT_FILE"; _assert $? "output excludes: $1"; }
expect_re()      { grep -qE -- "$1" "$LAST_OUT_FILE"; _assert $? "output matches /$1/"; }
expect_file()    { [ -f "$1" ]; _assert $? "file exists: $1"; }
expect_file_has(){ [ -f "$1" ] && grep -qF -- "$2" "$1"; _assert $? "file $1 contains: $2"; }

# count_out <substr> <expected-count>  — exact occurrence count assertion
count_out() {
    local n; n="$(grep -cF -- "$1" "$LAST_OUT_FILE")"
    [ "$n" -eq "$2" ]; _assert $? "count('$1') == $2 (was $n)"
}

note() { printf '%s\n' "$*" >> "$CUR_NOTE_FILE"; printf '%s  note: %s%s\n' "$C_DIM" "$*" "$C_RST"; }

# skip <id> <reason...>  — emit a skipped record (e.g. GPU disabled)
skip() {
    CUR_ID="$1"; shift
    local reason="$*"
    CUR_TITLE="$reason"; CUR_VERDICT='skip'; CUR_SKIP=1
    CUR_START="$(_now)"
    CUR_LOG=""; CUR_ASSERT_FILE=""; CUR_NOTE_FILE=""
    printf '%s» %s  SKIP%s  %s\n' "$C_SKIP" "$CUR_ID" "$C_RST" "$reason"
    E2E_SKIP_REASON="$reason" python3 "$E2E_ROOT/render_report.py" --emit-skip \
        --section "$E2E_SECTION" --id "$CUR_ID" >> "$E2E_RESULTS/results.jsonl"
}

end_step() {
    local end dur logrel
    end="$(_now)"
    dur="$(awk -v a="$CUR_START" -v b="$end" 'BEGIN{printf "%.3f", b-a}')"
    logrel="logs/$(basename "$CUR_LOG")"
    case "$CUR_VERDICT" in
        pass) printf '  %s✓ pass%s (%ss)\n' "$C_PASS" "$C_RST" "$dur" ;;
        fail) printf '  %s✗ FAIL%s (%ss)  see %s\n' "$C_FAIL" "$C_RST" "$dur" "$logrel" ;;
    esac
    E2E_DUR="$dur" E2E_LOGREL="$logrel" \
    python3 "$E2E_ROOT/render_report.py" --emit-step \
        --section "$E2E_SECTION" --id "$CUR_ID" --title "$CUR_TITLE" \
        --verdict "$CUR_VERDICT" \
        --asserts "$CUR_ASSERT_FILE" --notes "$CUR_NOTE_FILE" \
        >> "$E2E_RESULTS/results.jsonl"
}

# require_gpu  — call at the top of a GPU test body; returns 1 (so the caller
# `return`s) when GPU serving is disabled, after emitting one skip record.
gpu_enabled() { [ "${E2E_ENABLE_GPU:-0}" = "1" ]; }
