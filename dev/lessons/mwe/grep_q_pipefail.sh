#!/usr/bin/env bash
# `producer | grep -q pattern` under `set -o pipefail` can report failure
# although the pattern matched: grep -q exits at its first match, the
# producer's next write gets SIGPIPE (exit 141), and pipefail reports that.
set -o pipefail
seq 1 1000000 | grep -q '^1$'; echo "pipeline: $?  (141 = the producer's SIGPIPE, not a miss)"
seq 1 1000000 > /tmp/grep-q-mwe.txt; grep -q '^1$' /tmp/grep-q-mwe.txt; echo "file:     $?"
rm -f /tmp/grep-q-mwe.txt
