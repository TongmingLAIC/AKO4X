#!/bin/bash
# PreToolUse hook: blocks re-running bench.sh when .bench_pending exists.
# Prevents consecutive benchmarks without recording results.
set -euo pipefail
cat > /dev/null  # consume stdin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.bench_pending" ]; then
  echo "BLOCKED: Update ITERATIONS.md before running another benchmark." >&2
  echo "Record the previous results first, then you can bench again." >&2
  exit 2
fi

exit 0
