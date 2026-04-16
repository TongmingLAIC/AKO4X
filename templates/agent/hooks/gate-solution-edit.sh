#!/bin/bash
# PreToolUse hook: blocks edits to solution/ files when .bench_pending exists.
# Forces the agent to update ITERATIONS.md before starting the next iteration.
set -euo pipefail
cat > /dev/null  # consume stdin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.bench_pending" ]; then
    echo "BLOCKED: Previous labeled benchmark hasn't been logged to ITERATIONS.md." >&2
    echo "" >&2
    echo "Fix: edit ITERATIONS.md to add a row for the previous run, then retry." >&2
    echo "Or: 'rm .bench_pending' to force-clear (use sparingly, e.g. for compile-error recovery)." >&2
    exit 2
fi

exit 0
