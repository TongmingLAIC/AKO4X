#!/bin/bash
# PreToolUse hook: blocks edits to solution/ files when .bench_pending exists.
# Forces the agent to update ITERATIONS.md before starting the next iteration.
set -euo pipefail
cat > /dev/null  # consume stdin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.bench_pending" ]; then
  echo "BLOCKED: Update ITERATIONS.md before editing solution/ files." >&2
  echo "The benchmark produced results that must be recorded first." >&2
  exit 2
fi

exit 0
