#!/bin/bash
# PreToolUse hook: blocks labeled (real-iteration) bench.sh when .bench_pending exists.
# Smoke tests (--first/--index/--group, no --label) are always allowed.
set -euo pipefail

INPUT="$(cat)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse command from JSON input
COMMAND=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print((d.get('tool_input', {}) or {}).get('command', ''))
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

# Smoke tests (no --label) are always allowed — they're not iterations.
if [[ "$COMMAND" != *"--label"* ]] && [[ ! "$COMMAND" =~ [[:space:]]-l[[:space:]] ]]; then
    exit 0
fi

if [ -f "$PROJECT_ROOT/.bench_pending" ]; then
    echo "BLOCKED: Previous labeled benchmark hasn't been logged to ITERATIONS.md." >&2
    echo "" >&2
    echo "Fix: edit ITERATIONS.md to add a row for the previous run, then retry." >&2
    echo "Or: 'rm .bench_pending' to force-clear (use sparingly)." >&2
    echo "" >&2
    echo "Note: smoke tests (--first N, --index, --group WITHOUT --label) are always allowed." >&2
    exit 2
fi

exit 0
