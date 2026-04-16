#!/bin/bash
# PostToolUse hook: fires after bench.sh runs.
# Creates .bench_pending marker ONLY for real iterations (labeled + valid results).
# Skips smoke tests (--first/--index/--group without --label) and failed runs.
set -euo pipefail

INPUT="$(cat)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Extract command + output from the hook's JSON input
read_field() {
    python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception:
    print('')
    sys.exit(0)
ti = d.get('tool_input', {}) or {}
tr = d.get('tool_response', {}) or {}
if isinstance(tr, str):
    tr = {'output': tr}
key = '$1'
if key == 'command':
    print(ti.get('command', ''))
elif key == 'output':
    print(tr.get('output', '') or tr.get('stdout', ''))
elif key == 'exit_code':
    print(tr.get('exit_code', 0))
" <<< "$INPUT" 2>/dev/null || echo ""
}

COMMAND=$(read_field command)
OUTPUT=$(read_field output)

# A "real iteration" = labeled run that produced a valid final score.
# Smoke tests (--first/--index/--group) and failed runs don't require logging.
has_label=false
if [[ "$COMMAND" == *"--label"* ]] || [[ "$COMMAND" =~ [[:space:]]-l[[:space:]] ]]; then
    has_label=true
fi

is_valid_run=true
if [[ "$OUTPUT" == *"INVALID"* ]] || \
   [[ "$OUTPUT" == *"COMPILE_ERROR"* ]] || \
   [[ "$OUTPUT" == *"No results returned"* ]] || \
   [[ "$OUTPUT" != *"FINAL SCORE"* ]]; then
    is_valid_run=false
fi

if [ "$has_label" = "false" ] || [ "$is_valid_run" = "false" ]; then
    # Debug / smoke / failed run — don't require logging
    if [ "$is_valid_run" = "false" ] && [ "$has_label" = "true" ]; then
        echo "(labeled run failed — no iteration logging required; fix and retry)"
    fi
    exit 0
fi

touch "$PROJECT_ROOT/.bench_pending"

# Extract score for a more informative reminder
SCORE=$(grep -oE 'FINAL SCORE[^:]*:\s*[0-9.]+x' <<< "$OUTPUT" | grep -oE '[0-9.]+x' | head -1 || echo "?")

cat << EOF
══ Iteration complete (score: ${SCORE}) ══
Before editing solution/ or running another labeled bench, update ITERATIONS.md:
  1. Add a summary row (architectural change) OR parameter-sweep row
  2. git commit: bench(<score>): <description>

Smoke tests (--first N, --index, --group, no --label) remain allowed.
profile.sh remains allowed.
If stuck: 'rm .bench_pending' to force-clear (use sparingly).
══════════════════════════
EOF

exit 0
