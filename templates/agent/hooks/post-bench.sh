#!/bin/bash
# PostToolUse hook: fires after successful bench.sh runs.
# Creates .bench_pending marker and outputs a reminder to the agent.
set -euo pipefail
cat > /dev/null  # consume stdin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

touch "$PROJECT_ROOT/.bench_pending"

cat << 'EOF'
══ Benchmark complete ══
Hooks enforce the following before your next code change:
  1. Update ITERATIONS.md (summary row + detail entry or parameter sweep row)
  2. git commit: bench(<score>): <description>
If scores have stalled for several iterations, consider BEFORE next code change:
  → Profile: bash scripts/profile.sh <index>  (identify bottleneck)
  → Search: WebSearch for optimization patterns, API docs, hardware specs
Editing solution/ or re-running bench.sh is BLOCKED until ITERATIONS.md is updated.
══════════════════════════
EOF

exit 0
