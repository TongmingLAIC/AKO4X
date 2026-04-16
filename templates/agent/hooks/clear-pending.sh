#!/bin/bash
# PostToolUse hook: clears .bench_pending marker when ITERATIONS.md is modified.
# Triggers from Edit, MultiEdit, or Write tools. This unblocks the next iteration.
set -euo pipefail
cat > /dev/null  # consume stdin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

rm -f "$PROJECT_ROOT/.bench_pending"

exit 0
