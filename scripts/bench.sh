#!/bin/bash
# Quick benchmark runner
# Usage: bash scripts/bench.sh [label] [--force-baseline]
cd "$(dirname "$0")/.." || exit 1

# Parse arguments: first positional arg is label, --force-baseline is a flag
ARGS=()
LABEL=""
for arg in "$@"; do
    if [ "$arg" = "--force-baseline" ]; then
        ARGS+=(--force-baseline)
    elif [[ "$arg" == --* ]]; then
        echo "Warning: Unknown flag '$arg' ignored." >&2
    elif [ -z "$LABEL" ]; then
        LABEL="$arg"
        ARGS+=(--label "$arg")
    else
        echo "Warning: Extra positional argument '$arg' ignored (already using label '$LABEL')." >&2
    fi
done

conda run -n fi-bench --no-capture-output python scripts/run_local.py "${ARGS[@]}"
