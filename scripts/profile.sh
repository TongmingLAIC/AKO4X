#!/bin/bash
# NCU profiler — profile a solution on a specific workload
# Usage: bash scripts/profile.sh --list | <workload_index> [options]
cd "$(dirname "$0")/.." || exit 1
python scripts/run_profile.py "$@"
