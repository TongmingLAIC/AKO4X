"""Print the shared review prompt after every N labeled Codex benchmarks.

Called by bench.sh, so no host-specific hook API is needed. Advisory only:
the wrapper preserves the benchmark's exit status even if this helper fails.
"""

from __future__ import annotations

import sys
from pathlib import Path


def review_message(root: Path, args: list[str]) -> str:
    if not any(arg == "--label" or arg.startswith("--label=") for arg in args):
        return ""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with (root / "config.toml").open("rb") as f:
        config = tomllib.load(f).get("advisory", {})
    frequency = config.get("frequency", 3)
    if not config.get("enabled", True) or type(frequency) is not int or frequency <= 0:
        return ""
    trajectory = root / "trajectory"
    count = sum(p.is_dir() and p.name.count("_") >= 2 for p in trajectory.iterdir())
    if count and count % frequency == 0:
        return (f"Self-review checkpoint (after {count} labeled benchmarks).\n\n"
                + (root / "docs" / "iteration-review.md").read_text())
    return ""


if __name__ == "__main__":
    try:
        message = review_message(Path(__file__).resolve().parent.parent, sys.argv[1:])
        if message:
            print(message)
    except (OSError, ValueError, ImportError):
        pass  # An advisory must not turn a benchmark into a failure.
