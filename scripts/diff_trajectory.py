"""
Trajectory Diff Tool.

Compares two benchmark trajectory entries to show per-workload and per-group
speedup changes. No benchmark execution needed — reads saved results.json files.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAJECTORY_DIR = PROJECT_ROOT / "trajectory"


def find_entries():
    """Return trajectory entries sorted by name (timestamp-prefixed → chronological)."""
    if not TRAJECTORY_DIR.is_dir():
        return []
    return sorted(
        [d for d in TRAJECTORY_DIR.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda d: d.name,
    )


def match_entry(query, entries):
    """Find a trajectory entry matching a substring query. Returns the most recent match."""
    matches = [e for e in entries if query in e.name]
    if not matches:
        print(f"Error: No trajectory entry matches '{query}'.", file=sys.stderr)
        print(f"Run with --list to see available entries.", file=sys.stderr)
        sys.exit(1)
    return matches[-1]  # most recent (sorted by timestamp)


def load_results(entry_dir):
    """Load results.json from a trajectory entry."""
    with open(entry_dir / "results.json") as f:
        return json.load(f)


def short_label(entry_dir):
    """Extract a short display label from the trajectory folder name."""
    name = entry_dir.name
    # Strip timestamp prefix (YYYYMMDD_HHMMSS_)
    parts = name.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return name


def diff_results(data_a, data_b):
    """Compare two trajectory results. Returns diff summary dict."""
    score_a = data_a.get("score") or {}
    score_b = data_b.get("score") or {}

    results_a = {}
    for def_name, traces in data_a.get("results", {}).items():
        for uuid, result in traces.items():
            results_a[uuid] = result

    results_b = {}
    for def_name, traces in data_b.get("results", {}).items():
        for uuid, result in traces.items():
            results_b[uuid] = result

    # Per-workload comparison
    all_uuids = sorted(set(results_a.keys()) | set(results_b.keys()))
    workloads = []
    for uuid in all_uuids:
        ra = results_a.get(uuid)
        rb = results_b.get(uuid)
        if ra and rb:
            lat_a = ra.get("latency_ms")
            lat_b = rb.get("latency_ms")
            sf_a = ra.get("speedup_factor")
            sf_b = rb.get("speedup_factor")
            axes = rb.get("axes", ra.get("axes", {}))
            workloads.append({
                "uuid": uuid,
                "axes": axes,
                "latency_a": lat_a,
                "latency_b": lat_b,
                "speedup_a": sf_a,
                "speedup_b": sf_b,
                "status_a": ra.get("status"),
                "status_b": rb.get("status"),
            })

    return {
        "score_a": score_a.get("final_score"),
        "score_b": score_b.get("final_score"),
        "group_a": score_a.get("group_scores", {}),
        "group_b": score_b.get("group_scores", {}),
        "group_axis": score_b.get("group_axis") or score_a.get("group_axis", ""),
        "workloads": workloads,
    }


def print_diff(label_a, label_b, diff):
    """Print a compact diff summary."""
    sa = diff["score_a"]
    sb = diff["score_b"]

    print(f"{label_a}  →  {label_b}")
    print()

    # Overall score
    if sa is not None and sb is not None:
        delta = sb - sa
        pct = (delta / sa * 100) if sa != 0 else 0
        arrow = "+" if delta >= 0 else ""
        print(f"Score: {sa:.2f}x → {sb:.2f}x  ({arrow}{delta:.2f}x, {arrow}{pct:.1f}%)")
    elif sb is not None:
        print(f"Score: ? → {sb:.2f}x")
    elif sa is not None:
        print(f"Score: {sa:.2f}x → ?")
    else:
        print("Score: ? → ?")

    # Per-group
    group_a = diff["group_a"]
    group_b = diff["group_b"]
    group_axis = diff["group_axis"]
    def _sort_key(g):
        try:
            return (0, int(g))
        except (ValueError, TypeError):
            return (1, str(g))
    all_groups = sorted(set(group_a.keys()) | set(group_b.keys()), key=_sort_key)

    if all_groups:
        print(f"\nBy {group_axis}:")
        group_deltas = []
        for g in all_groups:
            ga = group_a.get(g, {})
            gb = group_b.get(g, {})
            sfa = ga.get("speedup")
            sfb = gb.get("speedup")
            la = ga.get("latency_ms")
            lb = gb.get("latency_ms")
            if sfa is not None and sfb is not None:
                delta = sfb - sfa
                pct = (delta / sfa * 100) if sfa != 0 else 0
                group_deltas.append((g, delta, pct, sfa, sfb, la, lb))

        # Find best/worst
        best_g = max(group_deltas, key=lambda x: x[2]) if group_deltas else None
        worst_g = min(group_deltas, key=lambda x: x[2]) if group_deltas else None

        for g, delta, pct, sfa, sfb, la, lb in group_deltas:
            arrow = "+" if delta >= 0 else ""
            marker = ""
            if best_g and g == best_g[0] and best_g[2] > 0:
                marker = "  ▲ best"
            elif worst_g and g == worst_g[0] and worst_g[2] < 0:
                marker = "  ▼ worst"
            lat_str = ""
            if la is not None and lb is not None:
                lat_str = f"  ({la:.3f}→{lb:.3f}ms)"
            print(f"  {str(g):>8}  {sfa:.2f}x → {sfb:.2f}x  ({arrow}{pct:.1f}%){lat_str}{marker}")

    # Per-workload summary
    workloads = diff["workloads"]
    improved = sum(1 for w in workloads if w["speedup_a"] and w["speedup_b"] and w["speedup_b"] > w["speedup_a"])
    regressed = sum(1 for w in workloads if w["speedup_a"] and w["speedup_b"] and w["speedup_b"] < w["speedup_a"])
    unchanged = sum(1 for w in workloads if w["speedup_a"] and w["speedup_b"] and w["speedup_b"] == w["speedup_a"])
    status_changed = sum(1 for w in workloads if w["status_a"] != w["status_b"])

    parts = []
    if improved:
        parts.append(f"{improved} improved")
    if regressed:
        parts.append(f"{regressed} regressed")
    if unchanged:
        parts.append(f"{unchanged} unchanged")
    if status_changed:
        parts.append(f"{status_changed} status changed")
    if parts:
        print(f"\nPer-workload: {', '.join(parts)}")


def list_entries(entries):
    """Print available trajectory entries."""
    if not entries:
        print("No trajectory entries found.")
        return
    print(f"Trajectory entries ({len(entries)}):\n")
    for i, entry in enumerate(entries):
        data = load_results(entry)
        score = data.get("score", {})
        sf = score.get("final_score")
        label = data.get("label", "")
        score_str = f"{sf:.2f}x" if sf is not None else "?"
        print(f"  {i:>3}  {score_str:>8}  {entry.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark trajectory entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/diff_trajectory.py                  # compare last two runs
  python scripts/diff_trajectory.py iter-9           # compare iter-9 vs latest
  python scripts/diff_trajectory.py iter-9 iter-10   # compare two specific runs
  python scripts/diff_trajectory.py --list           # list available entries
""",
    )
    parser.add_argument("a", nargs="?", default=None, help="First trajectory entry (substring match)")
    parser.add_argument("b", nargs="?", default=None, help="Second trajectory entry (substring match)")
    parser.add_argument("--list", action="store_true", help="List available trajectory entries")
    args = parser.parse_args()

    entries = find_entries()

    if args.list:
        list_entries(entries)
        return

    if len(entries) < 2 and args.a is None:
        print("Error: Need at least 2 trajectory entries to compare.", file=sys.stderr)
        print("Run benchmarks with --label to create entries.", file=sys.stderr)
        sys.exit(1)

    if args.a is None:
        # Compare last two
        entry_a, entry_b = entries[-2], entries[-1]
    elif args.b is None:
        # Compare specified vs latest
        entry_a = match_entry(args.a, entries)
        entry_b = entries[-1]
        if entry_a == entry_b and len(entries) >= 2:
            entry_b = entries[-1]
            entry_a = match_entry(args.a, entries[:-1]) if args.a else entries[-2]
    else:
        entry_a = match_entry(args.a, entries)
        entry_b = match_entry(args.b, entries)

    data_a = load_results(entry_a)
    data_b = load_results(entry_b)

    label_a = short_label(entry_a)
    label_b = short_label(entry_b)

    diff = diff_results(data_a, data_b)
    print_diff(label_a, label_b, diff)


if __name__ == "__main__":
    main()
