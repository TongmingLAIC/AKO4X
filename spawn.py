#!/usr/bin/env python3
"""spawn.py - Create isolated child environments for GPU kernel optimization.

Usage:
    python spawn.py --operator <name>                                 # local, auto-detect GPU, reference kernel
    python spawn.py --operator <name> --name "experiment_1"           # local with label
    python spawn.py --operator <name> --backend modal --gpu b200      # Modal B200, reference kernel
    python spawn.py --operator <name> --kernel /path/to/kernel.py     # start from your own kernel
    python spawn.py --operator <name> --dataset /path/to/dataset      # custom dataset
    python spawn.py --operator <name> --task /path/to/task.md         # custom task template
    python spawn.py                                                   # list available operators
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
BASE_DIR = PARENT_DIR.parent

# Active benchmark's template directory under templates/ (it pairs with the
# templates/skills/<this>/ skill slot of the same name). Porting AKO4X to a
# different benchmark changes this one constant plus those template dirs and
# the scripts/benchmark_adapter.py seam — see docs/porting.md.
BENCHMARK_DIR = "benchmark"


USAGE_LINE = (
    "Usage: python spawn.py --operator <name> [--dataset <path>] "
    "[--backend local|modal] [--gpu <name>] [--agent claude] "
    "[--kernel <path>] [--name <label>] [--strict-config] [--task <path>]"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create isolated child environments for GPU kernel optimization",
    )
    parser.add_argument("--operator", default="")
    parser.add_argument("--family", default="",
                        help="Closed-loop campaign family: authoritative "
                             "reference/<family>/ selector. Overrides "
                             "operator-prefix archive discovery.")
    parser.add_argument("--backend", default="local", choices=["local", "modal"])
    parser.add_argument("--gpu", default="")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--name", default="", dest="label")
    parser.add_argument("--strict-config", action="store_true",
                        help="Fail (instead of warn) when a colocated "
                             "config.toml is missing frozen evaluation.toml "
                             "[benchmark] keys (closed-loop comparability gate).")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--baseline", default="", help="Path to expert baseline Solution JSON")
    return parser.parse_args()


def _detect_gpu_rocm():
    """Probe rocm-smi for AMD Instinct GPUs. Returns (slug_lower, display_upper) or None.

    Prefers `rocm-smi --showproductname --json` (stable across ROCm versions) and
    falls back to text parsing if JSON output is unavailable. Card Series strings
    like "AMD Instinct MI300X" / "MI325X" / "MI350X" / "MI355X" yield slugs
    "mi300x" / "mi325x" / etc. The MI\\w+ regex matches the model token without
    constraining to a fixed digit count, so future Instinct families are picked
    up without code changes. Consumer Radeon cards are not auto-detected; pass
    --gpu explicitly for those.

    On mixed-vendor / mixed-card hosts (e.g. a Radeon display adapter enumerating
    before an MI300X), we scan ALL cards for the first MI-matching series rather
    than committing to the first card with any non-empty Card Series — otherwise
    a leading non-Instinct device would mask the accelerator.
    """
    if not shutil.which("rocm-smi"):
        return None
    series = None
    result = subprocess.run(
        ["rocm-smi", "--showproductname", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # Output may be prefixed with WARNING lines; locate the JSON object.
        m = re.search(r"\{.*\}", result.stdout, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}
            # Scan all cards for the first MI-matching series. Case-insensitive
            # key lookup defends against rocm-smi builds that emit "Card series"
            # instead of "Card Series".
            for card_key in sorted(data):
                card_fields = data[card_key] if isinstance(data[card_key], dict) else {}
                card_series = next(
                    (v for k, v in card_fields.items() if k.lower() == "card series"),
                    "",
                )
                if card_series and re.search(r"\bMI\w+\b", card_series):
                    series = card_series
                    break
    if series is None:
        # Fallback: text mode. Match the first MI-named card in the output;
        # case-insensitive against the "Card Series" key in case the build emits
        # a different casing.
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        m = re.search(r"Card\s+Series:\s*(.*\bMI\w+\b.*?)$",
                      result.stdout, re.MULTILINE | re.IGNORECASE)
        if not m:
            return None
        series = m.group(1).strip()
    # Slug from the model token (e.g. "AMD Instinct MI355X" → "MI355X").
    model = re.search(r"\b(MI\w+)\b", series)
    if not model:
        return None
    slug = model.group(1).upper()
    print(f"Auto-detected GPU: {series} (using --gpu {slug.lower()})")
    return slug.lower(), slug


def resolve_gpu(gpu_arg, backend):
    """Resolve GPU slug and display name. Auto-detect for local, require for modal."""
    if gpu_arg:
        gpu = gpu_arg.lower()
        return gpu, gpu.upper()

    if backend == "modal":
        sys.exit("Error: --gpu is required for modal backend")

    # Local mode: auto-detect via nvidia-smi.
    # On hosts with nvidia-smi present, behavior is identical to prior versions —
    # the entire NVIDIA branch below is unchanged. Only when nvidia-smi is
    # entirely absent do we fall through to the ROCm probe.
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        gpu_full = result.stdout.strip().split("\n")[0].strip()
        if not gpu_full:
            sys.exit("Error: No GPU detected. Use --gpu <name>.")

        # Data-center cards first (A100 / H100 / B200 / L40S / T4 / V100);
        # fall back to consumer naming (RTX / GTX / RX <model>) which has a
        # space between the brand and the model number.
        dc = re.search(r"\b([ABHLTV]\d{2,3}[A-Z]*)\b", gpu_full)
        if dc:
            slug = dc.group(1)
        else:
            consumer = re.search(r"\b(RTX|GTX|RX)\s*(\d{3,4}[A-Z]*)\b", gpu_full)
            if consumer:
                slug = f"{consumer.group(1)}{consumer.group(2)}"
            else:
                sys.exit(f"Error: Could not identify GPU model from '{gpu_full}'. Use --gpu <name>.")

        gpu = slug.lower()
        print(f"Auto-detected GPU: {gpu_full} (using --gpu {gpu})")
        return gpu, slug.upper()

    # nvidia-smi absent → try ROCm. This is the only new exit path.
    detected = _detect_gpu_rocm()
    if detected is not None:
        return detected

    sys.exit("Error: No GPU specified and nvidia-smi not found. Use --gpu <name>.")


def load_agent_config(agent):
    """Load agent config JSON, return (config_dict, task_filename)."""
    config_path = PARENT_DIR / "templates" / "agent" / f"{agent}.json"
    if not config_path.is_file():
        sys.exit(f"Error: Agent config not found: {config_path}")
    config = json.loads(config_path.read_text())
    return config, config["task_filename"]


def load_evaluation_config(op_type):
    """Load benchmark defaults + per-op_type overrides.

    Reads templates/benchmark/evaluation.toml (the BENCHMARK_DIR slot); merges
    [default] (benchmark-bound bench params + tolerance defaults) with the
    per-op_type section if present. Returns a dict of key-value pairs to
    include in the child's config.toml [benchmark] section.
    """
    eval_path = PARENT_DIR / "templates" / BENCHMARK_DIR / "evaluation.toml"
    if not eval_path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    try:
        with open(eval_path, "rb") as f:
            eval_config = tomllib.load(f)
    except Exception:
        return {}
    merged = {}
    merged.update(eval_config.get("default", {}))
    merged.update(eval_config.get(op_type, {}))
    return merged


def _toml_scalars(d):
    """Format a flat dict as TOML `key = value` lines (bool before int —
    bool is an int subclass; non-scalars dropped). Shared by the generated
    and colocated config paths so their [benchmark] emission can't drift."""
    lines = []
    for key, value in d.items():
        if isinstance(value, bool):
            lines.append(f'{key} = {"true" if value else "false"}')
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, float):
            lines.append(f'{key} = {value}')
        elif isinstance(value, int):
            lines.append(f'{key} = {value}')
    return lines


def resolve_dataset(dataset_arg):
    """Resolve dataset path from arg or environment variable."""
    dataset_path = dataset_arg or os.environ.get("AKO_DATASET_PATH") or os.environ.get("FIB_DATASET_PATH", "")
    if not dataset_path:
        print("Error: No dataset path specified.")
        sys.exit("Set AKO_DATASET_PATH (FIB_DATASET_PATH also works) or use --dataset to specify the path to the benchmark trace set.")
    dataset = Path(dataset_path)
    if not dataset.is_dir():
        sys.exit(f"Error: Dataset directory does not exist: {dataset.resolve()}\n"
                 f"Check the path and try again, or set AKO_DATASET_PATH.")
    if not (dataset / "definitions").is_dir():
        sys.exit(f"Error: Dataset directory exists ({dataset.resolve()}) but has no 'definitions/' subdirectory.\n"
                 f"Expected structure: {dataset}/definitions/<category>/<operator>.json\n"
                 f"Is this a valid benchmark trace set? (The default benchmark expects the flashinfer-trace layout.)")
    return dataset


def list_operators(dataset_path):
    """Print available operators (only those with workload files) and exit."""
    print("Available operators:")
    print()
    count = 0
    for def_file in sorted(dataset_path.glob("definitions/*/*.json")):
        op_name = def_file.stem
        op_type = def_file.parent.name
        workloads_file = dataset_path / "workloads" / op_type / f"{op_name}.jsonl"
        if workloads_file.is_file():
            print(f"  {op_name}  ({op_type})")
            count += 1
    print()
    print(f"{count} operators available.")
    print(USAGE_LINE)
    sys.exit(0)


def discover_operator(dataset_path, operator):
    """Find operator definition and workloads files. Returns (definition_path, workloads_path, op_type)."""
    # Sort for deterministic selection: filesystem glob order is undefined, and
    # operator names that appear under two op_type categories would otherwise
    # produce non-reproducible op_type picks across hosts.
    matches = sorted(dataset_path.glob(f"definitions/*/{operator}.json"))
    if not matches:
        print(f"Error: Operator '{operator}' not found in {dataset_path}/definitions/")
        sys.exit("Run without --operator to list available operators.")
    if len(matches) > 1:
        print(
            f"Warning: operator '{operator}' is defined under multiple op_types "
            f"({[m.parent.name for m in matches]}); picking '{matches[0].parent.name}'. "
            f"Rename the duplicates to disambiguate.",
            file=sys.stderr,
        )
    definition_path = matches[0]
    op_type = definition_path.parent.name
    workloads_path = dataset_path / "workloads" / op_type / f"{operator}.jsonl"
    if not workloads_path.is_file():
        sys.exit(f"Error: Workloads file not found: {workloads_path}")
    return definition_path, workloads_path, op_type


def validate_kernel_path(kernel_path):
    """Verify --kernel path exists when provided."""
    kp = Path(kernel_path)
    if not kp.exists():
        sys.exit(f"Error: Kernel path not found: {kernel_path}")


def discover_prior_lessons(operator, campaign_family=""):
    """Find the reference archive for this spawn.

    If `campaign_family` is given (closed-loop: master passes `--family`), it
    is AUTHORITATIVE — use exactly `reference/<campaign_family>/` if it exists,
    else no prior archive. Never fall back to operator-prefix scan when a
    family is set: that scan keys on the operator name alone and would re-bind
    a `<op>-<gpu>` campaign to the plain `<op>` dir (the documented "new
    hardware = new family" split), drifting prior-lessons + archive_seed_path
    off the campaign.

    With no `campaign_family` (manual / non-closed-loop spawn), match by
    either (a) exact equality — current convention is `family == operator`
    kebab-cased, so `reference/mla-paged-decode-h16-ckv512-kpe64-ps1/`
    matches operator `mla_paged_decode_h16_ckv512_kpe64_ps1` — or
    (b) underscore-prefix — legacy shape-shared archive, e.g.
    `reference/dsa-sparse-attention/` matches
    `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

    Trailing underscore on the prefix form is required so `reference/gdn/`
    (hypothetical) wouldn't accidentally shadow `reference/gdn-decode/` for
    `gdn_decode_*` operators. Longest match wins (exact match naturally
    beats any shorter prefix).
    """
    ref_dir = PARENT_DIR / "reference"
    if not ref_dir.is_dir():
        return None
    if campaign_family:
        fam_dir = ref_dir / campaign_family
        return fam_dir if fam_dir.is_dir() else None
    matches = []
    for sub in ref_dir.iterdir():
        if not sub.is_dir():
            continue
        family = sub.name.replace("-", "_")
        if family == operator or operator.startswith(family + "_"):
            matches.append((len(family), sub))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def copy_prior_lessons(source_dir, child_dir):
    """Copy reference archive into child/docs/prior/.

    Copies:
      - *.md (README anchor pointer; optional TRAPS.md cross-variant gotchas)
      - baseline.json (frozen canonical per-workload reference latencies —
        the denominator variant result.json speedups are computed against)
      - variants/<name>/{kernel.py, config.toml, result.json, variance.json}
        (working prior kernels; each kernel.py header carries the variant's
        lessons and dead-ends per templates/agent/lessons-convention.md)

    Returns a short summary list of top-level entries copied, for logging.
    """
    prior_dir = child_dir / "docs" / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for md in sorted(source_dir.glob("*.md")):
        shutil.copy2(md, prior_dir / md.name)
        copied.append(md.name)
    baseline_src = source_dir / "baseline.json"
    if baseline_src.is_file():
        shutil.copy2(baseline_src, prior_dir / "baseline.json")
        copied.append("baseline.json")
    variants_src = source_dir / "variants"
    if variants_src.is_dir():
        variants_dst = prior_dir / "variants"
        if variants_dst.exists():
            shutil.rmtree(variants_dst)
        shutil.copytree(variants_src, variants_dst)
        n = sum(1 for _ in variants_dst.iterdir() if _.is_dir())
        copied.append(f"variants/ ({n} variants)")
    return copied


def discover_expert_baseline(dataset_path, operator, op_type, explicit_path=""):
    """Find an expert baseline solution JSON for the operator.

    FlashInfer-Trace layout (the convention this targets primarily):
        solutions/<author>/<op_type>/<operator>/<solution>.json

    `<author>` is the identity of whoever contributed the solution. The
    FlashInfer-Trace dataset card describes solutions as "contributed by
    either human experts or autonomous agent systems"; in practice upstream
    `flashinfer-ai/flashinfer-trace` populates both kinds in parallel —
    a `baseline/` dir (the human-curated expert reference) alongside
    model-keyed dirs holding agent-generated attempts
    (`claude-opus-4-1-20250805`, `gemini-2.5-pro`, `gpt-5-2025-08-07`,
    `gpt-o3`, `llama1b`, ...). Forks like `mlsys26-contest` follow the
    same layout (typically with just `baseline` populated).

    spawn.py treats authors as opaque: it scans
    `solutions/*/{op_type}/{operator}/` across all authors and returns the
    lex-first match. `baseline` sorts ahead of every model name we've seen,
    so when it's present we naturally pick the curated expert as the
    bench denominator — same behavior on upstream and on forks. Pass
    `--baseline <path>` to force a specific solution (e.g. to compare
    against a model-generated attempt instead).

    If `explicit_path` is provided, use it (bypasses discovery entirely).
    Otherwise warn — but still pick lex-first — when multiple authors
    contribute solutions for the same operator.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            sys.exit(f"Error: Baseline not found: {explicit_path}")
        return p
    solutions_dir = dataset_path / "solutions"
    if not solutions_dir.is_dir():
        return None
    matches = sorted(solutions_dir.glob(f"*/{op_type}/{operator}/*.json"))
    if not matches:
        return None
    picked = matches[0]
    picked_author = picked.relative_to(solutions_dir).parts[0]
    # Only warn when ambiguity is consequential: we picked a non-baseline
    # author. Upstream's common case (every op has baseline + 4–5 model-keyed
    # contributions) would otherwise fire on every spawn and become noise.
    if len(matches) > 1 and picked_author != "baseline":
        authors = sorted({m.relative_to(solutions_dir).parts[0] for m in matches})
        print(
            f"Warning: operator {operator!r} has expert solutions under "
            f"{len(authors)} author(s) ({authors}) and no `baseline` author; "
            f"picking lex-first {picked.relative_to(dataset_path)}. Pass "
            f"--baseline <path> to disambiguate.",
            file=sys.stderr,
        )
    return picked


def infer_language(kernel_path):
    """Infer language from kernel file extension. Returns (language, entry_point).

    `.hip` is the AMD ROCm equivalent of `.cu` — same role (device kernels
    compiled by hipcc) and the same convention of a sibling `binding.cpp` /
    `binding.py` providing the host-side hook. The actual ROCm build path
    (compile + link) is a downstream concern of the benchmark adapter; this
    function only classifies the kernel file so downstream tooling can route.
    """
    ext_map = {
        ".cu": ("cuda", "binding.py::kernel"),
        ".cpp": ("cpp", "binding.py::kernel"),
        ".hip": ("hip", "binding.py::kernel"),  # ROCm device kernel; mirrors .cu's host-binding convention
        ".py": ("python", "kernel.py::run"),
    }
    kp = Path(kernel_path)
    if kp.is_dir():
        # Check for .cu or .cpp files in directory
        for ext, (lang, ep) in ext_map.items():
            if list(kp.glob(f"*{ext}")):
                return lang, ep
        return "python", "kernel.py::run"
    ext = kp.suffix.lower()
    lang, ep = ext_map.get(ext, ("python", "kernel.py::run"))
    return lang, ep


def make_child_name(label):
    """Generate child directory name.

    With label: ako4x-run-{label} (error if exists).
    Without label: ako4x-run-{YYYYMMDD_HHMMSS}.
    """
    from datetime import datetime

    if label:
        return f"ako4x-run-{label}"
    return f"ako4x-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def render_template(task_text, placeholders):
    """Render task template with placeholder substitution.

    Block placeholders (alone on a line) replace the entire line.
    Inline placeholders are substituted within lines.
    """
    block_keys = {
        "{{PRIOR_LESSONS_BLOCK}}",
    }
    inline_keys = [
        "{{OPERATOR}}", "{{GPU_NAME}}",
    ]

    lines = []
    for line in task_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped in block_keys:
            try:
                content = placeholders[stripped]
            except KeyError:
                raise KeyError(f"Missing placeholder value for {stripped!r} in template rendering")
            if content:
                lines.append(content)
                if not content.endswith("\n"):
                    lines.append("\n")
            # Empty content consumes the placeholder line entirely (no blank
            # line left behind — callers control spacing via placeholder text).
        else:
            for key in inline_keys:
                if key in line:
                    line = line.replace(key, placeholders[key])
            lines.append(line)
    return "".join(lines)


def populate_child(child_dir, *, operator, op_type, gpu, backend, kernel_path,
                   definition_path, workloads_path, dataset_path, agent_config,
                   expert_baseline_path=None, prior_lessons_dir=None,
                   strict_config=False):
    """Create child directory structure and populate with files."""
    # Create directories — flat solution/ (no language subdirectory)
    for subdir in [".claude", "docs", "scripts", "solution"]:
        (child_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy common files
    shutil.copy2(PARENT_DIR / "templates" / "gitignore", child_dir / ".gitignore")
    shutil.copy2(definition_path, child_dir / "docs" / "definition.json")
    shutil.copy2(workloads_path, child_dir / "docs" / "workloads.jsonl")
    shutil.copy2(PARENT_DIR / "scripts" / "pack_solution.py", child_dir / "scripts" / "pack_solution.py")
    shutil.copy2(PARENT_DIR / "scripts" / "benchmark_adapter.py", child_dir / "scripts" / "benchmark_adapter.py")
    shutil.copy2(PARENT_DIR / "scripts" / "bench_utils.py", child_dir / "scripts" / "bench_utils.py")
    shutil.copy2(PARENT_DIR / "scripts" / "CLAUDE.md", child_dir / "scripts" / "CLAUDE.md")
    # Language references live as SKILLs at .claude/skills/<skill>/. The
    # pure-PyTorch fallback (language=python) has no separate SKILL — its
    # convention is inlined under the benchmark SKILL's entry_point
    # per-language list.

    # Copy iterations template
    shutil.copy2(PARENT_DIR / "templates" / "iterations.md", child_dir / "ITERATIONS.md")

    # Determine language and entry_point for config.toml
    language = "python"
    entry_point = "kernel.py::run"

    if not kernel_path:
        # No --kernel: extract reference from definition.json → solution/kernel.py
        definition = json.loads(definition_path.read_text())
        ref = definition.get("reference", "")
        if not ref:
            sys.exit("Error: No reference field in definition.json")
        (child_dir / "solution" / "kernel.py").write_text(ref)
    else:
        # --kernel provided: copy file(s) to solution/
        kp = Path(kernel_path)
        if kp.is_dir():
            # Filter to source extensions (match single-file sibling branch
            # below) so result.json / variance.json / other run metadata don't
            # pollute solution/ and cascade into every trajectory snapshot.
            for f in kp.iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in (".py", ".cu", ".cpp", ".h", ".hpp", ".cuh", ".toml"):
                    if f.name != "config.toml":
                        shutil.copy2(f, child_dir / "solution" / f.name)
        else:
            # Rename the entry-point source to match config.toml's entry_point
            # filename (e.g. kernel.py::run), otherwise pack_solution fails with
            # "Entry source file 'kernel.py' not found in sources".
            dest_name = infer_language(kernel_path)[1].split("::", 1)[0] if kp.suffix.lower() == ".py" else kp.name
            shutil.copy2(kp, child_dir / "solution" / dest_name)
            # Copy sibling files (binding.py, other source files). Skip any
            # sibling whose name collides with the renamed entry destination:
            # otherwise a sibling literally named `kernel.py` (distinct from the
            # entry source `mykernel.py`) would silently overwrite the just-renamed
            # entry, spawning the child with a kernel the user never asked for.
            for sibling in kp.parent.iterdir():
                if sibling.is_file() and sibling != kp:
                    ext = sibling.suffix.lower()
                    if ext in (".py", ".cu", ".cpp", ".h", ".hpp", ".cuh", ".toml"):
                        if sibling.name == "config.toml":
                            continue
                        if sibling.name == dest_name:
                            print(f"Warning: sibling {sibling} would overwrite the "
                                  f"renamed entry source ({dest_name}); skipping.",
                                  file=sys.stderr)
                            continue
                        shutil.copy2(sibling, child_dir / "solution" / sibling.name)

        # If config.toml is colocated with the kernel it is the variant's
        # authoritative BUILD config (language / entry_point /
        # destination_passing_style) and carries forward any [benchmark]
        # keys it set. But it must NOT silently run with a degraded
        # [benchmark]: an archived/minimal variant config (e.g. one that
        # wrote only use_isolated_runner) would otherwise fall back to the
        # flashinfer_bench library defaults instead of this benchmark's
        # frozen evaluation.toml params — a non-comparable score. Admission
        # rule: keep what the colocated config carries, COMPLETE any missing
        # frozen evaluation.toml keys (warn), or fail loud under
        # --strict-config. The child config is regenerated (not copied +
        # appended) so a carried duplicate key can't desync the [benchmark]
        # table and runtime-owned keys are always spawn-current.
        colocated_config = (kp if kp.is_dir() else kp.parent) / "config.toml"
        if colocated_config.is_file():
            print(f"Note: Using config.toml from {colocated_config.resolve()}")
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(colocated_config, "rb") as f:
                colocated = tomllib.load(f)
            cbuild = colocated.get("build", {})
            language = cbuild.get("language", language)
            entry_point = cbuild.get("entry_point", entry_point)
            dps = bool(cbuild.get("destination_passing_style", False))
            colocated_bench = colocated.get("benchmark", {})

            eval_params = load_evaluation_config(op_type)
            missing = sorted(k for k in eval_params if k not in colocated_bench)
            if missing:
                eval_src = PARENT_DIR / "templates" / BENCHMARK_DIR / "evaluation.toml"
                detail = (
                    f"colocated config.toml [benchmark] is missing "
                    f"{len(missing)} frozen evaluation.toml key(s): "
                    f"{', '.join(missing)} (source: {eval_src})"
                )
                if strict_config:
                    sys.exit(
                        f"Error: {detail}. Refusing to spawn a non-comparable "
                        f"child (--strict-config). Regenerate the variant's "
                        f"config.toml or add the keys explicitly."
                    )
                print(
                    f"WARNING: {detail} — completing from evaluation.toml so "
                    f"the run stays comparable; the colocated config's own "
                    f"[benchmark] values are kept where present.",
                    file=sys.stderr,
                )

            # Precedence (low→high): evaluation.toml frozen defaults <
            # colocated's carried [benchmark] (carry-forward authority) <
            # spawn-time runtime-owned keys (always overwritten — a
            # copied/moved variant must not keep a stale absolute
            # archive_seed_path or a backend from another spawn).
            merged_bench = {**eval_params, **colocated_bench}
            merged_bench["backend"] = backend
            if prior_lessons_dir:
                merged_bench["archive_seed_path"] = str(
                    (prior_lessons_dir / "baseline.json").resolve()
                )
            # Isolated runner is the safe default (some variants alias
            # module-level state across workloads in a persistent runner);
            # evaluation.toml [default] normally sets it — this only matters
            # if both eval and the colocated config are silent.
            merged_bench.setdefault("use_isolated_runner", True)

            dataset_line = (
                f'dataset_path = "{dataset_path.resolve()}"\n'
                if backend == "local" else ""
            )
            config_toml = (
                f'[solution]\n'
                f'name = "{operator}-solution"\n'
                f'definition = "{operator}"\n'
                f'author = "user"\n'
                f'\n'
                f'[build]\n'
                f'gpu = "{gpu}"\n'
                f'{dataset_line}'
                f'\n'
                f'# ── Agent-configurable (update to match your kernel) ──\n'
                f'language = "{language}"\n'
                f'entry_point = "{entry_point}"\n'
                f'destination_passing_style = {"true" if dps else "false"}\n'
                f'\n'
                f'[benchmark]\n'
                + "\n".join(_toml_scalars(merged_bench)) + "\n"
            )
            (child_dir / "config.toml").write_text(config_toml)
        else:
            # Infer language from file extension
            language, entry_point = infer_language(kernel_path)

    # Generate config.toml (unless colocated config was already copied)
    config_path = child_dir / "config.toml"
    if not config_path.exists():
        dataset_line = f'dataset_path = "{dataset_path.resolve()}"\n' if backend == "local" else ""

        # Benchmark-bound defaults + per-op_type overrides come from the
        # benchmark's evaluation.toml.
        bench_params = load_evaluation_config(op_type)
        # Spawn-time runtime metadata. bench_utils.py reads `backend` for the
        # baseline-staleness check, and `archive_seed_path` to know where to
        # auto-promote a first-time reference profile.
        bench_params["backend"] = backend
        if prior_lessons_dir:
            archive_seed = (prior_lessons_dir / "baseline.json").resolve()
            bench_params["archive_seed_path"] = str(archive_seed)

        # Format [benchmark] section as TOML
        bench_lines = _toml_scalars(bench_params)

        config_toml = (
            f'[solution]\n'
            f'name = "{operator}-solution"\n'
            f'definition = "{operator}"\n'
            f'author = "user"\n'
            f'\n'
            f'[build]\n'
            f'gpu = "{gpu}"\n'
            f'{dataset_line}'
            f'\n'
            f'# ── Agent-configurable (update to match your kernel) ──\n'
            f'language = "{language}"\n'
            f'entry_point = "{entry_point}"\n'
            f'destination_passing_style = false\n'
            f'\n'
            f'[benchmark]\n'
            + "\n".join(bench_lines) + "\n"
        )
        config_path.write_text(config_toml)

    # Ensure [advisory] section exists (applies to both the colocated-copy
    # branch above and the generated branch). The advisory hook reads
    # [advisory] frequency / enabled at runtime; missing section means it
    # silently falls back to defaults, which is fine but hides the
    # configurability from users.
    config_text = config_path.read_text() if config_path.exists() else ""
    # Match the literal section header on its own line; the bare substring check
    # would suppress the append if "[advisory]" appeared anywhere else (a
    # comment, a quoted string, a key value), silently leaving the section
    # absent.
    if not re.search(r"^\[advisory\]\s*$", config_text, re.MULTILINE):
        with open(config_path, "a") as f:
            f.write('\n[advisory]\n')
            f.write('# Advisory review hook: prints a self-review prompt to stderr\n')
            f.write('# after every N labeled benches. See templates/agent/hooks/\n')
            f.write('# advisory-review.sh. No gate, no blocking.\n')
            f.write('frequency = 3\n')
            f.write('enabled = true\n')

    # Baseline seed copy: the archive at reference/<family>/baseline.json is
    # the single golden. When present, copy into child as the denominator;
    # when absent, child's first reference profile auto-promotes there (the
    # archive_seed_path injected into [benchmark] above tells save_baseline
    # where to write). Either way the archive is never overwritten by a run.
    if prior_lessons_dir:
        archive_seed = prior_lessons_dir / "baseline.json"
        if archive_seed.is_file():
            shutil.copy2(archive_seed, child_dir / "baseline.json")
            print(f"Seeded baseline.json from reference/{prior_lessons_dir.name}/baseline.json")
        else:
            print(f"No archive baseline at reference/{prior_lessons_dir.name}/baseline.json — "
                  f"child's first reference profile will auto-promote there.")

    # Copy expert baseline if available
    if expert_baseline_path:
        shutil.copy2(expert_baseline_path, child_dir / "expert_baseline.json")

    # Copy prior-session lessons archive if available
    if prior_lessons_dir:
        copied = copy_prior_lessons(prior_lessons_dir, child_dir)
        if copied:
            print(f"Copied prior-session notes from reference/{prior_lessons_dir.name}/ "
                  f"({len(copied)} files: {', '.join(copied)})")

    # Generate backend-specific bench.sh / profile.sh / sanitize.sh and copy runner scripts
    if backend == "local":
        bench_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\npython scripts/run_local.py "$@"\n'
        profile_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\npython scripts/run_local_profile.py "$@"\n'
        sanitize_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\npython scripts/run_local_sanitize.py "$@"\n'
        shutil.copy2(PARENT_DIR / "scripts" / "run_local.py", child_dir / "scripts" / "run_local.py")
        shutil.copy2(PARENT_DIR / "scripts" / "run_local_profile.py", child_dir / "scripts" / "run_local_profile.py")
        shutil.copy2(PARENT_DIR / "scripts" / "run_local_sanitize.py", child_dir / "scripts" / "run_local_sanitize.py")
    else:
        # Venv-discovery prelude: AKO4X's modal CLI typically lives at
        # <workspace>/.venv/bin/modal but the user shell may not have venv
        # activated. If `modal` isn't on PATH, walk up looking for a venv.
        # Idempotent — short-circuits when modal is already discoverable.
        modal_prelude = (
            'if ! command -v modal >/dev/null 2>&1; then\n'
            '    for cand in .venv ../.venv ../../.venv ../../../.venv; do\n'
            '        if [ -x "$cand/bin/modal" ]; then\n'
            '            export PATH="$(cd "$cand/bin" && pwd):$PATH"\n'
            '            break\n'
            '        fi\n'
            '    done\n'
            'fi\n'
        )
        bench_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\n' + modal_prelude + 'modal run scripts/run_modal.py "$@"\n'
        profile_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\n' + modal_prelude + 'modal run scripts/run_modal_profile.py "$@"\n'
        sanitize_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\n' + modal_prelude + 'modal run scripts/run_modal_sanitize.py "$@"\n'
        shutil.copy2(PARENT_DIR / "scripts" / "run_modal.py", child_dir / "scripts" / "run_modal.py")
        shutil.copy2(PARENT_DIR / "scripts" / "run_modal_profile.py", child_dir / "scripts" / "run_modal_profile.py")
        shutil.copy2(PARENT_DIR / "scripts" / "run_modal_sanitize.py", child_dir / "scripts" / "run_modal_sanitize.py")

    shutil.copy2(PARENT_DIR / "scripts" / "diff_trajectory.py", child_dir / "scripts" / "diff_trajectory.py")

    diff_sh = '#!/bin/bash\ncd "$(dirname "$0")/.." || exit 1\npython scripts/diff_trajectory.py "$@"\n'
    for name, content in [("bench.sh", bench_sh), ("profile.sh", profile_sh),
                          ("sanitize.sh", sanitize_sh), ("diff.sh", diff_sh)]:
        p = child_dir / "scripts" / name
        p.write_text(content)
        p.chmod(0o755)

    # Copy advisory hook + slash commands into child env.
    # Hook fires after each labeled bench, printing a self-review prompt to
    # stderr (see templates/agent/hooks/advisory-review.sh). /review slash
    # command shares the same prompt file. Soft protocol only — no gating.
    hooks_src = PARENT_DIR / "templates" / "agent" / "hooks"
    hooks_dst = child_dir / ".claude" / "hooks"
    if hooks_src.is_dir():
        shutil.copytree(hooks_src, hooks_dst)
        for hook_file in hooks_dst.iterdir():
            if hook_file.is_file():
                hook_file.chmod(0o755)

    commands_src = PARENT_DIR / "templates" / "agent" / "commands"
    commands_dst = child_dir / ".claude" / "commands"
    if commands_src.is_dir():
        shutil.copytree(commands_src, commands_dst)

    # Copy SKILLs into child/.claude/skills/ for Claude Code progressive-disclosure discovery.
    # Sub finds skills by name + description (frontmatter); body + supporting docs load on demand.
    skills_src = PARENT_DIR / "templates" / "skills"
    skills_dst = child_dir / ".claude" / "skills"
    if skills_src.is_dir():
        shutil.copytree(skills_src, skills_dst)

    # Generate .claude/settings.local.json (permissions + advisory hook).
    # The advisory hook is soft — it prints a self-review prompt to stderr
    # after every N labeled benches but never blocks. See templates/agent/
    # hooks/advisory-review.sh.
    settings = {
        "permissions": agent_config["permissions"],
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "if": "Bash(*bench.sh*)",
                        "command": ".claude/hooks/advisory-review.sh",
                    }],
                },
            ],
        },
    }
    settings_path = child_dir / ".claude" / "settings.local.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def init_git(child_dir, operator, backend):
    """Initialize git repo in child directory. Skips gracefully if git is unavailable."""
    if not shutil.which("git"):
        print("Warning: git not found, skipping repository initialization.", file=sys.stderr)
        return

    try:
        msg = f"Initial commit (spawned from AKO4X, operator={operator}, backend={backend})"
        subprocess.run(["git", "init", "-q"], cwd=child_dir, check=True)
        # Set repo-local user config so commit works without global git config
        subprocess.run(["git", "config", "user.name", "ako4x"], cwd=child_dir, check=True)
        subprocess.run(["git", "config", "user.email", "ako4x@local"], cwd=child_dir, check=True)
        subprocess.run(["git", "add", "-A"], cwd=child_dir, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=child_dir, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Warning: git initialization failed ({e}), continuing without git.", file=sys.stderr)


def main():
    args = parse_args()

    # Resolve dataset + list operators (GPU not needed for listing)
    dataset_path = resolve_dataset(args.dataset)
    if not args.operator:
        list_operators(dataset_path)

    # Resolve GPU (after listing, so `python spawn.py` with no args doesn't
    # demand a GPU just to show what's available)
    gpu, gpu_name = resolve_gpu(args.gpu, args.backend)

    # Load agent config
    agent_config, task_filename = load_agent_config(args.agent)

    # Resolve task template path
    task_path = PARENT_DIR / "templates" / "task.md"
    if not task_path.is_file():
        sys.exit(f"Error: Task template not found: {task_path}")

    # Discover operator
    definition_path, workloads_path, op_type = discover_operator(dataset_path, args.operator)
    operator = args.operator

    print(f"Operator: {operator} ({op_type})")
    print(f"Definition: {definition_path}")
    print(f"Workloads: {workloads_path}")
    print(f"Dataset: {dataset_path}")

    # Validate
    if args.kernel:
        validate_kernel_path(args.kernel)

    # Discover expert baseline
    expert_baseline = discover_expert_baseline(dataset_path, operator, op_type, args.baseline)
    if expert_baseline:
        print(f"Expert baseline: {expert_baseline}")

    # Discover prior-session lessons archive (reference/<family>/). When the
    # closed-loop master passes --family it is authoritative; otherwise fall
    # back to operator-prefix discovery for manual spawns.
    prior_lessons_dir = discover_prior_lessons(operator, args.family)
    if prior_lessons_dir:
        print(f"Prior-session archive: {prior_lessons_dir}")

    # Build the prior-lessons block. Empty string renders as a blank line;
    # non-empty points agents at the copied archive so they don't retrace
    # dead-ends documented in a previous session on the same operator family.
    if prior_lessons_dir:
        # H3 sub-section under ## Operator; trailing blank line separates
        # from the next H2 heading. Empty block consumes the placeholder
        # line entirely (see render_template), so the no-prior case keeps
        # the single blank above without adding one below.
        prior_lessons_block = (
            f"### Prior-session kernels ({prior_lessons_dir.name})\n\n"
            f"`docs/prior/` holds **working kernel variants** from earlier "
            f"optimizations of this operator family. Read `docs/prior/README.md` "
            f"first — it identifies the current anchor variant and its fallbacks. "
            f"The fastest starting point is `docs/prior/variants/<anchor>/kernel.py`; "
            f"its header comment describes architecture, key lessons, and "
            f"dead-ends tried on that variant.\n\n"
        )
    else:
        prior_lessons_block = ""

    placeholders = {
        "{{OPERATOR}}": operator,
        "{{GPU_NAME}}": gpu_name,
        "{{PRIOR_LESSONS_BLOCK}}": prior_lessons_block,
    }

    # Determine child directory name
    child_name = make_child_name(args.label)
    child_dir = BASE_DIR / child_name
    if child_dir.exists():
        sys.exit(f"Error: Directory already exists: {child_dir}\n"
                 f"Choose a different --name or remove the existing directory.")

    # Populate child environment
    populate_child(
        child_dir,
        operator=operator,
        op_type=op_type,
        gpu=gpu,
        backend=args.backend,
        kernel_path=args.kernel,
        definition_path=definition_path,
        workloads_path=workloads_path,
        dataset_path=dataset_path,
        agent_config=agent_config,
        expert_baseline_path=expert_baseline,
        prior_lessons_dir=prior_lessons_dir,
        strict_config=args.strict_config,
    )

    # Render and write task file
    task_text = task_path.read_text()
    rendered = render_template(task_text, placeholders)
    (child_dir / task_filename).write_text(rendered)

    # Git init
    init_git(child_dir, operator, args.backend)

    # Summary
    print()
    print("===== Child environment created =====")
    print(f"  Path:     {child_dir}")
    print(f"  Operator: {operator}")
    print(f"  Backend:  {args.backend}")
    print(f"  GPU:      {gpu_name}")
    print(f"  Dataset:  {dataset_path}")
    if args.kernel:
        print(f"  Kernel:   {args.kernel}")
    print()
    print("Next steps:")
    print(f"  cd {child_dir}")
    print(f"  claude")
    print()
    print("  # Then send a prompt to start optimizing, e.g.:")
    print('  > Read CLAUDE.md and optimize the kernel. Try your best.')
    print("=======================================")


if __name__ == "__main__":
    main()
