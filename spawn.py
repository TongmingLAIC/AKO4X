#!/usr/bin/env python3
"""spawn.py - Create isolated child environments for GPU kernel optimization.

Usage:
    python spawn.py --operator <name>                                 # local, auto-detect GPU, from scratch
    python spawn.py --operator <name> --name "experiment_1"           # local with label
    python spawn.py --operator <name> --backend modal --gpu b200      # Modal B200, from scratch
    python spawn.py --operator <name> --mode existing --kernel /path/to/kernel.py
    python spawn.py --operator <name> --dataset /path/to/dataset      # custom dataset
    python spawn.py --operator <name> --task /path/to/task.md         # custom task template
    python spawn.py --operator <name> --hints /path/to/hints.md      # custom hints
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

# Import generate_context from scripts/
sys.path.insert(0, str(PARENT_DIR / "scripts"))
from generate_context import generate_context


USAGE_LINE = (
    "Usage: python spawn.py --operator <name> [--dataset <path>] "
    "[--mode scratch|existing] [--backend local|modal] "
    "[--gpu <name>] [--agent claude] [--kernel <path>] [--name <label>] "
    "[--task <path>] [--hints <path>]"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create isolated child environments for GPU kernel optimization",
    )
    parser.add_argument("--operator", default="")
    parser.add_argument("--mode", default="scratch", choices=["scratch", "existing"])
    parser.add_argument("--backend", default="local", choices=["local", "modal"])
    parser.add_argument("--gpu", default="")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--name", default="", dest="label")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--task", "--prompt", default="", dest="task")
    parser.add_argument("--hints", "--info", default="", dest="hints")
    return parser.parse_args()


def resolve_gpu(gpu_arg, backend):
    """Resolve GPU slug and display name. Auto-detect for local, require for modal."""
    if gpu_arg:
        gpu = gpu_arg.lower()
        return gpu, gpu.upper()

    if backend == "modal":
        sys.exit("Error: --gpu is required for modal backend")

    # Local mode: auto-detect via nvidia-smi
    if not shutil.which("nvidia-smi"):
        sys.exit("Error: No GPU specified and nvidia-smi not found. Use --gpu <name>.")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    gpu_full = result.stdout.strip().split("\n")[0].strip()
    if not gpu_full:
        sys.exit("Error: No GPU detected. Use --gpu <name>.")

    match = re.search(r"[A-Z]\d+", gpu_full)
    if not match:
        sys.exit(f"Error: Could not identify GPU model from '{gpu_full}'. Use --gpu <name>.")

    gpu = match.group().lower()
    print(f"Auto-detected GPU: {gpu_full} (using --gpu {gpu})")
    return gpu, gpu.upper()


def load_agent_config(agent):
    """Load agent config JSON, return (config_dict, task_filename, hints_filename)."""
    config_path = PARENT_DIR / "templates" / "agent" / f"{agent}.json"
    if not config_path.is_file():
        sys.exit(f"Error: Agent config not found: {config_path}")
    config = json.loads(config_path.read_text())
    return config, config["task_filename"], config["hints_filename"]


def resolve_dataset(dataset_arg):
    """Resolve dataset path from arg or environment variable."""
    dataset_path = dataset_arg or os.environ.get("FIB_DATASET_PATH", "")
    if not dataset_path:
        print("Error: No dataset path specified.")
        sys.exit("Set FIB_DATASET_PATH or use --dataset to specify the path to the flashinfer-bench trace set.")
    dataset = Path(dataset_path)
    if not dataset.is_dir():
        sys.exit(f"Error: Dataset directory does not exist: {dataset.resolve()}\n"
                 f"Check the path and try again, or set FIB_DATASET_PATH.")
    if not (dataset / "definitions").is_dir():
        sys.exit(f"Error: Dataset directory exists ({dataset.resolve()}) but has no 'definitions/' subdirectory.\n"
                 f"Expected structure: {dataset}/definitions/<category>/<operator>.json\n"
                 f"Is this a valid flashinfer-bench dataset?")
    return dataset


def list_operators(dataset_path):
    """Print available operators and exit."""
    print("Available operators:")
    print()
    for def_file in sorted(dataset_path.glob("definitions/*/*.json")):
        op_name = def_file.stem
        op_type = def_file.parent.name
        print(f"  {op_name}  ({op_type})")
    print()
    print(USAGE_LINE)
    sys.exit(0)


def discover_operator(dataset_path, operator):
    """Find operator definition and workloads files. Returns (definition_path, workloads_path, op_type)."""
    matches = list(dataset_path.glob(f"definitions/*/{operator}.json"))
    if not matches:
        print(f"Error: Operator '{operator}' not found in {dataset_path}/definitions/")
        sys.exit("Run without --operator to list available operators.")
    definition_path = matches[0]
    op_type = definition_path.parent.name
    workloads_path = dataset_path / "workloads" / op_type / f"{operator}.jsonl"
    if not workloads_path.is_file():
        sys.exit(f"Error: Workloads file not found: {workloads_path}")
    return definition_path, workloads_path, op_type


def validate_existing_mode(kernel_path):
    """Validate kernel path for existing mode."""
    if not kernel_path:
        sys.exit("Error: --mode existing requires --kernel <path>")
    kp = Path(kernel_path)
    if not kp.exists():
        sys.exit(f"Error: Kernel path not found: {kernel_path}")


def infer_language(kernel_path):
    """Infer language from kernel file extension. Returns (language, entry_point)."""
    ext_map = {
        ".cu": ("cuda", "binding.py::kernel"),
        ".cpp": ("cpp", "binding.py::kernel"),
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


def read_fragment(path, fallback=None):
    """Read a fragment file. If it doesn't exist, return fallback or exit with error."""
    if path.is_file():
        return path.read_text()
    if fallback is not None:
        return fallback
    sys.exit(f"Error: Fragment not found: {path}")


def make_child_name(label):
    """Generate child directory name.

    With label: ako4fib-run-{label} (error if exists).
    Without label: ako4fib-run-{YYYYMMDD_HHMMSS}.
    """
    from datetime import datetime

    if label:
        return f"ako4fib-run-{label}"
    return f"ako4fib-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def render_template(task_text, placeholders):
    """Render task template with placeholder substitution.

    Block placeholders (alone on a line) replace the entire line.
    Inline placeholders are substituted within lines.
    """
    block_keys = {
        "{{OBJECTIVE}}", "{{BACKEND}}",
        "{{SHAPE_SUMMARY}}", "{{WORKLOAD_SUMMARY}}",
        "{{LANGUAGE_GUIDE}}",
    }
    inline_keys = [
        "{{OPERATOR_DESCRIPTION}}", "{{OPERATOR}}",
        "{{NUM_WORKLOADS}}", "{{GPU_NAME}}",
    ]

    lines = []
    for line in task_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped in block_keys:
            try:
                content = placeholders[stripped]
            except KeyError:
                raise KeyError(f"Missing placeholder value for {stripped!r} in template rendering")
            lines.append(content)
            if not content.endswith("\n"):
                lines.append("\n")
        else:
            for key in inline_keys:
                if key in line:
                    line = line.replace(key, placeholders[key])
            lines.append(line)
    return "".join(lines)


def populate_child(child_dir, *, operator, gpu, mode, backend, kernel_path,
                   definition_path, workloads_path, dataset_path, agent_config, hints_path, hints_filename):
    """Create child directory structure and populate with files."""
    # Create directories — flat solution/ (no language subdirectory)
    for subdir in [".claude", "docs", "scripts", "solution"]:
        (child_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy common files
    shutil.copy2(PARENT_DIR / "templates" / "gitignore", child_dir / ".gitignore")
    shutil.copy2(definition_path, child_dir / "docs" / "definition.json")
    shutil.copy2(workloads_path, child_dir / "docs" / "workloads.jsonl")
    shutil.copy2(PARENT_DIR / "scripts" / "pack_solution.py", child_dir / "scripts" / "pack_solution.py")
    shutil.copy2(PARENT_DIR / "scripts" / "bench_utils.py", child_dir / "scripts" / "bench_utils.py")
    shutil.copytree(PARENT_DIR / "templates" / "languages", child_dir / "docs" / "languages")

    # Copy hints file
    shutil.copy2(Path(hints_path), child_dir / hints_filename)

    # Determine language and entry_point for config.toml
    language = "python"
    entry_point = "kernel.py::run"

    if mode == "scratch":
        # Extract reference from definition.json → solution/kernel.py
        definition = json.loads(definition_path.read_text())
        ref = definition.get("reference", "")
        if not ref:
            sys.exit("Error: No reference field in definition.json")
        (child_dir / "solution" / "kernel.py").write_text(ref)
    else:
        # Existing mode: copy kernel files to solution/
        kp = Path(kernel_path)
        if kp.is_dir():
            # Copy all files from directory
            for f in kp.iterdir():
                if f.is_file():
                    shutil.copy2(f, child_dir / "solution" / f.name)
        else:
            # Copy the kernel file
            shutil.copy2(kp, child_dir / "solution" / kp.name)
            # Copy sibling files (binding.py, other source files)
            for sibling in kp.parent.iterdir():
                if sibling.is_file() and sibling != kp:
                    ext = sibling.suffix.lower()
                    if ext in (".py", ".cu", ".cpp", ".h", ".hpp", ".cuh", ".toml"):
                        if sibling.name != "config.toml":
                            shutil.copy2(sibling, child_dir / "solution" / sibling.name)

        # If config.toml colocated with kernel, use it
        colocated_config = (kp if kp.is_dir() else kp.parent) / "config.toml"
        if colocated_config.is_file():
            shutil.copy2(colocated_config, child_dir / "config.toml")
            print(f"Note: Using config.toml from {colocated_config.resolve()}")
            # Read language/entry_point from colocated config to skip generation
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(colocated_config, "rb") as f:
                colocated = tomllib.load(f)
            language = colocated.get("build", {}).get("language", language)
            entry_point = colocated.get("build", {}).get("entry_point", entry_point)
        else:
            # Infer language from file extension
            language, entry_point = infer_language(kernel_path)

    # Generate config.toml (unless colocated config was already copied)
    config_path = child_dir / "config.toml"
    if not config_path.exists():
        dataset_line = f'dataset_path = "{dataset_path.resolve()}"\n' if backend == "local" else ""
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
        )
        config_path.write_text(config_toml)

    # Copy backend-specific scripts
    if backend == "local":
        shutil.copy2(PARENT_DIR / "scripts" / "bench.sh", child_dir / "scripts" / "bench.sh")
        shutil.copy2(PARENT_DIR / "scripts" / "run_local.py", child_dir / "scripts" / "run_local.py")
    else:
        shutil.copy2(PARENT_DIR / "scripts" / "bench_modal.sh", child_dir / "scripts" / "bench.sh")
        shutil.copy2(PARENT_DIR / "scripts" / "run_modal.py", child_dir / "scripts" / "run_modal.py")

    # Generate .claude/settings.local.json
    settings = {"permissions": agent_config["permissions"]}
    settings_path = child_dir / ".claude" / "settings.local.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def init_git(child_dir, operator, mode, backend):
    """Initialize git repo in child directory. Skips gracefully if git is unavailable."""
    if not shutil.which("git"):
        print("Warning: git not found, skipping repository initialization.", file=sys.stderr)
        return

    try:
        msg = f"Initial commit (spawned from AKO4FIB, operator={operator}, mode={mode}, backend={backend})"
        subprocess.run(["git", "init", "-q"], cwd=child_dir, check=True)
        # Set repo-local user config so commit works without global git config
        subprocess.run(["git", "config", "user.name", "ako4fib"], cwd=child_dir, check=True)
        subprocess.run(["git", "config", "user.email", "ako4fib@local"], cwd=child_dir, check=True)
        subprocess.run(["git", "add", "-A"], cwd=child_dir, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=child_dir, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Warning: git initialization failed ({e}), continuing without git.", file=sys.stderr)


def main():
    args = parse_args()

    # Resolve GPU
    gpu, gpu_name = resolve_gpu(args.gpu, args.backend)

    # Load agent config
    agent_config, task_filename, hints_filename = load_agent_config(args.agent)

    # Resolve dataset
    dataset_path = resolve_dataset(args.dataset)

    # Resolve task and hints template paths
    task_path = Path(args.task) if args.task else PARENT_DIR / "templates" / "task.md"
    if not task_path.is_file():
        sys.exit(f"Error: Task template not found: {task_path}")

    hints_path = Path(args.hints) if args.hints else PARENT_DIR / "templates" / "hints.md"
    if not hints_path.is_file():
        sys.exit(f"Error: Hints file not found: {hints_path}")

    # List operators if none specified
    if not args.operator:
        list_operators(dataset_path)

    # Discover operator
    definition_path, workloads_path, op_type = discover_operator(dataset_path, args.operator)
    operator = args.operator

    print(f"Operator: {operator} ({op_type})")
    print(f"Definition: {definition_path}")
    print(f"Workloads: {workloads_path}")
    print(f"Dataset: {dataset_path}")

    # Validate
    if args.mode == "existing":
        validate_existing_mode(args.kernel)

    # Generate template context
    ctx = generate_context(definition_path, workloads_path)

    # Resolve fragments
    fragments_dir = PARENT_DIR / "templates" / "fragments"
    backend_content = read_fragment(fragments_dir / f"backend-{args.backend}.md")
    objective_raw = read_fragment(fragments_dir / f"objective-{args.mode}.md")
    language_guide = read_fragment(fragments_dir / "language-guide.md")

    # Pre-pass: substitute inline placeholders in objective fragment
    inline_vars = {
        "{{GPU_NAME}}": gpu_name,
        "{{OPERATOR_DESCRIPTION}}": ctx["operator_description"],
        "{{OPERATOR}}": operator,
    }
    objective = objective_raw
    for key, val in inline_vars.items():
        objective = objective.replace(key, val)

    # Build full placeholder map
    placeholders = {
        "{{OBJECTIVE}}": objective,
        "{{BACKEND}}": backend_content,
        "{{LANGUAGE_GUIDE}}": language_guide,
        "{{SHAPE_SUMMARY}}": ctx["shape_summary"],
        "{{WORKLOAD_SUMMARY}}": ctx["workload_summary"],
        "{{OPERATOR}}": operator,
        "{{OPERATOR_DESCRIPTION}}": ctx["operator_description"],
        "{{NUM_WORKLOADS}}": ctx["num_workloads"],
        "{{GPU_NAME}}": gpu_name,
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
        gpu=gpu,
        mode=args.mode,
        backend=args.backend,
        kernel_path=args.kernel,
        definition_path=definition_path,
        workloads_path=workloads_path,
        dataset_path=dataset_path,
        agent_config=agent_config,
        hints_path=hints_path,
        hints_filename=hints_filename,
    )

    # Render and write task file
    task_text = task_path.read_text()
    rendered = render_template(task_text, placeholders)
    (child_dir / task_filename).write_text(rendered)

    # Git init
    init_git(child_dir, operator, args.mode, args.backend)

    # Summary
    print()
    print("===== Child environment created =====")
    print(f"  Path:     {child_dir}")
    print(f"  Operator: {operator}")
    print(f"  Mode:     {args.mode}")
    print(f"  Backend:  {args.backend}")
    print(f"  GPU:      {gpu_name}")
    print(f"  Dataset:  {dataset_path}")
    if args.mode == "existing":
        print(f"  Kernel:   {args.kernel}")
    print()
    print("Next steps:")
    print(f"  cd {child_dir}")
    print()
    print("  # Start your code agent, e.g.:")
    print("  claude")
    print()
    print("To run benchmark manually:")
    print("  bash scripts/bench.sh")
    print("=======================================")


if __name__ == "__main__":
    main()
