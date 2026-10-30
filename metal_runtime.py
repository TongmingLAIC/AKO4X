"""Command-based Metal validation and measurement; stdlib only, no GPU imports."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

TOKENS = {"repo", "build", "baseline", "reference", "candidate", "baseline_build",
          "reference_build", "candidate_build", "evidence", "output", "python"}
GPU_ENV = {"BENCH_GPU_LOCK", "NI_LOCK_HELD", "NACE_GPU_LOCK_HELD",
           "MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER"}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    required = {"name", "editable", "build", "smoke", "correctness", "benchmark"}
    optional = {"variables", "env", "timeout_seconds", "shader_validation"}
    if not isinstance(spec, dict) or not required <= spec.keys() or spec.keys() - required - optional:
        raise ValueError(f"config requires {sorted(required)}; optional keys: {sorted(optional)}")
    if not isinstance(spec["name"], str) or not spec["name"].strip():
        raise ValueError("name must be a non-empty string")
    editable = spec["editable"]
    if not isinstance(editable, list) or not editable:
        raise ValueError("editable must explicitly name source paths or glob patterns")
    for pattern in editable:
        if (not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute()
                or ".." in Path(pattern).parts or pattern.startswith((".git", ".ako", ".agents", ".claude"))):
            raise ValueError(f"invalid editable pattern: {pattern!r}")
    for key in ("variables", "env"):
        values = spec.setdefault(key, {})
        if not isinstance(values, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()
        ):
            raise ValueError(f"{key} must map strings to strings")
    if TOKENS & spec["variables"].keys() or GPU_ENV & spec["env"].keys():
        raise ValueError("variables/env cannot override runner paths or GPU lock/validation controls")
    timeout = spec.setdefault("timeout_seconds", 1800)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
        raise ValueError("timeout_seconds must be in (0, 86400]")
    shader = spec.setdefault("shader_validation", {"enabled": True})
    if not isinstance(shader, dict) or type(shader.get("enabled")) is not bool:
        raise ValueError("shader_validation.enabled must be a boolean")
    if not shader["enabled"]:
        size = shader.get("static_threadgroup_bytes")
        if (type(size) is not int or not 16384 < size <= 32768
                or not isinstance(shader.get("reason"), str) or not shader["reason"].strip()):
            raise ValueError("disabling the layer requires its documented doubling limitation, "
                             "static_threadgroup_bytes in (16384, 32768], and a reason")
    bench = spec["benchmark"]
    keys = {"command", "metric", "direction", "pairs", "min_speedup", "max_relative_spread"}
    if not isinstance(bench, dict) or set(bench) != keys:
        raise ValueError(f"benchmark requires exactly {sorted(keys)}")
    if bench["direction"] not in ("lower", "higher"):
        raise ValueError("benchmark.direction must be lower or higher")
    if not isinstance(bench["metric"], str) or not bench["metric"].strip():
        raise ValueError("benchmark.metric must be a dotted JSON field path")
    if type(bench["pairs"]) is not int or not 2 <= bench["pairs"] <= 100:
        raise ValueError("benchmark.pairs must be between 2 and 100 (A B A B minimum)")
    for key, low, high in (("min_speedup", 1, 1000), ("max_relative_spread", 0, 1)):
        value = bench[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < value <= high:
            raise ValueError(f"benchmark.{key} must be in ({low}, {high}]")
    commands = []
    for phase in ("build", "smoke", "correctness"):
        if not isinstance(spec[phase], list) or not spec[phase]:
            raise ValueError(f"{phase} requires at least one command")
        commands.extend(spec[phase])
    commands.append(bench["command"])
    for command in commands:
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("commands must be non-empty argv arrays; shell text is not accepted")
        for argument in command:
            unknown = set(re.findall(r"\{(\w+)\}", argument)) - TOKENS - spec["variables"].keys()
            if unknown:
                raise ValueError(f"unknown command placeholders: {sorted(unknown)}")
    if not any("{output}" in x for x in bench["command"]):
        raise ValueError("benchmark.command must write JSON to {output}")
    return spec


def expand(command: list[str], context: dict) -> list[str]:
    return [re.sub(r"\{(\w+)\}", lambda m: str(context[m[1]]), part) for part in command]


@contextlib.contextmanager
def exclusive_lock(path: Path, timeout: float = 30):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"lock is busy: {path}") from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_command(command: list[str], *, cwd: Path, env: dict, evidence: Path,
                name: str, timeout: float, check: bool = True) -> dict:
    record = {"argv": command, "cwd": str(cwd), "timed_out": False}
    start = time.monotonic()
    process = None
    try:
        with (evidence / f"{name}.stdout").open("xb") as stdout, \
                (evidence / f"{name}.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
            finally:
                # Kill the group even if its leader exited: an orphan benchmark
                # must not outlive this command and our exclusive GPU lock.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            record["returncode"] = process.returncode
    except OSError as exc:
        record.update(returncode=127, error=str(exc))
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        write_json(evidence / f"{name}.command.json", record)
    if check and (record.get("returncode") != 0 or record["timed_out"]):
        raise RuntimeError(f"{name} failed (exit {record.get('returncode')}, "
                           f"timeout={record['timed_out']}); see {evidence / (name + '.stderr')}")
    return record


def read_metric(path: Path, field: str) -> float:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("invalid") != []:
        raise ValueError(f"{path.name}: a valid measurement must include an empty invalid array")
    value = data
    for component in field.split("."):
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path.name}: {field} must be finite and positive")
    return float(value)


def evaluate(spec: dict, *, baseline: Path, reference: Path, candidate: Path,
             evidence: Path, gpu_lock: Path, allow_gpu: bool, report: dict) -> None:
    if not allow_gpu:
        raise ValueError("GPU execution requires --allow-gpu; init and build are CPU-only")
    import sys

    builds = evidence.parent / "build"
    context = {**spec["variables"], "python": sys.executable, "evidence": evidence,
               "baseline": baseline, "reference": reference, "candidate": candidate,
               "baseline_build": builds / "baseline", "reference_build": builds / "reference",
               "candidate_build": builds / "candidate"}
    if baseline == reference:
        context["reference_build"] = context["baseline_build"]
    env = {k: v for k, v in os.environ.items() if k not in GPU_ENV}
    env.update(spec["env"])

    def commands(phase, ctx, command_env, prefix):
        for index, command in enumerate(spec[phase]):
            run_command(expand(command, ctx), cwd=Path(ctx["repo"]), env=command_env,
                        evidence=evidence, name=f"{prefix}-{index}", timeout=spec["timeout_seconds"])

    for role in ("baseline", "reference", "candidate"):
        if role == "reference" and reference == baseline:
            continue
        context[f"{role}_build"].mkdir(parents=True)
        commands("build", {**context, "repo": context[role], "build": context[f"{role}_build"]}, env, f"build-{role}")

    with exclusive_lock(gpu_lock):
        env.update(BENCH_GPU_LOCK=str(gpu_lock), NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1",
                   MTL_SHADER_VALIDATION="0", MTL_DEBUG_LAYER="0")
        layer = "1" if spec["shader_validation"]["enabled"] else "0"
        report["shader_validation"] = spec["shader_validation"]
        for role in ("reference", "candidate"):
            ctx = {**context, "repo": context[role], "build": context[f"{role}_build"]}
            commands("smoke", ctx, {**env, "MTL_SHADER_VALIDATION": layer, "MTL_DEBUG_LAYER": layer}, f"smoke-{role}")
        commands("correctness", {**context, "repo": candidate, "build": context["candidate_build"]}, env, "correctness")
        report["correctness_passed"] = True
        bench = spec["benchmark"]
        measurements = report.setdefault("measurements", [])
        for pair in range(1, bench["pairs"] + 1):
            for letter, role in (("A", "reference"), ("B", "candidate")):
                label = f"{letter}{pair}"
                output = evidence / f"{label}.json"
                ctx = {**context, "repo": context[role], "build": context[f"{role}_build"], "output": output}
                run_command(expand(bench["command"], ctx), cwd=context[role], env=env,
                            evidence=evidence, name=label, timeout=spec["timeout_seconds"])
                measurements.append({"label": label, "role": role, "value": read_metric(output, bench["metric"])})
    a = [m["value"] for m in measurements if m["role"] == "reference"]
    b = [m["value"] for m in measurements if m["role"] == "candidate"]
    ratios = [x / y if bench["direction"] == "lower" else y / x for x, y in zip(a, b)]
    spreads = {key: (max(values) - min(values)) / statistics.median(values)
               for key, values in (("reference", a), ("candidate", b))}
    threshold = max(bench["min_speedup"], 1 + max(spreads.values()))
    noisy = max(spreads.values()) > bench["max_relative_spread"]
    passed = not noisy and min(ratios) >= threshold
    report.update(accepted=passed, status="accepted" if passed else "rejected", pair_speedups=ratios,
                  median_speedup=statistics.median(ratios), relative_spread=spreads,
                  required_speedup=threshold, reason="passed" if passed else
                  ("noisy_measurements" if noisy else "no_measurable_improvement"))
