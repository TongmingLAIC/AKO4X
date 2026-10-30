#!/usr/bin/env python3
"""Inspect Metal host tooling without creating a GPU device or submitting work."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def probe(command, env, timeout=20):
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
        return {"argv": command, "returncode": result.returncode,
                "stdout": result.stdout[-12000:], "stderr": result.stderr[-6000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": command, "returncode": None, "error": str(error)}


def inspect(env):
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "developer_dir_override": env.get("DEVELOPER_DIR"),
              "gpu_device_created": False, "gpu_work_submitted": False}
    commands = {
        "selected_developer_dir": ["xcode-select", "-p"],
        "os": ["sw_vers"],
        "xcode": ["xcodebuild", "-version"],
        "sdk": ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        "metal_path": ["xcrun", "--find", "metal"],
        "metal_version": ["xcrun", "metal", "--version"],
        "xctrace_templates": ["xcrun", "xctrace", "list", "templates"],
    }
    report["checks"] = {name: probe(command, env) for name, command in commands.items()}
    report["metal_compiler_usable"] = report["checks"]["metal_version"]["returncode"] == 0
    report["xctrace_usable"] = report["checks"]["xctrace_templates"]["returncode"] == 0
    return report


def compile_probe(directory, env):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory.resolve() / "metal-device-caps"
    if output.exists() or output.is_symlink():
        raise ValueError(f"probe output already exists: {output}")
    source = Path(__file__).resolve().parent.parent / "assets/device_caps.mm"
    command = ["xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-fobjc-arc",
               "-Wall", "-Wextra", "-Werror", "-mmacosx-version-min=11.0", str(source),
               "-framework", "Foundation", "-framework", "Metal", "-o", str(output)]
    result = probe(command, env, timeout=120)
    result["binary"] = str(output) if result["returncode"] == 0 else None
    result["executed"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--developer-dir", type=Path)
    parser.add_argument("--compile-probe", type=Path, help="compile a host capability probe; do not execute it")
    parser.add_argument("--require-compiler", action="store_true", help="fail if the Metal compiler cannot run")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    env = os.environ.copy()
    if args.developer_dir:
        env["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
    report = inspect(env)
    status = int(args.require_compiler and not report["metal_compiler_usable"])
    if args.compile_probe:
        try:
            report["probe_build"] = compile_probe(args.compile_probe, env)
            status |= int(report["probe_build"]["returncode"] != 0)
        except (OSError, ValueError) as error:
            report["probe_build"] = {"error": str(error), "executed": False}
            status = 1
    text = json.dumps(report, indent=2) + "\n"
    if args.json:
        try:
            with args.json.open("x") as output:
                output.write(text)
        except OSError as error:
            print(f"Cannot create report: {error}", file=sys.stderr)
            return 2
    print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
