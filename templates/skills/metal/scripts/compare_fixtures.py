#!/usr/bin/env python3
"""Compare frozen CUDA/CPU/Metal raw-tensor fixtures. CPU-only, stdlib-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

FORMATS = {"float16": "e", "bfloat16": "H", "float32": "f", "bool": "B",
           "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
           "int32": "i", "uint32": "I", "int64": "q", "uint64": "Q"}
FLOATS = {"float16", "bfloat16", "float32"}


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("SHA-256 values must contain exactly 64 hexadecimal characters")
    return value.lower()


def load_manifest(path: Path):
    path = path.resolve()
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("expected a version-1 fixture manifest")
    if not isinstance(manifest.get("case_id"), str) or not manifest["case_id"].strip():
        raise ValueError("case_id must be nonempty")
    digest(manifest.get("input_sha256"))
    producer = manifest.get("producer")
    if not isinstance(producer, dict) or any(not isinstance(producer.get(k), str) or not producer[k].strip()
                                            for k in ("backend", "revision")):
        raise ValueError("producer must identify backend and revision")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("manifest must contain tensors")
    loaded = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("each tensor must be an object")
        name, dtype, shape = tensor.get("name"), tensor.get("dtype"), tensor.get("shape")
        if not isinstance(name, str) or not name or name in loaded:
            raise ValueError("tensor names must be nonempty and unique")
        if not isinstance(dtype, str) or dtype not in FORMATS:
            raise ValueError(f"{name}: unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"{name}: shape must contain nonnegative integers")
        file = tensor.get("file")
        if not isinstance(file, str) or not file or Path(file).is_absolute() or ".." in Path(file).parts:
            raise ValueError(f"{name}: file must be relative to the manifest")
        payload = (path.parent / file).resolve()
        if not payload.is_relative_to(path.parent):
            raise ValueError(f"{name}: payload leaves fixture directory")
        expected_size = math.prod(shape) * struct.calcsize("<" + FORMATS[dtype])
        if payload.stat().st_size != expected_size:
            raise ValueError(f"{name}: payload byte count does not match shape and dtype")
        data = payload.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest(tensor.get("sha256")):
            raise ValueError(f"{name}: payload hash mismatch")
        if dtype == "bool" and any(value not in (0, 1) for value in data):
            raise ValueError(f"{name}: bool payload must use bytes 0 or 1")
        loaded[name] = (tensor, data)
    return manifest, loaded


def values(dtype, data):
    for (value,) in struct.iter_unpack("<" + FORMATS[dtype], data):
        if dtype == "bfloat16":
            value = struct.unpack("<f", struct.pack("<I", value << 16))[0]
        yield value


def compare(reference: Path, candidate: Path, *, atol=0.0, rtol=0.0,
            equal_nonfinite=False, bitwise=False):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
           for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    ref, a = load_manifest(reference)
    new, b = load_manifest(candidate)
    if ref["case_id"] != new["case_id"] or digest(ref["input_sha256"]) != digest(new["input_sha256"]):
        raise ValueError("case or exact input identity differs")
    if a.keys() != b.keys():
        raise ValueError("missing or extra output tensors")
    reports, count, all_bytes_equal = [], 0, True
    for name, (meta, left) in a.items():
        other, right = b[name]
        if meta["dtype"] != other["dtype"] or meta["shape"] != other["shape"]:
            raise ValueError(f"{name}: dtype/shape differs")
        same_bytes = left == right
        all_bytes_equal &= same_bytes
        mismatches = nonfinite_mismatches = 0
        max_absolute = 0
        elements = math.prod(meta["shape"])
        for x, y in zip(values(meta["dtype"], left), values(meta["dtype"], right)):
            if meta["dtype"] not in FLOATS:
                mismatch = x != y
                max_absolute = max(max_absolute, abs(x - y))
            elif not math.isfinite(x) or not math.isfinite(y):
                same = (math.isnan(x) and math.isnan(y)) or x == y
                mismatch = not (equal_nonfinite and same)
                nonfinite_mismatches += mismatch
            else:
                error = abs(x - y)
                tolerance = atol + rtol * abs(x)
                if not math.isfinite(tolerance):
                    raise ValueError("tolerance calculation overflowed")
                max_absolute = max(max_absolute, error)
                mismatch = error > tolerance
            mismatches += mismatch
        count += elements
        reports.append({"name": name, "elements": elements, "mismatches": mismatches,
                        "nonfinite_mismatches": nonfinite_mismatches,
                        "max_abs_error_finite": max_absolute, "bitwise_identical": same_bytes})
    numeric_passed = count > 0 and all(r["mismatches"] == 0 for r in reports)
    passed = numeric_passed and (not bitwise or all_bytes_equal)
    return {"passed": passed, "numeric_passed": numeric_passed, "elements_compared": count,
            "bitwise_identical": all_bytes_equal and count > 0,
            "case_id": ref["case_id"], "input_sha256": digest(ref["input_sha256"]),
            "reference_producer": ref["producer"], "candidate_producer": new["producer"],
            "atol": atol, "rtol": rtol, "equal_nonfinite": equal_nonfinite,
            "require_bitwise": bitwise, "tensors": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--equal-nonfinite", action="store_true")
    parser.add_argument("--bitwise", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.candidate, atol=args.atol, rtol=args.rtol,
                         equal_nonfinite=args.equal_nonfinite, bitwise=args.bitwise)
        status = 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        report, status = {"passed": False, "error": str(error)}, 2
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        try:
            # Preserve prior evidence rather than silently overwriting a result.
            with args.json.open("x") as output:
                output.write(text)
        except OSError as error:
            print(f"Cannot create report: {error}", file=sys.stderr)
            return 2
    print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
