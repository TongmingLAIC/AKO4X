"""CPU-only checks for the portable Metal skill's numerical and tooling helpers."""

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1] / "templates/skills/metal"
module_spec = importlib.util.spec_from_file_location("compare_fixtures", SKILL / "scripts/compare_fixtures.py")
fixtures = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(fixtures)


class FixtureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="metal fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, name, values=(1.0, 2.0), dtype="float32", shape=None, raw=None):
        directory = self.root / name
        directory.mkdir()
        formats = {"float16": "e", "bfloat16": "H", "float32": "f", "bool": "B",
                   "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
                   "int32": "i", "uint32": "I", "int64": "q", "uint64": "Q"}
        data = raw if raw is not None else struct.pack("<" + formats[dtype] * len(values), *values)
        (directory / "output.bin").write_bytes(data)
        manifest = {"version": 1, "case_id": "fixed-input-case",
                    "input_sha256": hashlib.sha256(b"frozen inputs and parameters").hexdigest(),
                    "producer": {"backend": "cpu", "revision": "test-reference"},
                    "tensors": [{"name": "output", "dtype": dtype,
                                 "shape": list(shape) if shape is not None else [len(values)],
                                 "file": "output.bin", "sha256": hashlib.sha256(data).hexdigest()}]}
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def edit(self, path, change):
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SKILL / "scripts/compare_fixtures.py"), *map(str, args)],
                              cwd=self.root, capture_output=True, text=True)

    def test_float_error_uses_reference_relative_tolerance(self):
        reference = self.fixture("reference", (2.0, -2.0))
        candidate = self.fixture("candidate", (2.125, -2.125))
        self.assertFalse(fixtures.compare(reference, candidate)["passed"])
        self.assertFalse(fixtures.compare(reference, candidate, rtol=0.06)["passed"])
        report = fixtures.compare(reference, candidate, rtol=0.0625)
        self.assertTrue(report["passed"])
        self.assertEqual(report["tensors"][0]["max_abs_error_finite"], 0.125)
        self.assertTrue(fixtures.compare(reference, candidate, atol=0.125)["passed"])

    def test_fixed_width_integers_are_exact_even_with_tolerance(self):
        cases = {"int8": 126, "uint8": 254, "int16": 32766, "uint16": 65534,
                 "int32": 2**31 - 2, "uint32": 2**32 - 2,
                 "int64": 2**63 - 2, "uint64": 2**64 - 2, "bool": 0}
        for dtype, value in cases.items():
            with self.subTest(dtype=dtype):
                a = self.fixture(dtype + "-a", (value,), dtype=dtype)
                b = self.fixture(dtype + "-b", (value + 1,), dtype=dtype)
                report = fixtures.compare(a, b, atol=100, rtol=1)
                self.assertFalse(report["passed"])
                self.assertEqual(report["tensors"][0]["max_abs_error_finite"], 1)
                self.assertTrue(fixtures.compare(a, a, bitwise=True)["passed"])

    def test_half_and_bfloat_decode_as_floats(self):
        for dtype, left, right in (("float16", 1.0, 1.25), ("bfloat16", 0x3F80, 0x3FA0)):
            with self.subTest(dtype=dtype):
                a = self.fixture(dtype + "-a", (left,), dtype=dtype)
                b = self.fixture(dtype + "-b", (right,), dtype=dtype)
                self.assertFalse(fixtures.compare(a, b, atol=0.24)["passed"])
                self.assertTrue(fixtures.compare(a, b, atol=0.25)["passed"])

    def test_nonfinite_policy_and_infinity_sign(self):
        a = self.fixture("a", (float("nan"), float("inf"), float("-inf")))
        b = self.fixture("b", (float("nan"), float("inf"), float("-inf")))
        c = self.fixture("c", (float("nan"), float("-inf"), float("-inf")))
        self.assertFalse(fixtures.compare(a, b)["passed"])
        self.assertTrue(fixtures.compare(a, b, equal_nonfinite=True)["passed"])
        self.assertFalse(fixtures.compare(a, c, equal_nonfinite=True)["passed"])
        finite = self.fixture("finite", (0.0, 0.0, 0.0))
        self.assertFalse(fixtures.compare(a, finite, equal_nonfinite=True)["passed"])

    def test_signed_zero_is_numeric_but_not_bitwise_equal(self):
        a = self.fixture("a", (0.0,))
        b = self.fixture("b", (-0.0,))
        report = fixtures.compare(a, b)
        self.assertTrue(report["numeric_passed"])
        self.assertFalse(report["bitwise_identical"])
        self.assertFalse(fixtures.compare(a, b, bitwise=True)["passed"])

    def test_empty_output_cannot_pass_but_scalar_can(self):
        empty = self.fixture("empty", (), shape=(0, 5))
        self.assertFalse(fixtures.compare(empty, empty)["passed"])
        scalar = self.fixture("scalar", (1.0,), shape=())
        self.assertEqual(fixtures.compare(scalar, scalar)["elements_compared"], 1)

    def test_input_and_output_identity_must_match(self):
        reference = self.fixture("reference")
        changes = [lambda d: d.update(case_id="another-case"),
                   lambda d: d.update(input_sha256="0" * 64),
                   lambda d: d["tensors"][0].update(name="different-output"),
                   lambda d: d["tensors"][0].update(dtype="uint32"),
                   lambda d: d["tensors"][0].update(shape=[1, 2])]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                candidate = self.fixture(str(index))
                self.edit(candidate, change)
                with self.assertRaises(ValueError):
                    fixtures.compare(reference, candidate)

    def test_invalid_manifest_metadata_is_rejected(self):
        changes = [lambda d: d.update(version=True), lambda d: d.update(case_id=""),
                   lambda d: d.update(input_sha256="not-a-digest"), lambda d: d.update(producer={}),
                   lambda d: d.update(tensors=[]),
                   lambda d: d["tensors"].append(d["tensors"][0].copy()),
                   lambda d: d["tensors"][0].update(shape=[True, 2]),
                   lambda d: d["tensors"][0].update(shape=[-1]),
                   lambda d: d["tensors"][0].update(dtype="unknown")]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                path = self.fixture(str(index))
                self.edit(path, change)
                with self.assertRaises(ValueError):
                    fixtures.compare(path, path)

    def test_payload_length_hash_and_bool_encoding_are_checked(self):
        short = self.fixture("short", raw=b"\0")
        with self.assertRaisesRegex(ValueError, "byte count"):
            fixtures.compare(short, short)
        changed = self.fixture("changed")
        (changed.parent / "output.bin").write_bytes(struct.pack("<ff", 3.0, 4.0))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            fixtures.compare(changed, changed)
        boolean = self.fixture("bool", (2,), dtype="bool")
        with self.assertRaisesRegex(ValueError, "bool payload"):
            fixtures.compare(boolean, boolean)

    def test_payload_cannot_escape_fixture_directory(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(struct.pack("<ff", 1.0, 2.0))
        for index, value in enumerate(("../outside.bin", str(outside), "link.bin")):
            with self.subTest(value=value):
                path = self.fixture(str(index))
                (path.parent / "link.bin").symlink_to(outside)
                self.edit(path, lambda d: d["tensors"][0].update(file=value))
                with self.assertRaises(ValueError):
                    fixtures.compare(path, path)

    def test_invalid_tolerances_fail_instead_of_disabling_gate(self):
        path = self.fixture("reference")
        for value in (-1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fixtures.compare(path, path, atol=value)
        with self.assertRaisesRegex(ValueError, "overflow"):
            fixtures.compare(path, path, rtol=1.7e308)

    def test_cli_exit_codes_and_reports_are_not_overwritten(self):
        a = self.fixture("a")
        b = self.fixture("b", (1.0, 3.0))
        report = self.root / "comparison.json"
        accepted = self.cli(a, a, "--json", report)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        original = report.read_bytes()
        rejected = self.cli(a, b)
        self.assertEqual(rejected.returncode, 1, rejected.stderr)
        self.assertFalse(json.loads(rejected.stdout)["passed"])
        invalid = self.cli(a, self.root / "missing.json")
        self.assertEqual(invalid.returncode, 2)
        duplicate = self.cli(a, b, "--json", report)
        self.assertEqual(duplicate.returncode, 2)
        self.assertEqual(report.read_bytes(), original)


class DoctorTest(unittest.TestCase):
    def test_copied_skill_detects_missing_toolchain_and_never_executes_probe(self):
        with tempfile.TemporaryDirectory(prefix="metal doctor ") as temporary:
            root = Path(temporary)
            copied = root / "standalone-skill"
            shutil.copytree(SKILL, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            binary = root / "bin"
            binary.mkdir()
            marker = root / "probe-executed"
            fake = f"#!{sys.executable}\n" + '''import os, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--find", "metal"]:
    print("/fake/Xcode/usr/bin/metal")
elif args == ["metal", "--version"]:
    print("missing Metal Toolchain component", file=sys.stderr)
    sys.exit(1)
elif "clang++" in args:
    output = Path(args[args.index("-o") + 1])
    output.write_text("#!" + sys.executable + "\\nfrom pathlib import Path; Path(" + repr(os.environ["FAKE_MARKER"]) + ").touch()\\n")
    output.chmod(0o755)
else:
    print("fake host tool; Metal System Trace")
'''
            for name in ("xcrun", "xcode-select", "xcodebuild", "sw_vers"):
                path = binary / name
                path.write_text(fake)
                path.chmod(0o755)
            env = {**os.environ, "PATH": str(binary), "FAKE_MARKER": str(marker)}
            command = [sys.executable, str(copied / "scripts/metal_doctor.py")]
            report_path = root / "doctor.json"
            result = subprocess.run([*command, "--compile-probe", str(root / "build"), "--json", str(report_path)],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["checks"]["metal_path"]["returncode"], 0)
            self.assertFalse(report["metal_compiler_usable"])
            self.assertTrue(report["xctrace_usable"])
            self.assertTrue(Path(report["probe_build"]["binary"]).is_file())
            self.assertFalse(report["probe_build"]["executed"])
            self.assertFalse(marker.exists())
            required = subprocess.run([*command, "--require-compiler"], cwd=root, env=env,
                                      capture_output=True, text=True)
            self.assertEqual(required.returncode, 1, required.stderr)
            self.assertFalse(json.loads(required.stdout)["metal_compiler_usable"])
            original = report_path.read_bytes()
            duplicate = subprocess.run([*command, "--json", str(report_path)], cwd=root, env=env,
                                       capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual(report_path.read_bytes(), original)
            # Neither helper imports a module from the template repository.
            help_result = subprocess.run([sys.executable, str(copied / "scripts/compare_fixtures.py"), "--help"],
                                         cwd=root, capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)


if __name__ == "__main__":
    unittest.main()
