"""CPU-only integration tests. Every would-be GPU command is a tiny fake process."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import metal_campaign as campaign
from metal_runtime import exclusive_lock, load_spec, read_metric, run_command, write_json


DRIVER = '''import fcntl, json, os, sys
from pathlib import Path
phase = sys.argv[1]
repo = Path(sys.argv[2])
data = json.loads((repo / "src/kernel.metal").read_text())
if os.environ.get("FAKE_PHASE_LOG"):
    with open(os.environ["FAKE_PHASE_LOG"], "a") as log:
        log.write(phase + (":" + Path(sys.argv[3]).stem if phase == "bench" else "") + "\\n")
if phase != "profile":
    assert "MTL_CAPTURE_ENABLED" not in os.environ
if phase != "build":
    assert os.environ["NI_LOCK_HELD"] == os.environ["NACE_GPU_LOCK_HELD"] == "1"
    with open(os.environ["BENCH_GPU_LOCK"], "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("outer runner does not hold GPU lock")
if phase == "build":
    assert "NI_LOCK_HELD" not in os.environ
elif phase == "smoke":
    assert os.environ["MTL_SHADER_VALIDATION"] == os.environ.get("FAKE_EXPECT_LAYER", "1")
    assert os.environ["MTL_DEBUG_LAYER"] == os.environ.get("FAKE_EXPECT_LAYER", "1")
elif phase == "correctness":
    assert os.environ["MTL_SHADER_VALIDATION"] == "0"
    print("frozen baseline:", (Path(sys.argv[3]) / "src/kernel.metal").read_text())
    sys.exit(0 if data.get("correct", True) else 1)
elif phase == "bench":
    assert os.environ["MTL_SHADER_VALIDATION"] == "0"
    output = Path(sys.argv[3])
    values = data.get("samples", [data["latency"], data["latency"]])
    value = values[int(output.stem[1:]) - 1]
    output.write_text(json.dumps({"latency_ms": value, "invalid": data.get("invalid", [])}))
    if data.get("mutate") and output.stem == "B2":
        (repo / "src/kernel.metal").write_text("changed during benchmark")
elif phase == "profile":
    assert os.environ["MTL_SHADER_VALIDATION"] == "0"
    assert os.environ["MTL_DEBUG_LAYER"] == "0"
    assert os.environ["MTL_CAPTURE_ENABLED"] == "1"
    helper = Path(sys.argv[4]) / "scripts/compare_fixtures.py"
    assert helper.is_file()
    if data.get("profile_mutate"):
        helper.write_text("changed during profiling")
    if data.get("profile_failure"):
        raise RuntimeError("fake capture failed")
    (Path(sys.argv[3]) / "profile.json").write_text(json.dumps({"latency_ms": 0.001, "invalid": ["profiling"]}))
'''


class NativeCampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ako metal test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "native project"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src/kernel.metal").write_text(json.dumps({"latency": 10.0}))
        (self.repo / "host.mm").write_text("// frozen native host\n")
        (self.repo / "driver.py").write_text(DRIVER)
        (self.repo / ".gitignore").write_text("build/\n*.bin\n")
        campaign.git(self.repo, "init", "-q", "-b", "main")
        campaign.seal(self.repo)
        self.config = self.root / "config.json"
        self.spec = {
            "name": "fake CPU command campaign", "editable": ["src/*.metal", "src/*.mm"],
            "build": [["{python}", "{repo}/driver.py", "build", "{repo}"]],
            "smoke": [["{python}", "{repo}/driver.py", "smoke", "{repo}"]],
            "correctness": [["{python}", "{candidate}/driver.py", "correctness", "{candidate}", "{baseline}"]],
            "benchmark": {"command": ["{python}", "{repo}/driver.py", "bench", "{repo}", "{output}"],
                          "metric": "latency_ms", "direction": "lower", "pairs": 2,
                          "min_speedup": 1.02, "max_relative_spread": 0.05},
            "timeout_seconds": 5,
        }
        write_json(self.config, self.spec)
        self.output = self.root / "campaign"
        campaign.initialize(self.repo, "HEAD", self.config, self.output, {"name": "codex"})
        self.workspace = self.output / "workspace"
        self.gpu_lock = self.root / "fake-gpu.lock"
        info = patch.object(campaign, "system_info", return_value={"fake": "CPU-only"})
        info.start()
        self.addCleanup(info.stop)

    def change(self, **values):
        (self.workspace / "src/kernel.metal").write_text(json.dumps({"latency": 5.0, **values}))

    def run_round(self, label, *, agent=False, allow=True):
        return campaign.run_round(self.output, candidate_source=None if agent else self.workspace,
                                  prompt="optimize the native kernel", label=label, allow_gpu=allow,
                                  agent_timeout=2, gpu_lock=self.gpu_lock)

    def state(self):
        return json.loads((self.output / "state.json").read_text())

    def test_complete_sources_abab_and_best_survives_rejection(self):
        self.change()
        (self.workspace / "src/nested").mkdir()
        (self.workspace / "src/nested/new.metal").write_text("// new kernel\n")
        (self.workspace / "src/nested/dispatch.mm").write_text("// new host source\n")
        (self.workspace / "weights.bin").write_bytes(b"ignored model")
        original_status = campaign.git(self.workspace, "status", "--porcelain")
        first = self.run_round("faster")
        self.assertTrue(first["accepted"], first)
        self.assertEqual([m["label"] for m in first["measurements"]], ["A1", "B1", "A2", "B2"])
        self.assertEqual(first["pair_speedups"], [2.0, 2.0])
        saved = self.output / self.state()["best"]
        self.assertTrue((saved / "src/nested/dispatch.mm").is_file())
        self.assertTrue((saved / "src/nested/new.metal").is_file())
        self.assertFalse((saved / "weights.bin").exists())
        self.assertEqual(campaign.git(self.workspace, "status", "--porcelain"), original_status)
        self.assertIn("new.metal", (saved.parent / "evidence/candidate.patch").read_text())
        self.change(latency=9.0)
        second = self.run_round("slower")
        self.assertEqual(second["status"], "rejected", second)
        self.assertEqual(second["reference_commit"], first["candidate_commit"])
        self.assertEqual(second["baseline_commit"], first["baseline_commit"])
        self.assertEqual(self.state()["best_commit"], first["candidate_commit"])
        self.assertEqual(len((self.output / "history.jsonl").read_text().splitlines()), 2)
        with self.assertRaises(FileExistsError):
            self.run_round("faster")

    def test_gpu_opt_in_precedes_agent_and_build(self):
        self.change()
        with patch.object(campaign, "run_command") as command, self.assertRaises(ValueError):
            self.run_round("unauthorized", agent=True, allow=False)
        command.assert_not_called()
        self.assertEqual(list((self.output / "rounds").iterdir()), [])
        campaign.cpu_build(self.output, self.workspace)
        self.assertEqual(self.state()["best"], "baseline")

    def test_correctness_invalid_nan_and_negative_fail_closed(self):
        bad = [{"correct": False}, {"invalid": ["GPU busy"]}, {"latency": -1},
               {"latency": float("nan")}, {"latency": True}, {"samples": [5, 5.6]}]
        for index, values in enumerate(bad):
            with self.subTest(values=values):
                self.change(**values)
                result = self.run_round(f"bad-{index}")
                self.assertFalse(result["accepted"], result)
                self.assertEqual(self.state()["best"], "baseline")
                if index == 0:
                    self.assertFalse(result["correctness_passed"])
                    self.assertFalse((self.output / f"rounds/bad-{index}/evidence/A1.json").exists())

    def test_frozen_config_and_out_of_scope_sources_are_rejected(self):
        self.change()
        (self.workspace / "driver.py").write_text("raise SystemExit(0)\n")
        result = self.run_round("changed-oracle")
        self.assertIn("outside editable scope", result["reason"])
        self.assertFalse(result["correctness_passed"])
        (self.output / "evaluation.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_round("changed-config")

    def test_sources_modified_after_compilation_cannot_pass(self):
        self.change(mutate=True)
        result = self.run_round("self-modifying")
        self.assertFalse(result["accepted"])
        self.assertEqual(self.state()["best"], "baseline")

    def profile_campaign(self):
        self.spec["profile"] = [["{python}", "{repo}/driver.py", "profile", "{repo}", "{evidence}", "{skill_dir}"]]
        self.spec["build"].append(["{python}", "{skill_dir}/scripts/compare_fixtures.py", "--help"])
        self.events = self.root / "phases.log"
        self.spec["env"] = {"FAKE_PHASE_LOG": str(self.events)}
        write_json(self.config, self.spec)
        self.output = self.root / "profile-campaign"
        campaign.initialize(self.repo, "HEAD", self.config, self.output, {"name": "codex"})
        self.workspace = self.output / "workspace"

    def test_profile_runs_after_abab_under_lock_with_separate_capture_env(self):
        self.profile_campaign()
        self.change()
        with patch.dict(os.environ, {"MTL_CAPTURE_ENABLED": "1"}):
            # The inherited capture flag must also be stripped from CPU builds.
            campaign.cpu_build(self.output, self.workspace)
            result = self.run_round("profile")
        self.assertTrue(result["accepted"], result)
        self.assertEqual(self.events.read_text().splitlines()[-5:], ["bench:A1", "bench:B1", "bench:A2", "bench:B2", "profile"])
        self.assertEqual([item["value"] for item in result["measurements"]], [10, 5, 10, 5])
        self.assertEqual(result["pair_speedups"], [2.0, 2.0])
        self.assertEqual(result["profiling"], {"commands_completed": 1,
                         "included_in_acceptance_timing": False, "gpu_activity_inspected": False})

    def test_failed_profile_prevents_promotion_and_preserves_timings(self):
        self.profile_campaign()
        self.change(profile_failure=True)
        result = self.run_round("failed-profile")
        self.assertEqual(result["status"], "error", result)
        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["measurements"]), 4)
        self.assertEqual(self.state()["best"], "baseline")
        self.assertIn("fake capture failed", (self.output / "rounds/failed-profile/evidence/profile-0.stderr").read_text())
        with exclusive_lock(self.gpu_lock, timeout=0):
            pass

    def test_profile_cannot_change_frozen_helper_and_promote_candidate(self):
        self.profile_campaign()
        self.change(profile_mutate=True)
        result = self.run_round("changed-helper")
        self.assertFalse(result["accepted"], result)
        self.assertIn("copied runtime/skill changed", result["reason"])
        self.assertEqual(self.state()["best"], "baseline")

    def test_runtime_mutation_is_rejected_before_agent_or_gpu(self):
        helper = self.output / "runtime/skills/metal/scripts/compare_fixtures.py"
        helper.write_text("raise SystemExit(0)\n")
        with patch.object(campaign, "run_command") as command, patch.object(campaign, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "copied runtime/skill changed"):
                self.run_round("modified-runtime", agent=True)
            command.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(list((self.output / "rounds").iterdir()), [])

    def test_runtime_mutation_during_coding_is_rejected_before_evaluation(self):
        def fake_agent(command, *, cwd, env, evidence, name, timeout, check):
            (cwd / "src/kernel.metal").write_text(json.dumps({"latency": 5.0}))
            (evidence / "agent.stdout").write_text(
                '{"type":"thread.started","thread_id":"tampering-test"}\n{"type":"turn.completed"}\n')
            (self.output / "runtime/skills/metal/scripts/compare_fixtures.py").write_text("modified by agent")
            return {"returncode": 0, "timed_out": False}

        with patch.object(campaign, "run_command", side_effect=fake_agent), patch.object(campaign, "evaluate") as evaluate:
            result = self.run_round("agent-changed-helper", agent=True)
            evaluate.assert_not_called()
        self.assertFalse(result["accepted"], result)
        self.assertIn("copied runtime/skill changed", result["reason"])

    def test_legacy_campaign_without_runtime_hashes_remains_readable(self):
        state = self.state()
        del state["runtime_sha256"]
        write_json(self.output / "state.json", state)
        self.change()
        result = self.run_round("legacy")
        self.assertTrue(result["accepted"], result)

    def test_external_source_symlink_cannot_enter_evaluation(self):
        self.change()
        outside = self.root / "external.metal"
        outside.write_text("mutable external source")
        (self.workspace / "src/external.metal").symlink_to(outside)
        result = self.run_round("external-link")
        self.assertFalse(result["accepted"])
        self.assertIn("symlink leaves", result["reason"])
        self.assertFalse(result["correctness_passed"])

    def test_documented_layer_exception_still_runs_small_checks(self):
        self.spec["shader_validation"] = {"enabled": False, "static_threadgroup_bytes": 20480,
                                          "reason": "Documented macOS validation-layer doubling"}
        self.spec["env"] = {"FAKE_EXPECT_LAYER": "0"}
        write_json(self.config, self.spec)
        self.output = self.root / "exception-campaign"
        campaign.initialize(self.repo, "HEAD", self.config, self.output, {"name": "codex"})
        self.workspace = self.output / "workspace"
        self.change()
        result = self.run_round("exception")
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["shader_validation"]["static_threadgroup_bytes"], 20480)
        self.assertTrue((self.output / "rounds/exception/evidence/smoke-candidate-0.command.json").exists())

    def test_real_fake_codex_rounds_inherit_best_and_lessons(self):
        binary = self.root / "bin"
        binary.mkdir()
        executable = binary / "codex"
        executable.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
assert sys.argv[1] == "exec" and "--json" in sys.argv
task = json.loads(Path(".ako/task.json").read_text())
prior = task["prior_attempts"]
if prior:
    assert "reduce row loads" in prior[-1]["lessons"]
data = json.loads(Path("src/kernel.metal").read_text())
data["latency"] *= 0.8
Path("src/kernel.metal").write_text(json.dumps(data))
Path(".ako/LESSONS.md").write_text("Hypothesis: reduce row loads. CPU fake only.")
Path("build").mkdir()
Path("build/stale-binary").write_text("must not enter the measured checkout")
print(json.dumps({"type": "thread.started", "thread_id": "native-" + str(len(prior))}))
print(json.dumps({"type": "turn.completed"}))
''')
        executable.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"]}):
            first = self.run_round("agent-one", agent=True)
            second = self.run_round("agent-two", agent=True)
        self.assertTrue(first["accepted"], first)
        self.assertTrue(second["accepted"], second)
        self.assertEqual(second["reference_commit"], first["candidate_commit"])
        self.assertEqual(second["agent_session_id"], "native-1")
        self.assertTrue((self.output / "rounds/agent-one/evidence/agent.stdout").is_file())
        self.assertNotIn(".ako/task.json", campaign.git(self.output / self.state()["best"], "ls-files"))
        self.assertFalse((self.output / self.state()["best"] / "build/stale-binary").exists())

    def test_incomplete_agent_session_cannot_reach_measurement(self):
        with patch.object(campaign, "run_command", return_value={"returncode": 0, "timed_out": False}):
            result = self.run_round("incomplete-agent", agent=True)
        self.assertEqual(result["status"], "error")
        self.assertIn("thread.started", result["reason"])
        self.assertEqual(self.state()["best"], "baseline")

    def test_interruption_records_failure_and_keeps_best(self):
        self.change()
        with patch.object(campaign, "evaluate", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_round("interrupted")
        result = json.loads((self.output / "rounds/interrupted/evidence/result.json").read_text())
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse(result["accepted"])
        self.assertEqual(self.state()["best"], "baseline")

    def test_copied_runtime_is_self_contained(self):
        result = subprocess.run([sys.executable, str(self.output / "runtime/metal_campaign.py"),
                                 "status", str(self.output)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["best"], "baseline")
        rejected = subprocess.run([sys.executable, str(self.output / "runtime/metal_campaign.py"),
                                   "bench", str(self.output), "--label", "unchanged", "--allow-gpu",
                                   "--gpu-lock", str(self.gpu_lock)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 1, rejected.stderr)
        self.assertIn("no_source_change", rejected.stdout)

    def test_config_requires_real_commands_and_documented_layer_exception(self):
        for update in ({"smoke": []}, {"correctness": []}, {"shader_validation": {"enabled": False}},
                       {"env": {"NI_LOCK_HELD": "1"}}, {"env": {"MTL_CAPTURE_ENABLED": "1"}},
                       {"variables": {"skill_dir": "mutable-helper"}}, {"profile": "shell command"},
                       {"profile": [[""]]}, {"profile": [["{unknown_path}"]]},
                       {"profile": [["profiler", "{output}"]]}):
            with self.subTest(update=update):
                write_json(self.config, {**self.spec, **update})
                with self.assertRaises(ValueError):
                    load_spec(self.config)
        write_json(self.config, {**self.spec, "shader_validation": {
            "enabled": False, "static_threadgroup_bytes": 20480, "reason": "Known macOS layer doubling; see kernel report"}})
        self.assertFalse(load_spec(self.config)["shader_validation"]["enabled"])


class ProcessAndMetricTest(unittest.TestCase):
    def test_timeout_kills_benchmark_descendants_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = "import time; from pathlib import Path; time.sleep(0.6); Path('escaped').write_text('bad')"
            parent = f"import subprocess, sys, time; print('started', flush=True); subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(20)"
            with self.assertRaises(RuntimeError):
                run_command([sys.executable, "-c", parent], cwd=root, env=os.environ.copy(),
                            evidence=root, name="timeout", timeout=0.15)
            record = json.loads((root / "timeout.command.json").read_text())
            self.assertTrue(record["timed_out"])
            self.assertIn("started", (root / "timeout.stdout").read_text())
            time.sleep(0.7)
            self.assertFalse((root / "escaped").exists())

    def test_exclusive_lock_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock"
            with exclusive_lock(lock), self.assertRaises(RuntimeError):
                with exclusive_lock(lock, timeout=0):
                    self.fail("must not obtain a second exclusive lock")
            with exclusive_lock(lock, timeout=0):
                pass

    def test_kapitsa_metric_requires_valid_positive_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "measure.json"
            write_json(path, {"invalid": [], "results": {"decode": {"median_tps": 73.0}}})
            self.assertEqual(read_metric(path, "results.decode.median_tps"), 73.0)
            path.write_text('{"results": {"decode": {"median_tps": 73.0}}}')
            with self.assertRaises(ValueError):
                read_metric(path, "results.decode.median_tps")


if __name__ == "__main__":
    unittest.main()
