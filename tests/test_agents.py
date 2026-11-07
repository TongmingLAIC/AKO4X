"""CPU-only regression checks for workspace creation and agent lifecycle."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spawn
from agent_runtime import codex_outcome, launch_command
from master import master
from scripts.advisory_review import review_message

ROOT = Path(__file__).resolve().parent.parent


def events(*values):
    return "\n".join(json.dumps(v) for v in values) + "\n"


class AgentLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".ako").mkdir()
        (self.root / ".ako/agent.json").write_text(json.dumps({"name": "codex"}))
        (self.root / "solution").mkdir()
        (self.root / "solution/kernel.py").write_text("def run(): return 1\n")

    def test_codex_lifecycle_and_exact_resume(self):
        transcript = events({"type": "thread.started", "thread_id": "thread-123"},
                            {"type": "turn.completed"})
        with patch.object(master, "_run_bounded", return_value=(transcript, "", 0, False)) as run, \
                patch.object(master, "_git_diff_head", return_value=""):
            result = master.run_sub_phase1(self.root, "optimize")
            self.assertEqual(result.exit_status, 0)
            self.assertEqual(result.session_id, "thread-123")
            self.assertEqual(run.call_args.args[0][0:2], ["codex", "exec"])
            self.assertNotIn("--session-id", run.call_args.args[0])
            self.assertEqual((self.root / ".ako/session-id.txt").read_text().strip(), result.session_id)
            (self.root / "PROPOSALS.md").write_text("none\n")
            phase2 = master.send_retrospective_prompt(self.root, result.session_id)
            self.assertEqual(phase2.exit_status, 0)
            command = run.call_args.args[0]
            self.assertIn("resume", command)
            self.assertIn(result.session_id, command)
            self.assertIn("AGENTS.md", command[-1])
            self.assertNotIn(".claude/skills/", command[-1])

    def test_failed_or_incomplete_json_is_not_a_successful_round(self):
        for transcript in ["", "not json", events({"type": "turn.completed"}),
                           events({"type": "thread.started", "thread_id": "abc"}),
                           events({"type": "thread.started", "thread_id": "abc"},
                                  {"type": "turn.failed", "error": {"message": "quota"}},
                                  {"type": "turn.completed"})]:
            with self.subTest(transcript=transcript), \
                    patch.object(master, "_run_bounded", return_value=(transcript, "", 0, False)), \
                    patch.object(master, "_git_diff_head", return_value=""):
                self.assertNotEqual(master.run_sub_phase1(self.root, "test").exit_status, 0)

    def test_timeout_preserves_events_and_actual_session(self):
        transcript = events({"type": "thread.started", "thread_id": "partial"})
        with patch.object(master, "_run_bounded", return_value=(transcript, "timeout", -1, True)), \
                patch.object(master, "_git_diff_head", return_value=""):
            result = master.run_sub_phase1(self.root, "test")
        self.assertTrue(result.timed_out)
        self.assertEqual(result.session_id, "partial")
        self.assertEqual(result.transcript_path.read_text(), transcript)
        self.assertIn("timeout", result.stderr_tail)

    def test_resume_rejects_wrong_or_missing_id(self):
        (self.root / ".ako/session-id.txt").write_text("actual\n")
        for session in ["", "other"]:
            with self.assertRaises(ValueError):
                master.send_retrospective_prompt(self.root, session)
        _, error = codex_outcome(events({"type": "thread.started", "thread_id": "other"},
                                        {"type": "turn.completed"}), "actual")
        self.assertTrue(error)

    def test_legacy_claude_workspace(self):
        (self.root / ".ako/agent.json").unlink()
        with patch.object(master, "_run_bounded", return_value=("", "", 0, False)) as run, \
                patch.object(master, "_git_diff_head", return_value=""):
            result = master.run_sub_phase1(self.root, "test")
        self.assertEqual(result.exit_status, 0)
        self.assertEqual(run.call_args.args[0][0], "claude")
        self.assertIn(result.session_id, run.call_args.args[0])

    def test_no_sandbox_bypass_is_added(self):
        for resume in (False, True):
            command = launch_command({"name": "codex"}, "task", "id", resume=resume)
            self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)

    def test_real_process_transport_with_fake_codex(self):
        binary = self.root / "bin"
        binary.mkdir()
        executable = binary / "codex"
        executable.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "exec"
assert "--json" in args
if "resume" in args:
    assert "integration-thread" in args
    Path("PROPOSALS.md").write_text("none\\n")
print(json.dumps({"type": "thread.started", "thread_id": "integration-thread"}))
print(json.dumps({"type": "turn.completed"}))
''')
        executable.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"]}), \
                patch.object(master, "_git_diff_head", return_value=""):
            result = master.run_sub_phase1(self.root, "transport test")
            self.assertEqual(result.exit_status, 0)
            second = master.send_retrospective_prompt(self.root, result.session_id)
        self.assertEqual(second.exit_status, 0)
        self.assertEqual(second.proposals_md_path.read_text(), "none\n")

    def test_missing_agent_binary_is_recorded_as_failure(self):
        stdout, stderr, status, timed_out = master._run_bounded(
            [str(self.root / "missing-binary")], cwd=self.root, timeout=1)
        self.assertEqual(status, 127)
        self.assertFalse(timed_out)
        self.assertTrue(stderr)


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ako agent test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        for directory in ("definitions/test", "workloads/test"):
            (self.dataset / directory).mkdir(parents=True)
        (self.dataset / "definitions/test/probe.json").write_text(json.dumps({
            "name": "probe", "reference": "def run(x): return x\n"}))
        (self.dataset / "workloads/test/probe.jsonl").write_text('{}\n')

    def spawn(self, agent):
        with patch.object(spawn, "BASE_DIR", self.root), patch.object(sys, "argv", [
            "spawn.py", "--operator", "probe", "--dataset", str(self.dataset),
            "--gpu", "b200", "--backend", "local", "--agent", agent, "--name", agent,
        ]):
            spawn.main()
        return next(self.root.glob(f"ako4x-run-{agent}*"))

    def test_codex_workspace_has_discoverable_instructions_and_skills(self):
        child = self.spawn("codex")
        self.assertTrue((child / "AGENTS.md").is_file())
        self.assertTrue((child / "scripts/AGENTS.md").is_file())
        self.assertTrue((child / ".agents/skills/bench/SKILL.md").is_file())
        self.assertFalse((child / ".claude").exists())
        self.assertFalse((child / ".codex/config.toml").exists())
        self.assertEqual(json.loads((child / ".ako/agent.json").read_text())["name"], "codex")
        self.assertTrue((child / "docs/iteration-review.md").is_file())
        # A failing benchmark must stay failed when the advisory runs.
        (child / "scripts/run_local.py").write_text("raise SystemExit(7)\n")
        env = dict(os.environ, AKO_PYTHON=sys.executable)
        result = subprocess.run(["bash", "scripts/bench.sh", "--label", "test"], cwd=child, env=env)
        self.assertEqual(result.returncode, 7)

    def test_claude_workspace_retains_hook(self):
        child = self.spawn("claude")
        self.assertTrue((child / "CLAUDE.md").is_file())
        self.assertTrue((child / ".claude/skills/bench/SKILL.md").is_file())
        self.assertTrue((child / ".claude/settings.local.json").is_file())
        self.assertFalse((child / "AGENTS.md").exists())

    def test_review_frequency_and_disabled_setting(self):
        (self.root / "docs").mkdir()
        (self.root / "docs/iteration-review.md").write_text("Review the evidence.")
        (self.root / "config.toml").write_text("[advisory]\nfrequency = 2\n")
        (self.root / "trajectory/20260101_120000_a").mkdir(parents=True)
        self.assertEqual(review_message(self.root, ["--label", "a"]), "")
        (self.root / "trajectory/20260101_120001_b").mkdir()
        self.assertIn("Review the evidence", review_message(self.root, ["--label=b"]))
        self.assertEqual(review_message(self.root, ["--ab-compare", "b"]), "")
        (self.root / "config.toml").write_text("[advisory]\nfrequency = 0\n")
        self.assertEqual(review_message(self.root, ["--label", "b"]), "")

    def test_campaign_records_codex_for_later_children(self):
        spec = importlib.util.spec_from_file_location("campaign_start", ROOT / "scripts/campaign_start.py")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        (self.root / "master").mkdir()
        (self.root / "master/MASTER.md").write_text("Read CLAUDE.md and .claude/skills/bench/.\n")
        (self.root / "README.md").write_text("original\n")
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], cwd=self.root, check=True)
        with patch.object(campaign, "REPO_ROOT", self.root), \
                patch.object(campaign, "dataset_preflight"), patch.object(sys, "argv", [
                    "campaign_start.py", "--operator", "probe", "--agent", "codex"]):
            campaign.main()
        self.assertIn("AGENTS.md", (self.root / "AGENTS.md").read_text())
        self.assertEqual(json.loads((self.root / "campaign-agent.json").read_text())["name"], "codex")
        with patch.object(master, "ROOT", self.root), patch.object(master, "_read_campaign_env", return_value=("b200", "local")), \
                patch.object(master, "_run_bounded", return_value=(f"  Path: {self.root}\n", "", 0, False)) as run:
            master.spawn_child("probe", None, "round-1", family="probe")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--agent") + 1], "codex")
        self.assertEqual(command[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
