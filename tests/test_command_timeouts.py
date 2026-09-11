"""No command outlasts the run: each is capped at the time left, and a timeout
kills everything the command started, not just its shell."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from harness import budget, loop, tools, trace
from test_api_retries import ScriptedClient
from test_media_limits import reply


def gone(pid, wait_s=5):
    """True once pid has exited; a zombie awaiting its reaper counts."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.1)
    return False


class RunShellTest(unittest.TestCase):
    def test_a_timeout_kills_what_the_command_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            output, ok = tools.run_shell("sleep 30 & echo $! > child; wait", tmp,
                                         timeout=1)
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual((output, ok), ("[timed out after 1s]", False))
            self.assertTrue(gone(int(Path(tmp, "child").read_text())))

    def test_a_command_that_finishes_is_reported_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, ok = tools.run_shell("echo out; echo err >&2; exit 3", tmp, timeout=10)
        self.assertEqual((output, ok), ("out\n[stderr]\nerr\n[exit 3]", False))


class ShellCapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.trace_path = Path(self.tmp.name) / "runs" / "trace.jsonl"

    def run_with(self, script, max_seconds=3600, **kwargs):
        run = loop.Run(client=ScriptedClient(script), model="fake", system="system",
                       instruction="task", work_dir=self.work, output_path=None,
                       budget=budget.Budget(max_seconds=max_seconds),
                       trace=trace.Trace(self.trace_path), **kwargs)
        return run

    def shell_event(self):
        lines = self.trace_path.read_text().splitlines()
        event, = [e for e in map(json.loads, lines) if e["kind"] == "shell"]
        return event

    def tool_result(self, run):
        return next(m["content"] for m in run.messages if m["role"] == "tool")

    def test_a_command_is_cut_off_when_the_run_is(self):
        run = self.run_with([reply(("shell", {"command": "sleep 30"}))], max_seconds=2)
        started = time.monotonic()
        _, termination = run.run()

        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(termination, "TIMEOUT")
        self.assertLessEqual(self.shell_event()["timeout_s"], 2)
        self.assertIn("[timed out after", self.tool_result(run))

    def test_the_command_timeout_applies_when_there_is_time(self):
        run = self.run_with([reply(("shell", {"command": "sleep 30"})), reply(text="done")],
                            command_timeout=1)
        _, termination = run.run()

        self.assertEqual(termination, "ABANDONED")
        event = self.shell_event()
        self.assertEqual((event["timeout_s"], event["ok"]), (1, False))
        self.assertEqual(self.tool_result(run), "[timed out after 1s]")

    def test_nothing_is_run_once_the_time_is_up(self):
        run = self.run_with([reply(("shell", {"command": "touch ran"})), reply(text="done")])
        run.budget.remaining = lambda: 0.5
        run.run()

        self.assertFalse((self.work / "ran").exists())
        self.assertEqual(self.tool_result(run), "[not run: the run's time is up]")


if __name__ == "__main__":
    unittest.main()
