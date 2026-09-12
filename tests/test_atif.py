"""The run as an ATIF trajectory.

`atif_issues` restates harbor-atif2otel's validate_trajectory, field for field.
The harness does not import Harbor - an instrument that depends on the thing
measuring with it is the dependency backwards - so the contract is written out
here and checked against every trajectory the tests build.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import atif, budget, loop, media, trace
from test_api_retries import ScriptedClient
from test_media_limits import reply

SUPPORTED_SCHEMA_VERSIONS = {f"ATIF-v1.{minor}" for minor in range(8)}
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def atif_issues(trajectory):
    """Harbor's own validation rules, restated. Empty means valid."""
    issues = []
    if not trajectory.get("schema_version"):
        issues.append("missing required field: schema_version")
    elif trajectory["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        issues.append(f"unsupported schema_version: {trajectory['schema_version']}")

    agent = trajectory.get("agent")
    if not isinstance(agent, dict):
        issues.append("missing or invalid required field: agent")
    else:
        if not agent.get("name"):
            issues.append("agent.name is required")
        if not agent.get("version"):
            issues.append("agent.version is required")

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        issues.append("steps must be a non-empty array")
        return issues
    for i, step in enumerate(steps):
        if "step_id" not in step:
            issues.append(f"steps[{i}]: missing step_id")
        if step.get("source") not in ("system", "user", "agent"):
            issues.append(f"steps[{i}]: bad source {step.get('source')!r}")
        if "message" not in step:
            issues.append(f"steps[{i}]: missing required field: message")
        calls = step.get("tool_calls", [])
        if not isinstance(calls, list):
            issues.append(f"steps[{i}].tool_calls: must be an array")
            continue
        for j, call in enumerate(calls):
            if not call.get("tool_call_id"):
                issues.append(f"steps[{i}].tool_calls[{j}]: missing tool_call_id")
            if not call.get("function_name"):
                issues.append(f"steps[{i}].tool_calls[{j}]: missing function_name")
    return issues


class StallOnce:
    """Stands in for the progress tracker: stalls on the turn it is told to,
    pointing at the message that turn began on."""

    stall_reason = "repeat"

    def __init__(self, on_turn, loop_start_idx):
        self.on_turn = on_turn
        self.loop_start_idx = loop_start_idx
        self.turns = 0

    def record(self, actions, loop_start):
        self.turns += 1
        return "repeat" if self.turns == self.on_turn else None

    def reset_after_rewind(self):
        pass

    def state(self):
        return {}


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.runs = Path(self.tmp.name) / "runs"
        self.trajectory_path = self.runs / "trajectory.json"

    def run_with(self, script, progress=None, **kwargs):
        run = loop.Run(client=ScriptedClient(script), model="muse-spark-1.2-contributor",
                       system="system prompt", instruction="edit the video",
                       work_dir=self.work, output_path=None, budget=budget.Budget(),
                       trace=trace.Trace(self.runs / "trace.jsonl"),
                       trajectory_path=self.trajectory_path,
                       messages_path=self.runs / "messages.json",
                       agent_version="abc1234", **kwargs)
        if progress is not None:
            run.progress = progress
        run.run()
        return run, json.loads(self.trajectory_path.read_text())

    def test_a_finished_run_is_a_valid_atif_trajectory(self):
        _, written = self.run_with([reply(("shell", {"command": "echo hi"})),
                                    reply(text="done")])

        self.assertEqual(atif_issues(written), [])
        self.assertEqual(written["schema_version"], "ATIF-v1.7")
        self.assertEqual(written["agent"], {"name": "videoagent-harness",
                                            "version": "abc1234",
                                            "model_name": "muse-spark-1.2-contributor"})
        self.assertEqual([step["source"] for step in written["steps"]],
                         ["system", "user", "agent", "agent"])

    def test_a_tool_result_folds_into_the_step_that_asked_for_it(self):
        _, written = self.run_with([reply(("shell", {"command": "echo hi"})),
                                    reply(text="done")])

        step = written["steps"][2]
        call, = step["tool_calls"]
        self.assertEqual((call["function_name"], call["arguments"]),
                         ("shell", {"command": "echo hi"}))
        result, = step["observation"]["results"]
        self.assertEqual(result["source_call_id"], call["tool_call_id"])
        self.assertEqual(result["content"], "hi")
        self.assertNotIn("tool", [s["source"] for s in written["steps"]])

    def test_a_turn_carries_its_tokens_and_the_run_its_totals(self):
        _, written = self.run_with([reply(("shell", {"command": "true"})),
                                    reply(text="done")])

        self.assertEqual(written["steps"][2]["metrics"],
                         {"prompt_tokens": 1, "completion_tokens": 1})
        self.assertEqual(written["final_metrics"]["total_prompt_tokens"], 2)
        self.assertEqual(written["final_metrics"]["total_steps"], len(written["steps"]))
        self.assertIn("termination=ABANDONED", written["notes"])

    def test_every_step_is_stamped_in_utc(self):
        _, written = self.run_with([reply(text="done")])
        for step in written["steps"]:
            self.assertRegex(step["timestamp"], ISO_UTC)

    def test_what_was_seen_is_described_rather_than_embedded(self):
        frame = self.work / "frame.jpg"
        frame.write_bytes(b"not really a jpeg")
        with mock.patch.object(media, "probe_duration", return_value=None):
            _, written = self.run_with([reply(("view", {"paths": [str(frame)]})),
                                        reply(text="done")])

        seen, = [step for step in written["steps"]
                 if step["source"] == "user" and "image_url" in step["message"]]
        self.assertIn("omitted from the trajectory", seen["message"])
        self.assertNotIn("base64", seen["message"])
        self.assertEqual(atif_issues(written), [])

    def test_a_rewound_turn_leaves_no_step_behind(self):
        run, written = self.run_with(
            [reply(("shell", {"command": "echo stuck"})),
             reply(("shell", {"command": "echo after"})), reply(text="done")],
            progress=StallOnce(on_turn=1, loop_start_idx=2))

        self.assertEqual(run.rewinds, 1)
        self.assertEqual(len(run.messages), len(run.records))
        self.assertNotIn("echo stuck", json.dumps(written))
        self.assertEqual(atif_issues(written), [])

    def test_arguments_that_were_not_json_are_still_recorded(self):
        broken = reply(("shell", {"command": "ok"}))
        broken.choices[0].message.tool_calls[0].function.arguments = "{not json"
        _, written = self.run_with([broken, reply(text="done")])

        call, = written["steps"][2]["tool_calls"]
        self.assertEqual(call["arguments"], {"raw_arguments": "{not json"})
        self.assertEqual(atif_issues(written), [])


class BuildTest(unittest.TestCase):
    def test_a_result_whose_call_was_rewound_away_is_dropped(self):
        messages = [{"role": "tool", "tool_call_id": "gone", "content": "orphan"}]
        self.assertEqual(atif.steps(messages, [{}]), [])

    def test_an_empty_run_still_says_what_it_was(self):
        built = atif.build([], [], model="m", version="v")
        self.assertEqual(built["final_metrics"]["total_steps"], 0)
        self.assertEqual(built["agent"]["version"], "v")
        self.assertTrue(built["session_id"])


if __name__ == "__main__":
    unittest.main()
