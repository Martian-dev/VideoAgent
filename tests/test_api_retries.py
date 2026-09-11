"""A failed model request - a 500, a 400, a dropped connection - is retried with
exponential backoff, and only a request that fails every retry ends the run.
None of it may outlast the run's wall clock."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import budget, loop, trace
from test_media_limits import ApiError, reply

SERVER_ERROR = ApiError(500, "Error code: 500 - Internal server error")


class ScriptedClient:
    """Answers each request with the next item: an exception is raised, anything
    else is returned."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @property
    def calls(self):
        return len(self.requests)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ApiRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.trace_path = Path(self.tmp.name) / "runs" / "trace.jsonl"
        patcher = mock.patch.object(loop.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, script, max_seconds=3600, **kwargs):
        client = ScriptedClient(script)
        run = loop.Run(client=client, model="fake", system="system",
                       instruction="task", work_dir=self.work, output_path=None,
                       budget=budget.Budget(max_seconds=max_seconds),
                       trace=trace.Trace(self.trace_path), **kwargs)
        return client, run

    def events(self, kind):
        lines = self.trace_path.read_text().splitlines()
        return [event for event in map(json.loads, lines) if event["kind"] == kind]

    def delays(self):
        return [call.args[0] for call in self.sleep.call_args_list]

    def test_a_server_error_that_clears_is_retried_and_the_run_carries_on(self):
        client, run = self.run_with([SERVER_ERROR] * 3 + [reply(text="done")])
        _, termination = run.run()

        self.assertEqual(client.calls, 4)
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual(self.delays(), [15, 30, 60])
        self.assertEqual([(e["attempt"], e["status"]) for e in self.events("api_retry")],
                         [(1, 500), (2, 500), (3, 500)])
        end, = self.events("end")
        self.assertEqual(end["api_retries"], 3)

    def test_the_fifth_retry_can_still_rescue_the_run(self):
        client, run = self.run_with([SERVER_ERROR] * 5 + [reply(text="done")])
        _, termination = run.run()

        self.assertEqual(client.calls, 6)
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual(self.delays(), [15, 30, 60, 120, 240])

    def test_a_request_that_fails_every_retry_ends_the_run(self):
        client, run = self.run_with([SERVER_ERROR] * 6)
        with self.assertRaises(ApiError):
            run.run()

        self.assertEqual(client.calls, 6)
        self.assertEqual(self.delays(), [15, 30, 60, 120, 240])
        end, = self.events("end")
        self.assertEqual((end["termination"], end["api_retries"]), ("error:ApiError", 5))

    def test_bad_requests_and_dropped_connections_are_retried_too(self):
        client, run = self.run_with([ApiError(400, "Bad request"),
                                     ConnectionError("connection reset by peer"),
                                     reply(text="done")])
        _, termination = run.run()

        self.assertEqual(client.calls, 3)
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual([e["error"] for e in self.events("api_retry")],
                         ["ApiError", "ConnectionError"])

    def test_every_request_gets_its_own_retries(self):
        client, run = self.run_with(
            [SERVER_ERROR] * 4 + [reply(("shell", {"command": "true"}))]
            + [SERVER_ERROR] * 4 + [reply(text="done")])
        _, termination = run.run()

        self.assertEqual(termination, "ABANDONED")
        self.assertEqual(self.delays(), [15, 30, 60, 120] * 2)
        end, = self.events("end")
        self.assertEqual(end["api_retries"], 8)

    def test_the_number_of_retries_and_the_backoff_are_configurable(self):
        client, run = self.run_with([SERVER_ERROR] * 3, api_retries=2, retry_backoff=1)
        with self.assertRaises(ApiError):
            run.run()

        self.assertEqual(client.calls, 3)
        self.assertEqual(self.delays(), [1, 2])

    def test_a_request_gets_the_request_timeout_when_there_is_time(self):
        client, run = self.run_with([reply(text="done")])
        run.run()
        self.assertEqual(client.requests[0]["timeout"], loop.REQUEST_TIMEOUT_S)

    def test_a_request_timeout_is_capped_at_the_time_left(self):
        client, run = self.run_with([reply(text="done")], max_seconds=100)
        run.run()
        self.assertTrue(99 < client.requests[0]["timeout"] <= 100)

    def test_a_retry_the_time_left_cannot_wait_for_ends_the_run_as_a_timeout(self):
        # 20s left: the 15s wait fits, the 30s one does not.
        client, run = self.run_with([SERVER_ERROR, SERVER_ERROR, reply(text="done")],
                                    max_seconds=20)
        _, termination = run.run()

        self.assertEqual(client.calls, 2)
        self.assertEqual(termination, "TIMEOUT")
        self.assertEqual(self.delays(), [15])
        gave_up, = self.events("api_retry_out_of_time")
        self.assertEqual((gave_up["attempt"], gave_up["delay_s"]), (2, 30))

    def test_no_request_is_made_once_the_time_is_up(self):
        client, run = self.run_with([], max_seconds=0)
        _, termination = run.run()
        self.assertEqual((client.calls, termination), (0, "TIMEOUT"))


if __name__ == "__main__":
    unittest.main()
