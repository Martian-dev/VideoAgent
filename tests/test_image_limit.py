"""The per-request image limit: eviction by image count, paged delivery of
large views, and recovering from the API's 400 instead of ending the run."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import budget, loop, media, trace


def image_message(n, tag):
    content = [{"type": "text", "text": tag}]
    content += [{"type": "image_url", "image_url": {"url": f"{tag}-{i}"}}
                for i in range(n)]
    return {"role": "user", "content": content}


def image_urls(message):
    return [part["image_url"]["url"] for part in message["content"]
            if part.get("type") == "image_url"]


def media_blocks(messages):
    return sum(1 for message in messages if isinstance(message.get("content"), list)
               for part in message["content"] if part.get("type") in media.MEDIA_TYPES)


class EvictTest(unittest.TestCase):
    def test_images_are_capped_across_turns_newest_first(self):
        messages = [image_message(15, f"look{i}") for i in range(4)]
        media.evict(messages, keep_last=4, max_images=45)
        self.assertEqual(media.count_images(messages), 45)
        self.assertEqual(image_urls(messages[0]), [])
        self.assertEqual(len(image_urls(messages[1])), 15)

    def test_the_boundary_can_fall_inside_one_look(self):
        messages = [image_message(15, f"look{i}") for i in range(4)]
        media.evict(messages, keep_last=4, max_images=40)
        self.assertEqual(image_urls(messages[1]), [f"look1-{i}" for i in range(5, 15)])

    def test_keep_last_still_bounds_the_number_of_looks(self):
        messages = [image_message(1, f"look{i}") for i in range(6)]
        media.evict(messages, keep_last=4, max_images=45)
        self.assertEqual(media.count_images(messages), 4)

    def test_videos_do_not_count_against_the_image_limit(self):
        message = image_message(45, "look")
        message["content"] += [{"type": "video_url", "video_url": {"url": "v"}}] * 2
        self.assertEqual(media.evict([message], keep_last=4, max_images=45), 0)


def look(kind, n):
    return media.Look(path=f"/work/{kind}{n}", kind=kind)


class DeliveriesTest(unittest.TestCase):
    def test_a_large_view_is_paged(self):
        queue = media.Deliveries(max_images=45)
        note = queue.add([look("image", n) for n in range(100)])
        self.assertIn("45 of these in the message below, the rest over the next 2 turns",
                      note)
        self.assertEqual([len(queue.release()) for _ in range(3)], [45, 45, 10])
        self.assertFalse(queue)

    def test_a_small_view_goes_out_at_once(self):
        queue = media.Deliveries(max_images=45)
        self.assertIsNone(queue.add([look("image", n) for n in range(45)]))
        self.assertEqual(len(queue.release()), 45)

    def test_a_view_behind_a_full_batch_says_when_it_arrives(self):
        queue = media.Deliveries(max_images=45)
        queue.add([look("image", n) for n in range(50)])
        note = queue.add([look("image", n) for n in range(10)])
        self.assertIn("these arrive 1 turn from now", note)

    def test_videos_ride_along_uncounted(self):
        queue = media.Deliveries(max_images=45)
        queue.add([look("image", n) for n in range(45)] + [look("video", 0)])
        self.assertEqual(len(queue.release()), 46)


class ImageLimitParseTest(unittest.TestCase):
    def test_the_api_error_is_recognised(self):
        error = ApiError(400, "Error code: 400 - {'error': {'message': 'request "
                              "contains 60 images, exceeding the maximum of 50 "
                              "allowed per request'}}")
        self.assertEqual(loop.image_limit(error), (60, 50))

    def test_other_errors_are_not(self):
        self.assertIsNone(loop.image_limit(ApiError(400, "invalid tool schema")))
        self.assertIsNone(loop.image_limit(ApiError(500, "maximum of 50 images")))


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        dumped = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            dumped["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.function.name,
                              "arguments": call.function.arguments}}
                for call in self.tool_calls]
        return dumped


def reply(*calls, text=None):
    tool_calls = [SimpleNamespace(id=f"call_{n}", function=SimpleNamespace(
                      name=name, arguments=json.dumps(arguments)))
                  for n, (name, arguments) in enumerate(calls)] or None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=FakeMessage(text, tool_calls),
                                 finish_reason="tool_calls" if tool_calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


class FakeClient:
    """Stands in for the Meta Model API, including its image limit."""

    def __init__(self, replies, limit=50, counts_video=False):
        self.replies = list(replies)
        self.limit = limit
        self.kinds = {"image_url", "video_url"} if counts_video else {"image_url"}
        self.accepted, self.refused = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, **_):
        n = sum(1 for message in messages if isinstance(message.get("content"), list)
                for part in message["content"] if part.get("type") in self.kinds)
        if n > self.limit:
            self.refused.append(n)
            raise ApiError(400, f"request contains {n} images, exceeding the "
                                f"maximum of {self.limit} allowed per request")
        self.accepted.append(n)
        return self.replies.pop(0)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.trace_path = Path(self.tmp.name) / "runs" / "trace.jsonl"
        patcher = mock.patch.object(media, "probe_duration", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def files(self, n, suffix=".jpg", start=0):
        paths = []
        for i in range(start, start + n):
            path = self.work / f"frame{i:03d}{suffix}"
            path.write_bytes(b"\xff\xd8\xff\xe0 not really a frame")
            paths.append(str(path))
        return paths

    def run_with(self, client, **kwargs):
        run = loop.Run(client=client, model="fake", system="system",
                       instruction="task", work_dir=self.work, output_path=None,
                       budget=budget.Budget(), trace=trace.Trace(self.trace_path),
                       **kwargs)
        return run, run.run()

    def events(self, kind):
        lines = self.trace_path.read_text().splitlines()
        return [event for event in map(json.loads, lines) if event["kind"] == kind]

    def test_a_view_over_the_limit_is_paged_across_turns(self):
        client = FakeClient([reply(("view", {"paths": self.files(100)})),
                             reply(text="noted"), reply(text="noted"),
                             reply(text="done")])
        run, (submitted, termination) = self.run_with(client)

        self.assertEqual(client.accepted, [0, 45, 45, 45])
        self.assertEqual(client.refused, [])
        self.assertEqual(client.replies, [])
        self.assertEqual(termination, "ABANDONED")
        stub = next(m["content"] for m in run.messages if m["role"] == "tool")
        self.assertIn("the rest over the next 2 turns", stub)
        headers = [m["content"][0]["text"] for m in run.messages
                   if m["role"] == "user" and isinstance(m["content"], list)]
        self.assertIn("(55 more images queued", headers[0])
        self.assertEqual([e["images"] for e in self.events("deliver")], [45, 45, 10])

    def test_a_refused_request_lowers_the_bound_and_retries(self):
        client = FakeClient([reply(("view", {"paths": self.files(30)})),
                             reply(text="done")], limit=10)
        run, (_, termination) = self.run_with(client)

        self.assertEqual(client.refused, [30])
        self.assertEqual(client.accepted, [0, 10])
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual((run.max_images, run.deliveries.max_images), (10, 10))
        retry, = self.events("image_limit_retry")
        self.assertEqual((retry["sent"], retry["api_limit"], retry["dropped"]),
                         (30, 10, 20))
        self.assertIn("at most 10 images per request", json.dumps(run.messages))

    def test_when_the_api_counts_what_we_do_not_every_image_is_dropped(self):
        paths = self.files(3) + self.files(3, suffix=".mp4", start=3)
        client = FakeClient([reply(("view", {"paths": paths})), reply(text="done")],
                            limit=5, counts_video=True)
        run, (_, termination) = self.run_with(client)

        self.assertEqual(client.refused, [6])
        self.assertEqual(client.accepted, [0, 0])
        self.assertEqual(media_blocks(run.messages), 0)
        self.assertEqual(termination, "ABANDONED")

    def test_other_bad_requests_still_raise(self):
        client = FakeClient([])
        client.chat.completions.create = mock.Mock(
            side_effect=ApiError(400, "invalid tool schema"))
        run = loop.Run(client=client, model="fake", system="system",
                       instruction="task", work_dir=self.work, output_path=None,
                       budget=budget.Budget(), trace=trace.Trace(self.trace_path))
        with self.assertRaises(ApiError):
            run.run()


if __name__ == "__main__":
    unittest.main()
