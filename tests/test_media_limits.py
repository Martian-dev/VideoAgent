"""The per-request media limits - 50 images, 5 videos: eviction by count, paged
delivery of large views, and recovering from the API's 400 instead of ending
the run."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import budget, loop, media, trace


def media_message(n, tag, kind="image_url"):
    content = [{"type": "text", "text": tag}]
    content += [{"type": kind, kind: {"url": f"{tag}-{i}"}} for i in range(n)]
    return {"role": "user", "content": content}


def urls(message, kind="image_url"):
    return [part[kind]["url"] for part in message["content"]
            if part.get("type") == kind]


class EvictTest(unittest.TestCase):
    def test_images_are_capped_across_turns_newest_first(self):
        messages = [media_message(15, f"look{i}") for i in range(4)]
        media.evict(messages, keep_last=4, max_images=45)
        self.assertEqual(media.count_media(messages), (45, 0))
        self.assertEqual(urls(messages[0]), [])
        self.assertEqual(len(urls(messages[1])), 15)

    def test_the_boundary_can_fall_inside_one_look(self):
        messages = [media_message(15, f"look{i}") for i in range(4)]
        media.evict(messages, keep_last=4, max_images=40)
        self.assertEqual(urls(messages[1]), [f"look1-{i}" for i in range(5, 15)])

    def test_keep_last_still_bounds_the_number_of_looks(self):
        messages = [media_message(1, f"look{i}") for i in range(6)]
        media.evict(messages, keep_last=4, max_images=45)
        self.assertEqual(media.count_media(messages), (4, 0))

    def test_images_and_videos_have_separate_budgets(self):
        message = media_message(45, "frame")
        message["content"] += media_message(7, "clip", "video_url")["content"][1:]
        media.evict([message], keep_last=4, max_images=45, max_videos=5)
        self.assertEqual(media.count_media([message]), (45, 5))
        self.assertEqual(urls(message, "video_url"), [f"clip-{i}" for i in range(2, 7)])


def looks(kind, n):
    return [media.Look(path=f"/work/{kind}{i}", kind=kind) for i in range(n)]


class DeliveriesTest(unittest.TestCase):
    def test_a_large_image_view_is_paged(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        note = queue.add(looks("image", 100))
        self.assertIn("45 of these in the message below, the rest over the next 2 turns",
                      note)
        self.assertEqual([len(queue.release()) for _ in range(3)], [45, 45, 10])
        self.assertFalse(queue)

    def test_videos_are_paged_at_their_own_limit(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        note = queue.add(looks("video", 12))
        self.assertIn("5 of these in the message below, the rest over the next 2 turns",
                      note)
        self.assertEqual([len(queue.release()) for _ in range(3)], [5, 5, 2])

    def test_a_small_view_goes_out_at_once(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        self.assertIsNone(queue.add(looks("image", 45) + looks("video", 5)))
        self.assertEqual(len(queue.release()), 50)

    def test_a_view_behind_a_full_batch_says_when_it_arrives(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        queue.add(looks("image", 50))
        self.assertIn("these arrive 1 turn from now", queue.add(looks("image", 10)))

    def test_a_batch_closes_at_whichever_cap_is_hit_first(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        queue.add(looks("image", 3) + looks("video", 7))
        self.assertEqual(len(queue.release()), 8)
        self.assertEqual(queue.queued(), "2 more videos")

    def test_what_is_still_queued_is_described(self):
        queue = media.Deliveries(max_images=45, max_videos=5)
        queue.add(looks("image", 50) + looks("video", 7))
        queue.release()
        self.assertEqual(queue.queued(), "5 more images and 7 more videos")


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class MediaLimitParseTest(unittest.TestCase):
    def test_the_image_error_is_recognised(self):
        error = ApiError(400, "Error code: 400 - {'error': {'message': 'request "
                              "contains 60 images, exceeding the maximum of 50 "
                              "allowed per request'}}")
        self.assertEqual(loop.media_limit(error), ("image", 60, 50))

    def test_the_video_error_is_recognised(self):
        error = ApiError(400, "request contains 6 videos, exceeding the maximum "
                              "of 5 allowed per request")
        self.assertEqual(loop.media_limit(error), ("video", 6, 5))

    def test_other_errors_are_not(self):
        self.assertIsNone(loop.media_limit(ApiError(400, "invalid tool schema")))
        self.assertIsNone(loop.media_limit(ApiError(500, "maximum of 50 images")))


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
    """Stands in for the Meta Model API, including its media limits."""

    def __init__(self, replies, image_limit=50, video_limit=5,
                 videos_count_as_images=False):
        self.replies = list(replies)
        self.image_limit, self.video_limit = image_limit, video_limit
        self.videos_count_as_images = videos_count_as_images
        self.accepted, self.refused = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, **_):
        images, videos = media.count_media(messages)
        if self.videos_count_as_images:
            images += videos
        if images > self.image_limit or videos > self.video_limit:
            self.refused.append((images, videos))
            n, kind, limit = ((images, "images", self.image_limit)
                              if images > self.image_limit
                              else (videos, "videos", self.video_limit))
            raise ApiError(400, f"request contains {n} {kind}, exceeding the "
                                f"maximum of {limit} allowed per request")
        self.accepted.append((images, videos))
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
            path = self.work / f"file{i:03d}{suffix}"
            path.write_bytes(b"not really media")
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

    def test_a_view_over_the_image_limit_is_paged_across_turns(self):
        client = FakeClient([reply(("view", {"paths": self.files(100)})),
                             reply(text="noted"), reply(text="noted"),
                             reply(text="done")])
        run, (_, termination) = self.run_with(client)

        self.assertEqual(client.accepted, [(0, 0), (45, 0), (45, 0), (45, 0)])
        self.assertEqual(client.refused, [])
        self.assertEqual(client.replies, [])
        self.assertEqual(termination, "ABANDONED")
        stub = next(m["content"] for m in run.messages if m["role"] == "tool")
        self.assertIn("the rest over the next 2 turns", stub)
        headers = [m["content"][0]["text"] for m in run.messages
                   if m["role"] == "user" and isinstance(m["content"], list)]
        self.assertIn("(55 more images queued", headers[0])
        self.assertEqual([e["images"] for e in self.events("deliver")], [45, 45, 10])

    def test_a_view_over_the_video_limit_is_paged_across_turns(self):
        client = FakeClient([reply(("view", {"paths": self.files(12, ".mp4")})),
                             reply(text="noted"), reply(text="noted"),
                             reply(text="done")])
        _, (_, termination) = self.run_with(client)

        self.assertEqual(client.accepted, [(0, 0), (0, 5), (0, 5), (0, 5)])
        self.assertEqual(client.refused, [])
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual([e["videos"] for e in self.events("deliver")], [5, 5, 2])

    def test_a_lower_image_limit_is_learned_and_retried(self):
        client = FakeClient([reply(("view", {"paths": self.files(30)})),
                             reply(text="done")], image_limit=10)
        run, (_, termination) = self.run_with(client)

        self.assertEqual(client.refused, [(30, 0)])
        self.assertEqual(client.accepted, [(0, 0), (10, 0)])
        self.assertEqual(termination, "ABANDONED")
        self.assertEqual((run.max_images, run.deliveries.max_images), (10, 10))
        retry, = self.events("media_limit_retry")
        self.assertEqual((retry["media_kind"], retry["sent_images"], retry["api_limit"],
                          retry["dropped"]), ("image", 30, 10, 20))
        self.assertIn("at most 10 images per request", json.dumps(run.messages))

    def test_a_lower_video_limit_is_learned_and_retried(self):
        client = FakeClient([reply(("view", {"paths": self.files(5, ".mp4")})),
                             reply(text="done")], video_limit=3)
        run, _ = self.run_with(client)

        self.assertEqual(client.refused, [(0, 5)])
        self.assertEqual(client.accepted, [(0, 0), (0, 3)])
        self.assertEqual((run.max_videos, run.deliveries.max_videos), (3, 3))

    def test_when_the_api_counts_what_we_do_not_all_media_is_dropped(self):
        paths = self.files(3) + self.files(3, suffix=".mp4", start=3)
        client = FakeClient([reply(("view", {"paths": paths})), reply(text="done")],
                            image_limit=5, videos_count_as_images=True)
        run, (_, termination) = self.run_with(client)

        self.assertEqual(client.refused, [(6, 3)])
        self.assertEqual(client.accepted, [(0, 0), (0, 0)])
        self.assertEqual(media.count_media(run.messages), (0, 0))
        self.assertEqual(termination, "ABANDONED")

    def test_other_bad_requests_are_retried_unchanged_then_raise(self):
        client = FakeClient([])
        client.chat.completions.create = mock.Mock(
            side_effect=ApiError(400, "invalid tool schema"))
        run = loop.Run(client=client, model="fake", system="system",
                       instruction="task", work_dir=self.work, output_path=None,
                       budget=budget.Budget(), trace=trace.Trace(self.trace_path))
        with mock.patch.object(loop.time, "sleep"), self.assertRaises(ApiError):
            run.run()
        self.assertEqual(client.chat.completions.create.call_count, loop.API_RETRIES + 1)
        self.assertEqual(self.events("media_limit_retry"), [])


if __name__ == "__main__":
    unittest.main()
