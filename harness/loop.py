"""The agent loop.

Call the model, execute what it asks for, hand back what happened, repeat. The
harness makes no decisions about the task: it does not summarise, plan, retry
failed edits, or steer. Its only judgement calls are mechanical - when a look is
too old to keep in context, how much media fits in one request, how long to wait
out a failing API, and when the budget is spent.

One wrinkle shapes the whole loop: the Meta Model API rejects content blocks
inside a role="tool" message. So a `view` costs two messages - a text stub as
the tool result, then a user message carrying the actual image or video.
"""
import json
import re
import time

from openai import OpenAI

from harness import atif, media, progress as progress_mod, tools
from harness import trace as trace_mod
from harness.budget import Termination

# A failed request is retried this many times, waiting RETRY_BACKOFF_S and
# doubling each time: 15, 30, 60, 120, 240 seconds, long enough to sit out an
# outage of several minutes. Only when every retry has failed does the error end
# the run.
API_RETRIES = 5
RETRY_BACKOFF_S = 15.0
# The longest one model request may take, before the cap at the time left.
# Observed calls take ~10s typically and 72s at worst, so this only ever
# catches a request that has hung.
REQUEST_TIMEOUT_S = 600.0

_LIMIT = re.compile(r"maximum of\s+(\d+)", re.IGNORECASE)
_CONTAINED = re.compile(r"(\d+)\s+(image|video)s?\b", re.IGNORECASE)


def media_limit(error):
    """(kind, count the API reported or None, its limit) when `error` is a
    per-request media cap - "request contains 60 images, exceeding the maximum
    of 50 allowed per request" - else None. kind is "image" or "video"."""
    if getattr(error, "status_code", None) != 400:
        return None
    text = str(error)
    limit = _LIMIT.search(text)
    if not limit:
        return None
    contained = _CONTAINED.search(text)
    if contained:
        return contained.group(2).lower(), int(contained.group(1)), int(limit.group(1))
    for kind in ("video", "image"):
        if kind in text.lower():
            return kind, None, int(limit.group(1))
    return None


class Run:
    """One episode: a model, a task, a container path, and a budget."""

    def __init__(self, client, model, system, instruction, work_dir, output_path,
                 budget, trace, max_view_bytes=media.MAX_BYTES, keep_last_looks=4,
                 max_images=media.MAX_IMAGES, max_videos=media.MAX_VIDEOS,
                 allow_rewind=True, trajectory_path=None, messages_path=None,
                 agent_version="unknown", api_retries=API_RETRIES,
                 retry_backoff=RETRY_BACKOFF_S, request_timeout=REQUEST_TIMEOUT_S,
                 command_timeout=tools.COMMAND_TIMEOUT_S):
        self.client = client
        self.model = model
        self.work_dir = work_dir
        self.output_path = output_path
        self.budget = budget
        self.trace = trace
        self.max_view_bytes = max_view_bytes
        self.keep_last_looks = keep_last_looks
        self.max_images = max_images
        self.max_videos = max_videos
        self.deliveries = media.Deliveries(max_images, max_videos)
        self.api_retries = api_retries
        self.retry_backoff = retry_backoff
        self.api_retries_used = 0
        self.request_timeout = request_timeout
        self.command_timeout = command_timeout

        self.messages = []
        # What ATIF needs and a chat message does not carry: when it was said,
        # and what the turn cost. Kept parallel to `messages` - appended with
        # them, truncated with them - so a rewind cannot put them out of step.
        self.records = []
        self.session_id = atif.session_id()
        self.agent_version = agent_version
        self._remember({"role": "system", "content": system})
        self._remember({"role": "user", "content": instruction})
        self.submitted = None
        self.delivered_elsewhere = False
        self.stop_reason = None
        self.allow_rewind = allow_rewind
        self.trajectory_path = trajectory_path
        self.messages_path = messages_path
        self.rewinds = 0
        self.progress = progress_mod.Progress(output_path=output_path,
                                              work_dir=str(work_dir))

    # -- the conversation --------------------------------------------------

    def _remember(self, message, metrics=None):
        """Append a message, and the facts about it ATIF needs."""
        self.messages.append(message)
        self.records.append({"time": atif.timestamp(), "metrics": metrics})

    def trajectory(self):
        """The run so far as an ATIF trajectory."""
        return atif.build(
            self.messages, self.records, model=self.model,
            version=self.agent_version, session=self.session_id,
            totals={"total_prompt_tokens": self.budget.input_tokens,
                    "total_completion_tokens": self.budget.output_tokens,
                    "total_cost_usd": round(self.budget.cost_usd, 4)},
            notes=(f"termination={self.stop_reason} "
                   f"submitted={self.submitted or False}"),
        )

    # -- model ------------------------------------------------------------

    def _call_model(self):
        """One request, kept under the per-request media limits and retried
        through API failures.

        The bounds are enforced here, where the request is built, so no path
        into the history - several looks in one turn, a rewind - can exceed
        them. The numbers are the API's, though, not ours: if a request is still
        refused for images or videos, lower that bound to what the API reported
        and retry, then retry once more with all media dropped. The model sees a
        marker where each one was. A transport limit is never a reason to end
        the run.

        Any other failure - a 500, a 400, a dropped connection - is retried
        unchanged, up to `api_retries` times with exponential backoff. A retry
        that succeeds carries on as if nothing happened; only when every one has
        failed does the error end the run. The SDK's own retries are off (see
        run.py), so this count is the only one.

        None of it may outlast the run's wall clock, because whatever runs the
        harness kills it at its own limit and the trajectory goes with it. Each
        request's timeout is capped at the time left, and a retry the budget
        cannot wait for is not attempted. Returns None when time runs out.
        """
        evicted = self._evict()
        if evicted:
            self.trace.write("evict", blocks=evicted)

        media_retries = retries = 0
        while True:
            remaining = self.budget.remaining()
            if remaining <= 0:
                return None
            images, videos = media.count_media(self.messages)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=tools.SCHEMA,
                    tool_choice="auto",
                    temperature=1,
                    timeout=min(self.request_timeout, remaining),
                )
                break
            except Exception as error:
                refused = media_limit(error)
                if refused is not None and media_retries < 2:
                    kind, counted, limit = refused
                    self._lower_limit(kind, limit)
                    marker = f"[dropped: the API accepts at most {limit} {kind}s per request]"
                    dropped = self._evict(marker) if media_retries == 0 else 0
                    # Nothing to trim means the API counts something we don't.
                    if not dropped:
                        dropped = media.evict(self.messages, keep_last=0, marker=marker)
                    media_retries += 1
                    self.trace.write("media_limit_retry", attempt=media_retries,
                                     media_kind=kind, sent_images=images,
                                     sent_videos=videos, api_counted=counted,
                                     api_limit=limit, max_images=self.max_images,
                                     max_videos=self.max_videos, dropped=dropped)
                    continue
                if retries == self.api_retries:
                    raise
                retries += 1
                delay = self.retry_backoff * 2 ** (retries - 1)
                failure = dict(attempt=retries, of=self.api_retries, delay_s=delay,
                               status=getattr(error, "status_code", None),
                               error=type(error).__name__, message=str(error)[:400])
                if delay >= self.budget.remaining():
                    self.trace.write("api_retry_out_of_time", **failure)
                    return None
                self.api_retries_used += 1
                self.trace.write("api_retry", **failure)
                time.sleep(delay)

        usage = response.usage
        if usage:
            self.budget.spend(input_tokens=usage.prompt_tokens or 0,
                              output_tokens=usage.completion_tokens or 0)
        return response.choices[0], usage, (images, videos)

    def _evict(self, marker=media.EVICTED):
        return media.evict(self.messages, keep_last=self.keep_last_looks,
                           max_images=self.max_images, max_videos=self.max_videos,
                           marker=marker)

    def _lower_limit(self, kind, limit):
        limit = max(1, limit)
        if kind == "video":
            self.max_videos = self.deliveries.max_videos = min(self.max_videos, limit)
        else:
            self.max_images = self.deliveries.max_images = min(self.max_images, limit)

    # -- tool dispatch ----------------------------------------------------

    def _do_shell(self, args):
        command = args.get("command", "")
        # Capped at the time left, so no command can carry the run past its
        # wall clock.
        timeout = min(self.command_timeout, self.budget.remaining())
        if timeout < 1:
            output, ok = "[not run: the run's time is up]", False
        else:
            output, ok = tools.run_shell(command, self.work_dir, timeout=timeout)
        self.trace.write("shell", command=command, ok=ok, chars=len(output),
                         timeout_s=round(timeout, 1))
        self.budget.record_result(ok)
        return output

    def _do_view(self, args):
        paths = args.get("paths") or args.get("path") or []
        looks, error = tools.run_view(paths, self.work_dir, self.max_view_bytes)
        self.trace.write(
            "view", paths=paths, ok=bool(looks), error=error,
            kinds=[look.kind for look in looks],
            bytes=sum(look.bytes_sent for look in looks),
        )
        self.budget.record_result(bool(looks))
        if not looks:
            return media.stub([], error=error or "nothing could be viewed")
        # The media itself cannot ride in the tool result; it is queued and
        # follows as a user turn - in batches, when there is more than one
        # request can carry.
        return media.stub(looks, error=error, note=self.deliveries.add(looks))

    def _deliver(self):
        """Send this turn's batch of queued looks. Returns it as an action for
        the progress tracker - new material in context is new information."""
        looks = self.deliveries.release()
        if not looks:
            return None
        queued = self.deliveries.queued()
        self._remember(media.user_message(looks, queued=queued))
        paths = [look.path for look in looks]
        self.trace.write("deliver", paths=paths, queued=queued,
                         images=sum(look.kind == "image" for look in looks),
                         videos=sum(look.kind == "video" for look in looks))
        return ("deliver", json.dumps(paths), "")

    def _do_submit(self, args):
        path = args.get("path", "")
        ok, detail, elsewhere = tools.check_submission(
            path, self.work_dir, self.output_path)
        self.trace.write("submit", path=path, ok=ok, detail=detail,
                         delivered_elsewhere=elsewhere)
        if ok:
            self.submitted = path
            self.delivered_elsewhere = elsewhere
            self.stop_reason = "submitted"
            return f"Accepted: {detail}"
        return f"Rejected: {detail}. Fix it and call submit again."

    def _dispatch(self, call):
        handlers = {"shell": self._do_shell, "view": self._do_view,
                    "submit": self._do_submit}
        handler = handlers.get(call.function.name)
        if handler is None:
            return f"Unknown tool: {call.function.name}"
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            self.trace.write("bad_tool_args", name=call.function.name,
                             raw=call.function.arguments[:400], error=str(e))
            return f"Could not parse arguments as JSON: {e}"
        return handler(args)

    # -- loop -------------------------------------------------------------

    def step(self):
        """One turn. Returns False when the run should stop."""
        called = self._call_model()
        if called is None:
            # The wall clock ran out before the model could answer.
            self.stop_reason = Termination.TIMEOUT
            return False
        choice, usage, (images, videos) = called
        message = choice.message
        self.budget.spend(turns=1)

        self.trace.write(
            "assistant", finish_reason=choice.finish_reason,
            content=(message.content or "")[:2000],
            tool_calls=[c.function.name for c in (message.tool_calls or [])],
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            images_in_context=images, videos_in_context=videos,
        )

        self._remember(message.model_dump(exclude_none=True),
                       metrics={"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens",
                                                             None)})

        if not message.tool_calls and not self.deliveries:
            # Nothing asked for and nothing submitted: the model has stopped.
            # With looks still queued it has not - it was answering a batch,
            # and the next batch is what comes back.
            self.stop_reason = self.stop_reason or Termination.ABANDONED
            return False

        loop_start = len(self.messages) - 1
        actions = []
        for call in message.tool_calls or []:
            result = self._dispatch(call)
            self._remember({
                "role": "tool", "tool_call_id": call.id, "content": result,
            })
            actions.append((call.function.name, call.function.arguments or "", result))

        if self.submitted:
            self.stop_reason = Termination.SUBMITTED
            return False

        delivered = self._deliver()
        if delivered:
            actions.append(delivered)

        stall = self.progress.record(actions, loop_start)
        if stall:
            if self.allow_rewind and self.rewinds == 0:
                # Truncate back to where the loop began rather than ending the
                # run. This adds no information - it only removes the context
                # the model got stuck in - so it stays clean for a capability
                # measurement. One per episode.
                self.rewinds += 1
                dropped = len(self.messages) - self.progress.loop_start_idx
                del self.messages[self.progress.loop_start_idx:]
                del self.records[self.progress.loop_start_idx:]
                # Looks queued by the dropped turns go with them.
                self.deliveries.clear()
                self.progress.reset_after_rewind()
                self.trace.write("rewind", stall=stall, dropped_messages=dropped,
                                 reason=self.progress.stall_reason)
            else:
                self.stop_reason = Termination.STALLED
                self.trace.write("stalled", stall=stall,
                                 reason=self.progress.stall_reason)
                return False

        limit = self.budget.exhausted()
        if limit:
            self.stop_reason = limit
            return False
        return True

    def run(self):
        self.trace.write("start", model=self.model, **self.budget.state())
        try:
            while self.step():
                pass
        except Exception as e:
            self.stop_reason = f"error:{type(e).__name__}"
            self.trace.write("error", type=type(e).__name__, message=str(e)[:800])
            raise
        finally:
            # Written in `finally` so a crashed or killed run still leaves a
            # readable transcript behind.
            if self.trajectory_path:
                try:
                    atif.save(self.trajectory_path, self.trajectory())
                except Exception as e:
                    self.trace.write("trajectory_failed", error=str(e)[:200])
            if self.messages_path:
                try:
                    trace_mod.save_trajectory(self.messages, self.messages_path)
                except Exception as e:
                    self.trace.write("messages_failed", error=str(e)[:200])
            self.trace.write("end", termination=self.stop_reason,
                             submitted=bool(self.submitted), path=self.submitted,
                             delivered_elsewhere=self.delivered_elsewhere,
                             rewinds=self.rewinds, api_retries=self.api_retries_used,
                             **self.budget.state(), **self.progress.state())
        return self.submitted, self.stop_reason
