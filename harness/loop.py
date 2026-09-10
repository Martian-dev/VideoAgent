"""The agent loop.

Call the model, execute what it asks for, hand back what happened, repeat. The
harness makes no decisions about the task: it does not summarise, plan, retry
failed edits, or steer. Its only judgement calls are mechanical - when a look is
too old to keep in context, how many images fit in one request, and when the
budget is spent.

One wrinkle shapes the whole loop: the Meta Model API rejects content blocks
inside a role="tool" message. So a `view` costs two messages - a text stub as
the tool result, then a user message carrying the actual image or video.
"""
import json
import re

from openai import OpenAI

from harness import media, progress as progress_mod, tools
from harness import trace as trace_mod
from harness.budget import Termination

_LIMIT = re.compile(r"maximum of\s+(\d+)", re.IGNORECASE)
_CONTAINED = re.compile(r"(\d+)\s+images?\b", re.IGNORECASE)


def image_limit(error):
    """(images the API counted or None, its limit) when `error` is the
    per-request image cap - "request contains 60 images, exceeding the maximum
    of 50 allowed per request" - else None."""
    if getattr(error, "status_code", None) != 400:
        return None
    text = str(error)
    limit = _LIMIT.search(text)
    if not limit or "image" not in text.lower():
        return None
    contained = _CONTAINED.search(text)
    return (int(contained.group(1)) if contained else None), int(limit.group(1))


class Run:
    """One episode: a model, a task, a container path, and a budget."""

    def __init__(self, client, model, system, instruction, work_dir, output_path,
                 budget, trace, max_view_bytes=media.MAX_BYTES, keep_last_looks=4,
                 max_images=media.MAX_IMAGES, allow_rewind=True,
                 trajectory_path=None):
        self.client = client
        self.model = model
        self.work_dir = work_dir
        self.output_path = output_path
        self.budget = budget
        self.trace = trace
        self.max_view_bytes = max_view_bytes
        self.keep_last_looks = keep_last_looks
        self.max_images = max_images
        self.deliveries = media.Deliveries(max_images)

        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        self.submitted = None
        self.delivered_elsewhere = False
        self.stop_reason = None
        self.allow_rewind = allow_rewind
        self.trajectory_path = trajectory_path
        self.rewinds = 0
        self.progress = progress_mod.Progress(output_path=output_path,
                                              work_dir=str(work_dir))

    # -- model ------------------------------------------------------------

    def _call_model(self):
        """One request, kept under the per-request image limit.

        The bound is enforced here, where the request is built, so no path into
        the history - several looks in one turn, a rewind - can exceed it. The
        number is the API's, though, not ours: if a request is still refused
        for images, lower the bound to what the API reported and retry, then
        retry once more with every image dropped. The model sees a marker where
        each one was. A transport limit is never a reason to end the run.
        """
        evicted = media.evict(self.messages, keep_last=self.keep_last_looks,
                              max_images=self.max_images)
        if evicted:
            self.trace.write("evict", blocks=evicted)

        for attempt in range(3):
            images = media.count_images(self.messages)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=tools.SCHEMA,
                    tool_choice="auto",
                    temperature=1,
                )
                break
            except Exception as error:
                refused = image_limit(error)
                if refused is None or attempt == 2:
                    raise
                counted, limit = refused
                self.max_images = self.deliveries.max_images = min(self.max_images, limit)
                marker = f"[dropped: the API accepts at most {limit} images per request]"
                dropped = 0
                if attempt == 0:
                    dropped = media.evict(self.messages, keep_last=self.keep_last_looks,
                                          max_images=self.max_images, marker=marker)
                # Nothing to trim means the API counts something we don't.
                if not dropped:
                    dropped = media.evict(self.messages, keep_last=0, marker=marker)
                self.trace.write("image_limit_retry", attempt=attempt + 1,
                                 sent=images, api_counted=counted, api_limit=limit,
                                 max_images=self.max_images, dropped=dropped)

        usage = response.usage
        if usage:
            self.budget.spend(input_tokens=usage.prompt_tokens or 0,
                              output_tokens=usage.completion_tokens or 0)
        return response.choices[0], usage, images

    # -- tool dispatch ----------------------------------------------------

    def _do_shell(self, args):
        command = args.get("command", "")
        output, ok = tools.run_shell(command, self.work_dir)
        self.trace.write("shell", command=command, ok=ok, chars=len(output))
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
        remaining = self.deliveries.images_pending()
        self.messages.append(media.user_message(looks, remaining=remaining))
        paths = [look.path for look in looks]
        self.trace.write("deliver", paths=paths, remaining=remaining,
                         images=sum(look.kind == "image" for look in looks))
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
        choice, usage, images = self._call_model()
        message = choice.message
        self.budget.spend(turns=1)

        self.trace.write(
            "assistant", finish_reason=choice.finish_reason,
            content=(message.content or "")[:2000],
            tool_calls=[c.function.name for c in (message.tool_calls or [])],
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            images_in_context=images,
        )

        self.messages.append(message.model_dump(exclude_none=True))

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
            self.messages.append({
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
            if self.trajectory_path:
                # Written in `finally` so a crashed or killed run still leaves a
                # readable transcript behind.
                try:
                    trace_mod.save_trajectory(self.messages, self.trajectory_path)
                except Exception as e:
                    self.trace.write("trajectory_failed", error=str(e)[:200])
            self.trace.write("end", termination=self.stop_reason,
                             submitted=bool(self.submitted), path=self.submitted,
                             delivered_elsewhere=self.delivered_elsewhere,
                             rewinds=self.rewinds,
                             **self.budget.state(), **self.progress.state())
        return self.submitted, self.stop_reason
