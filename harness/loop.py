"""The agent loop.

Call the model, execute what it asks for, hand back what happened, repeat. The
harness makes no decisions about the task: it does not summarise, plan, retry
failed edits, or steer. Its only judgement calls are mechanical - when a look is
too old to keep in context, and when the budget is spent.

One wrinkle shapes the whole loop: the Meta Model API rejects content blocks
inside a role="tool" message. So a `view` costs two messages - a text stub as
the tool result, then a user message carrying the actual image or video.
"""
import json

from openai import OpenAI

from harness import media, tools


class Run:
    """One episode: a model, a task, a container path, and a budget."""

    def __init__(self, client, model, system, instruction, work_dir, output_path,
                 budget, trace, max_view_bytes=media.MAX_BYTES, keep_last_looks=4):
        self.client = client
        self.model = model
        self.work_dir = work_dir
        self.output_path = output_path
        self.budget = budget
        self.trace = trace
        self.max_view_bytes = max_view_bytes
        self.keep_last_looks = keep_last_looks

        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        self.submitted = None
        self.stop_reason = None

    # -- model ------------------------------------------------------------

    def _call_model(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools.SCHEMA,
            tool_choice="auto",
            temperature=1,
        )
        usage = response.usage
        if usage:
            self.budget.spend(tokens=usage.total_tokens or 0)
        return response.choices[0], usage

    # -- tool dispatch ----------------------------------------------------

    def _do_shell(self, args):
        command = args.get("command", "")
        output, ok = tools.run_shell(command, self.work_dir)
        self.trace.write("shell", command=command, ok=ok, chars=len(output))
        return output, None

    def _do_view(self, args):
        paths = args.get("paths") or args.get("path") or []
        looks, error = tools.run_view(paths, self.work_dir, self.max_view_bytes)
        self.trace.write(
            "view", paths=paths, ok=bool(looks), error=error,
            kinds=[look.kind for look in looks],
            bytes=sum(look.bytes_sent for look in looks),
        )
        if not looks:
            return media.stub([], error=error or "nothing could be viewed"), None
        # The media itself cannot ride in the tool result; it follows as a user turn.
        return media.stub(looks, error=error), media.user_message(looks)

    def _do_submit(self, args):
        path = args.get("path", "")
        ok, detail = tools.check_submission(path, self.work_dir)
        self.trace.write("submit", path=path, ok=ok, detail=detail)
        if ok:
            self.submitted = path
            self.stop_reason = "submitted"
            return f"Accepted: {detail}", None
        return (f"Rejected: {detail}. Fix it and call submit again.", None)

    def _dispatch(self, call):
        handlers = {"shell": self._do_shell, "view": self._do_view,
                    "submit": self._do_submit}
        handler = handlers.get(call.function.name)
        if handler is None:
            return f"Unknown tool: {call.function.name}", None
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            self.trace.write("bad_tool_args", name=call.function.name,
                             raw=call.function.arguments[:400], error=str(e))
            return f"Could not parse arguments as JSON: {e}", None
        return handler(args)

    # -- loop -------------------------------------------------------------

    def step(self):
        """One turn. Returns False when the run should stop."""
        choice, usage = self._call_model()
        message = choice.message
        self.budget.spend(turns=1)

        self.trace.write(
            "assistant", finish_reason=choice.finish_reason,
            content=(message.content or "")[:2000],
            tool_calls=[c.function.name for c in (message.tool_calls or [])],
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

        self.messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            # Nothing asked for and nothing submitted: the model has stopped.
            self.stop_reason = self.stop_reason or "no_tool_calls"
            return False

        follow_ups = []
        for call in message.tool_calls:
            result, follow_up = self._dispatch(call)
            self.messages.append({
                "role": "tool", "tool_call_id": call.id, "content": result,
            })
            if follow_up:
                follow_ups.append(follow_up)
        self.messages.extend(follow_ups)

        if self.submitted:
            return False

        evicted = media.evict(self.messages, keep_last=self.keep_last_looks)
        if evicted:
            self.trace.write("evict", blocks=evicted)

        limit = self.budget.exhausted()
        if limit:
            self.stop_reason = f"budget:{limit}"
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
            self.trace.write("end", stop_reason=self.stop_reason,
                             submitted=self.submitted, **self.budget.state())
        return self.submitted, self.stop_reason
