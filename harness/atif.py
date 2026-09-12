"""The run as an ATIF trajectory.

ATIF - the Agent Trajectory Interchange Format - is what a viewer, a trajectory
judge or an OTel exporter reads, and what seeds one agent's session from
another's run. It is a published format, not any one runner's API, so this
module writes the JSON by hand: the harness stays an instrument that knows
nothing about whatever is measuring with it.

The conversation the harness keeps is a list of chat messages. ATIF wants one
object per step, with the results of a step's tool calls folded into the step
that made them, so the messages collapse: an assistant message and the tool
results answering it are a single step. What a chat message does not carry -
when it happened, and what the turn cost - travels beside it in `records`.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness import trace as trace_mod

SCHEMA_VERSION = "ATIF-v1.7"
AGENT_NAME = "videoagent-harness"
# Every role the harness uses, mapped to the three ATIF sources. Tool results
# are not a source: they belong to the step whose call they answer.
SOURCES = {"system": "system", "user": "user", "assistant": "agent"}


def session_id():
    return str(uuid.uuid4())


def timestamp(when=None):
    """An ATIF timestamp: ISO 8601, UTC, milliseconds, Z rather than +00:00."""
    moment = (datetime.fromtimestamp(when, timezone.utc) if when
              else datetime.now(timezone.utc))
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_of(content):
    """A message's content as text, media replaced by a description of it.

    Base64 payloads would make the file enormous - a single 30s clip is several
    megabytes of text - and they are not what anyone reads a trajectory for.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in trace_mod.MEDIA_TYPES:
            parts.append(trace_mod.describe(part))
        elif part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(part for part in parts if part)


def _arguments(call):
    """A tool call's arguments as an object.

    They arrive as a JSON string, and a model that emits a malformed one has
    already been answered with a parse error - the trajectory still has to say
    what it tried to send, so the raw text is kept instead.
    """
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"raw_arguments": raw}


def _tool_calls(message):
    return [
        {
            "tool_call_id": call.get("id"),
            "function_name": (call.get("function") or {}).get("name"),
            "arguments": _arguments(call),
        }
        for call in message.get("tool_calls") or []
    ]


def steps(messages, records, model=None):
    """Fold messages and their records into ATIF steps."""
    built, by_call = [], {}
    for message, record in zip(messages, records):
        record = record or {}
        role = message.get("role")
        if role == "tool":
            # The result of a call. It belongs to the step that made it; one
            # whose call was rewound away belongs nowhere and is dropped.
            step = by_call.get(message.get("tool_call_id"))
            if step is None:
                continue
            step.setdefault("observation", {"results": []})["results"].append({
                "source_call_id": message.get("tool_call_id"),
                "content": text_of(message.get("content")),
            })
            continue

        step = {
            "step_id": len(built) + 1,
            "timestamp": record.get("time"),
            "source": SOURCES.get(role, "agent"),
            "message": text_of(message.get("content")),
        }
        if role == "assistant":
            step["model_name"] = model
        calls = _tool_calls(message)
        if calls:
            step["tool_calls"] = calls
        if record.get("metrics"):
            step["metrics"] = record["metrics"]
        built.append(step)
        for call in calls:
            by_call[call["tool_call_id"]] = step
    return built


def build(messages, records, *, model=None, version="unknown", session=None,
          totals=None, notes=None):
    """The whole run as an ATIF trajectory dict."""
    built = steps(messages, records, model=model)
    final_metrics = dict(totals or {})
    final_metrics.setdefault("total_steps", len(built))
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session or session_id(),
        "agent": {"name": AGENT_NAME, "version": version or "unknown",
                  "model_name": model},
        "steps": built,
        "notes": notes,
        "final_metrics": final_metrics,
    }


def save(path, trajectory):
    """Write a trajectory where anything that reads ATIF will look for it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path
