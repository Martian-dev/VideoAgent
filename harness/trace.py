"""Append-only JSONL record of a run.

The score is one number; the trace is the finding. Everything needed to answer
"how did it work" - where it looked, whether it reviewed its own output, whether
it revised - is reconstructable from this file.
"""
import json
import time
from pathlib import Path


class Trace:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.started = time.time()
        self.counts = {}

    def write(self, kind, **fields):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        record = {"t": round(time.time() - self.started, 2), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def summary(self):
        return {"elapsed_s": round(time.time() - self.started, 1), **self.counts}


MEDIA_TYPES = {"image_url", "video_url", "input_audio"}


def describe(part):
    """A media block as the text that replaces it: what it was, and how big.

    Base64 payloads would make the file enormous - a single 30s clip is several
    megabytes of text - and they are not what anyone reads a trajectory for.
    The placeholder records that a look happened and how big it was.
    """
    kind = part.get("type")
    if kind == "input_audio":
        payload = (part.get("input_audio") or {}).get("data", "")
    else:
        payload = (part.get(kind) or {}).get("url", "")
    return f"[{kind}, {len(payload) * 3 // 4} bytes, omitted from the trajectory]"


def _redact(part):
    """Replace a media block with a description of what was there."""
    if part.get("type") not in MEDIA_TYPES:
        return part
    return {"type": "text", "text": describe(part)}


def save_trajectory(messages, path):
    """Write the full conversation as JSON, media replaced by placeholders.

    The event trace records that a command ran and how many characters came
    back; this records what was actually said and seen. Both are needed - the
    trace to measure a run, the trajectory to understand or publish it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    for message in messages:
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, list):
            copy["content"] = [_redact(part) for part in content]
        cleaned.append(copy)

    path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path
