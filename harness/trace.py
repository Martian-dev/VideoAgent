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
