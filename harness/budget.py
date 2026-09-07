"""Caps that make runs comparable.

Hitting a limit is a result, not an error: a run that spends 40 turns without
submitting tells you something a rubric score cannot.

`max_tokens` counts CUMULATIVE spend, not context size. History is resent every
turn, so cumulative tokens grow roughly quadratically with turn count while the
live context stays far smaller - a 24-turn run spent 802k cumulative while its
largest single prompt was 73k. Size this as a cost guard (802k input is about
8 cents at contributor pricing), not as a context limit. Turns and wall clock
are the caps that should actually bind.
"""
import time


class Budget:
    def __init__(self, max_turns=40, max_tokens=20_000_000, max_seconds=3600):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.turns = 0
        self.tokens = 0
        self.started = time.time()

    def spend(self, turns=0, tokens=0):
        self.turns += turns
        self.tokens += tokens

    @property
    def elapsed(self):
        return time.time() - self.started

    def exhausted(self):
        """Return the name of the limit that has been hit, or None."""
        if self.turns >= self.max_turns:
            return "turns"
        if self.tokens >= self.max_tokens:
            return "tokens"
        if self.elapsed >= self.max_seconds:
            return "wall_clock"
        return None

    def state(self):
        return {"turns": self.turns, "max_turns": self.max_turns,
                "tokens": self.tokens, "elapsed_s": round(self.elapsed, 1)}
