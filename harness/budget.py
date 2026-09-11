"""What bounds a run, and how it ended.

No turn cap. A turn count encodes an assumption about how many steps the work
should take, which is exactly the kind of harness opinion that contaminates a
capability measurement - a source that legitimately needs 120 turns should not
be killed for it. Terminal-Bench bounds runs on wall-clock alone for the same
reason.

What binds instead is real resources (time, money) and demonstrated pathology
(stalling, repeated errors). A run that hits a limit is a result, not an error,
so the outcome is always reported as two separate things: whether it submitted,
and why it stopped. Terminal-Bench keeps `is_resolved` and `FailureMode` apart
for this reason; pooling them makes the numbers lie.
"""
import time

# Contributor pricing, USD per million tokens.
INPUT_COST_PER_M = 0.10
OUTPUT_COST_PER_M = 0.20


class Termination:
    SUBMITTED = "SUBMITTED"
    TIMEOUT = "TIMEOUT"
    COST = "COST"
    TOO_MANY_ERRORS = "TOO_MANY_ERRORS"
    STALLED = "STALLED"
    # Not a limit: the model stopped asking for anything without submitting.
    ABANDONED = "ABANDONED"


class Budget:
    def __init__(self, max_seconds=3600, max_cost_usd=6.0, max_errors=10,
                 safety_turns=500):
        self.max_seconds = max_seconds
        self.max_cost_usd = max_cost_usd
        self.max_errors = max_errors
        # Not a capability limit - a runaway guard, set far beyond any plausible
        # legitimate run so it never binds in practice.
        self.safety_turns = safety_turns

        self.turns = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.consecutive_errors = 0
        self.started = time.time()

    def spend(self, turns=0, input_tokens=0, output_tokens=0):
        self.turns += turns
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_result(self, ok):
        self.consecutive_errors = 0 if ok else self.consecutive_errors + 1

    @property
    def cost_usd(self):
        return (self.input_tokens / 1e6 * INPUT_COST_PER_M
                + self.output_tokens / 1e6 * OUTPUT_COST_PER_M)

    @property
    def elapsed(self):
        return time.time() - self.started

    def remaining(self):
        """Seconds of wall clock left, never below zero."""
        return max(0.0, self.max_seconds - self.elapsed)

    def exhausted(self):
        """The Termination reason that applies, or None."""
        if self.elapsed >= self.max_seconds:
            return Termination.TIMEOUT
        if self.cost_usd >= self.max_cost_usd:
            return Termination.COST
        if self.consecutive_errors >= self.max_errors:
            return Termination.TOO_MANY_ERRORS
        if self.turns >= self.safety_turns:
            return Termination.TIMEOUT
        return None

    def state(self):
        return {"turns": self.turns, "elapsed_s": round(self.elapsed, 1),
                "cost_usd": round(self.cost_usd, 4),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "consecutive_errors": self.consecutive_errors}
