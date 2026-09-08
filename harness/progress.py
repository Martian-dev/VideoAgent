"""Detecting a stalled run, without asking the model whether it is stuck.

Grounding this in the model's own self-assessment does not work. "When Do Agent
Loops Mistake Stagnation for Progress?" (arXiv 2607.25152) measured a frontier
agent claiming improvement on all 54 cycles while 56% showed zero or negative
change; an in-band self-judge rejected 38% of genuine improvements and accepted
44% of real regressions. So every signal here is read off the filesystem and the
transcript, never from anything the model says about itself.

Two independent detectors:

  TIGHT LOOP    the same action producing the same observation, three times.
                Not "the same command three times" - ffmpeg at -ss 10, 20, 30 is
                three different actions, and a repeated command whose output
                changed is still new information. Identical action AND identical
                result means nothing entered the system. Threshold of 3 follows
                WebArena (arXiv 2307.13854, App. A.6).

  NO PROGRESS   nothing observable changed for N consecutive turns. Deliberately
                generous about what counts: extracting frames, viewing a new
                path, or trying a new command are all progress even though none
                of them advances the output file. A real run spent its first 24
                turns purely perceiving, and that is legitimate work.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorldState:
    """What is observably true on disk, independent of anything the model claims."""
    artifact_hash: str = ""
    artifact_duration: float = 0.0
    file_count: int = 0

    @classmethod
    def observe(cls, output_path, work_dir):
        artifact_hash, duration = "", 0.0
        # The output path is often unknown until the model submits: the task
        # names it, not the harness. Until then, work-dir activity carries the
        # progress signal on its own.
        output = Path(output_path) if output_path else None
        if output is not None and output.is_file() and output.stat().st_size:
            digest = hashlib.sha1()
            with output.open("rb") as handle:
                # The head is enough to notice a re-render and avoids hashing
                # hundreds of megabytes every turn.
                digest.update(handle.read(4 * 1024 * 1024))
            artifact_hash = digest.hexdigest()[:16]
            from harness import media
            duration = media.probe_duration(output) or 0.0

        work = Path(work_dir)
        file_count = sum(1 for p in work.rglob("*") if p.is_file()) if work.is_dir() else 0
        return cls(artifact_hash, duration, file_count)


@dataclass
class Progress:
    """Tracks whether each turn moved the world or the model's information."""
    output_path: str = None
    work_dir: str = "."
    grace_turns: int = 8
    repeat_threshold: int = 3
    idle_limit: int = 12

    turn: int = 0
    idle_streak: int = 0
    world: WorldState = field(default_factory=WorldState)
    seen_actions: set = field(default_factory=set)
    seen_views: set = field(default_factory=set)
    recent: list = field(default_factory=list)      # (action_sig, observation_hash)
    loop_start_idx: int = None
    stall_reason: str = ""

    @staticmethod
    def signature(name, arguments):
        return hashlib.sha1(f"{name}:{arguments}".encode()).hexdigest()[:16]

    @staticmethod
    def observation_hash(text):
        return hashlib.sha1((text or "").encode()).hexdigest()[:16]

    def record(self, actions, message_index):
        """Register one turn. `actions` is [(name, arguments, observation)].

        Returns None, "tight_loop" or "no_progress".
        """
        self.turn += 1
        moved = False

        for name, arguments, observation in actions:
            sig = self.signature(name, arguments)
            obs = self.observation_hash(observation)

            if sig not in self.seen_actions:
                moved = True                       # a new thing was tried
                self.seen_actions.add(sig)
            if name == "view":
                if arguments not in self.seen_views:
                    moved = True                   # looked somewhere new
                    self.seen_views.add(arguments)

            self.recent.append((sig, obs))
            if len(self.recent) > 40:
                self.recent.pop(0)

        current = WorldState.observe(self.output_path, self.work_dir)
        if (current.artifact_hash != self.world.artifact_hash
                or current.file_count != self.world.file_count):
            moved = True                           # the filesystem changed
        self.world = current

        self.idle_streak = 0 if moved else self.idle_streak + 1

        # A run needs room to orient before any of this applies.
        if self.turn <= self.grace_turns:
            return None

        if self._tight_loop():
            self.stall_reason = (
                f"the same action returned the same result "
                f"{self.repeat_threshold} times"
            )
            self.loop_start_idx = message_index
            return "tight_loop"

        if self.idle_streak >= self.idle_limit:
            self.stall_reason = (
                f"nothing observable changed for {self.idle_streak} turns: no new "
                f"files, no new looks, no new commands, no change to the output"
            )
            self.loop_start_idx = message_index
            return "no_progress"
        return None

    def _tight_loop(self):
        if len(self.recent) < self.repeat_threshold:
            return False
        tail = self.recent[-self.repeat_threshold:]
        return len(set(tail)) == 1                 # identical action AND result

    def reset_after_rewind(self):
        """Clear the loop evidence so a rewind gets a genuine second chance."""
        self.recent.clear()
        self.idle_streak = 0

    def state(self):
        return {"turn": self.turn, "idle_streak": self.idle_streak,
                "distinct_actions": len(self.seen_actions),
                "artifact_s": round(self.world.artifact_duration, 1),
                "files": self.world.file_count}
