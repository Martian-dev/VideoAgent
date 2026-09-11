"""The three tools, and how they execute.

Descriptions are factual: what the tool does and what it returns. None of them
say anything about video editing, which segments matter, or how to approach a
task - the model brings all of that.
"""
import contextlib
import os
import shutil
import signal
import subprocess
from pathlib import Path

from harness import media

SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a shell command in the container. Returns stdout, stderr and "
                "the exit code. Use this for ffmpeg, ffprobe, transcribe, and any "
                "other file or media operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": (
                "Load an image or video file into your context so you can see it. "
                "Accepts one path or several. Images and video only - audio cannot "
                "be viewed; use `transcribe` through the shell for speech."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to image or video files.",
                    }
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Declare the file at `path` your finished result. The run ends here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the finished file."}
                },
                "required": ["path"],
            },
        },
    },
]

MAX_OUTPUT_CHARS = 12000
# The longest one command may run, before the cap at the run's remaining time.
# Transcription runs at ~1.5x realtime, so a short ceiling would kill work the
# model legitimately asked for on a long source.
COMMAND_TIMEOUT_S = 1800


def run_shell(command, cwd, timeout=COMMAND_TIMEOUT_S):
    """Execute a command and format the result the way the model will read it.

    The command gets its own process group, and a timeout kills all of it.
    Killing only the shell would leave an ffmpeg or a transcribe it started
    running on, competing for the CPU with everything after it.
    """
    process = subprocess.Popen(
        command, shell=True, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        # Something that left the group can still hold the pipes open; the
        # harness does not wait on it.
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=5)
        return f"[timed out after {timeout:.0f}s]", False

    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    if process.returncode != 0:
        parts.append(f"[exit {process.returncode}]")
    text = "\n".join(parts) or "[no output]"

    if len(text) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        omitted = len(text) - MAX_OUTPUT_CHARS
        text = f"{text[:half]}\n\n[... {omitted} characters omitted ...]\n\n{text[-half:]}"
    return text, process.returncode == 0


def run_view(paths, cwd, max_bytes):
    """Load files into Looks. Returns (looks, error_text).

    The API rejects content blocks inside a tool result, so the caller sends a
    text stub as the tool result and the media as a following user message.
    """
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return [], "view needs at least one path"

    looks, errors = [], []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(cwd) / candidate
        try:
            looks.append(media.load(candidate, max_bytes=max_bytes))
        except media.MediaError as e:
            errors.append(str(e))
    return looks, ("; ".join(errors) if errors else None)


def check_submission(path, cwd, output_path=None):
    """Verify a submitted file exists and is playable.

    Returns (ok, detail, delivered_elsewhere).

    On acceptance the file is delivered to `output_path` if it is not already
    there. Whoever reads the result - a grader, a verifier - looks at one agreed
    location, and a good edit left in the scratch dir would score as no edit at
    all. Putting the accepted file where it was asked for is protocol, not an
    editorial decision, so it stays inside the harness.

    But delivering it and saying nothing would hide a real result. When the
    instruction names the output path, putting the file there is part of what
    the task asked for, and a run that had to be rescued did not do that. So the
    relocation is reported rather than absorbed: `delivered_elsewhere` is True
    when the harness had to move the file, and the analysis can tell those runs
    apart from the ones that landed it themselves.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    if not candidate.exists():
        return False, f"{path}: no such file", False
    if candidate.stat().st_size == 0:
        return False, f"{path}: file is empty", False

    duration = media.probe_duration(candidate)
    if duration is None:
        return (False,
                f"{path}: ffprobe cannot read a duration; is it a valid media file?",
                False)

    detail = f"{path}: {duration:.1f}s, {candidate.stat().st_size / 1e6:.1f} MB"
    delivered_elsewhere = False
    if output_path:
        target = Path(output_path)
        if candidate.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, target)
            detail += f"; delivered to {target}"
            delivered_elsewhere = True
    return True, detail, delivered_elsewhere
