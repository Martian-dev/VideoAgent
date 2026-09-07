"""The three tools, and how they execute.

Descriptions are factual: what the tool does and what it returns. None of them
say anything about video editing, which segments matter, or how to approach a
task - the model brings all of that.
"""
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


def run_shell(command, cwd, timeout=600):
    """Execute a command and format the result the way the model will read it."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(cwd), capture_output=True,
            text=True, timeout=timeout, errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]", False

    parts = []
    if result.stdout.strip():
        parts.append(result.stdout.rstrip())
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    if result.returncode != 0:
        parts.append(f"[exit {result.returncode}]")
    text = "\n".join(parts) or "[no output]"

    if len(text) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        omitted = len(text) - MAX_OUTPUT_CHARS
        text = f"{text[:half]}\n\n[... {omitted} characters omitted ...]\n\n{text[-half:]}"
    return text, result.returncode == 0


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


def check_submission(path, cwd):
    """Verify a submitted file exists and is playable. Returns (ok, detail)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    if not candidate.exists():
        return False, f"{path}: no such file"
    if candidate.stat().st_size == 0:
        return False, f"{path}: file is empty"

    duration = media.probe_duration(candidate)
    if duration is None:
        return False, f"{path}: ffprobe cannot read a duration; is it a valid media file?"
    return True, f"{path}: {duration:.1f}s, {candidate.stat().st_size / 1e6:.1f} MB"
