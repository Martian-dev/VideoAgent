"""Turn files on disk into content blocks the model can actually perceive.

This is the only part of the harness that knows anything about media, and even
here the knowledge is deliberately shallow: how to encode a file, how big it is,
and when to drop it from history. It knows nothing about scenes, cuts or edits -
that judgement belongs to the model.

The Meta Model API rejects content blocks inside a role="tool" message (400,
"did not match any supported type"), so a look costs two messages: a text stub
as the tool result, then a user message carrying the real blocks.
"""
import base64
import mimetypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

# A 30s clip runs ~4.5k prompt tokens, so a look is expensive and the budget is
# spent in history, not just on the turn that took it.
MAX_BYTES = 24 * 1024 * 1024


class MediaError(Exception):
    """Raised for anything the model could plausibly fix by trying again."""


@dataclass
class Look:
    """One act of perception: what was viewed, and the blocks it produced."""
    path: str
    kind: str                 # image | video
    blocks: list = field(default_factory=list)
    bytes_sent: int = 0
    note: str = ""


def classify(path):
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        # The model is deaf. Probed against muse-spark-1.2-contributor: audio_url
        # is rejected outright (400), while input_audio and file blocks are
        # ACCEPTED AND SILENTLY IGNORED - the model then answers as if it had
        # heard silence. Failing loudly here is the only safe behaviour; a
        # successful-looking view of an audio file would invite hallucination.
        raise MediaError(
            f"{path}: audio cannot be viewed - this model has no hearing. "
            f"Use the shell tool to transcribe it instead: "
            f"transcribe {path}"
        )
    raise MediaError(
        f"{path}: not a viewable file type. Viewable: "
        f"{', '.join(sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES | AUDIO_SUFFIXES))}"
    )


def _data_url(path, mime):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def probe_duration(path):
    """Seconds, or None when ffprobe cannot tell (still images)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out) if out and out != "N/A" else None
    except (ValueError, subprocess.SubprocessError):
        return None


def load(path, max_bytes=MAX_BYTES):
    """Read a file into content blocks. Raises MediaError with advice the model
    can act on - it has ffmpeg, so "too big" is a solvable problem for it."""
    file_path = Path(path)
    if not file_path.exists():
        raise MediaError(f"{path}: no such file")
    if file_path.is_dir():
        raise MediaError(f"{path}: is a directory")

    kind = classify(file_path)
    size = file_path.stat().st_size
    if size == 0:
        raise MediaError(f"{path}: file is empty")
    if size > max_bytes:
        raise MediaError(
            f"{path}: {size / 1e6:.1f} MB exceeds the {max_bytes / 1e6:.0f} MB "
            f"view limit. Use ffmpeg to make it smaller - a shorter span, a lower "
            f"resolution (-vf scale=480:-2), or extract stills instead of a clip."
        )

    mime = mimetypes.guess_type(str(file_path))[0] or {
        "image": "image/jpeg", "video": "video/mp4",
    }[kind]
    url = _data_url(file_path, mime)

    if kind == "image":
        block = {"type": "image_url", "image_url": {"url": url}}
    else:  # video; audio is rejected in classify()
        block = {"type": "video_url", "video_url": {"url": url}}

    duration = probe_duration(file_path)
    note = f"{kind}, {size / 1e6:.1f} MB"
    if duration:
        note += f", {duration:.1f}s"

    return Look(path=str(file_path), kind=kind, blocks=[block],
                bytes_sent=size, note=note)


def stub(looks, error=None):
    """The text that goes back as the tool result. The API will not carry media
    in a tool message, so this only announces what follows."""
    if error:
        return f"view failed: {error}"
    described = ", ".join(f"{look.path} ({look.note})" for look in looks)
    plural = "s" if len(looks) != 1 else ""
    return f"Attached below: {described}. {len(looks)} item{plural} in the next message."


def user_message(looks):
    """The message that actually carries the media into context."""
    content = [{"type": "text",
                "text": "Contents of " + ", ".join(look.path for look in looks) + ":"}]
    for look in looks:
        content.extend(look.blocks)
    return {"role": "user", "content": content}


def evict(messages, keep_last=6):
    """Drop media blocks from older turns, leaving a text marker in their place.

    History is resent every turn, so without this a dozen looks quietly consume
    the context window. Replacing them with a marker rather than deleting the
    message keeps the transcript honest, and makes the model's response to
    forgetting - whether it takes notes or just looks again - something we
    measure rather than something we solve for it.
    """
    media_types = {"image_url", "video_url"}
    seen = 0
    evicted = 0

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if not any(part.get("type") in media_types for part in content):
            continue

        seen += 1
        if seen <= keep_last:
            continue

        replacement = []
        for part in content:
            if part.get("type") in media_types:
                replacement.append({"type": "text",
                                    "text": "[earlier look, no longer in context]"})
                evicted += 1
            else:
                replacement.append(part)
        message["content"] = replacement

    return evicted
