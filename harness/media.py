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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

# A 30s clip runs ~4.5k prompt tokens, so a look is expensive and the budget is
# spent in history, not just on the turn that took it.
MAX_BYTES = 24 * 1024 * 1024

# The API refuses a request carrying more than 50 images or 5 videos (400).
# History is resent every turn, so those bound everything in context, not just
# the latest look. The image default leaves headroom below its limit; the video
# limit is too small to spare any.
MAX_IMAGES = 45
MAX_VIDEOS = 5

MEDIA_TYPES = {"image_url", "video_url"}
EVICTED = "[earlier look, no longer in context]"


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


class Deliveries:
    """Looks waiting to go out, released one request's worth per turn.

    The API caps images and videos per request, so a view of more than that
    cannot be answered in a single message. Refusing it would make the model
    re-plan around a transport limit; instead the looks are paged, one batch per
    turn, and eviction keeps each request under both caps.
    """

    def __init__(self, max_images=MAX_IMAGES, max_videos=MAX_VIDEOS):
        self.max_images = max_images
        self.max_videos = max_videos
        self.pending = deque()

    def __bool__(self):
        return bool(self.pending)

    def _batches(self, looks):
        """Split looks, in order, into what one request can carry."""
        caps = {"image": self.max_images, "video": self.max_videos}
        counts = dict.fromkeys(caps, 0)
        batches, batch = [], []
        for look in looks:
            if counts[look.kind] >= caps[look.kind]:
                batches.append(batch)
                batch, counts = [], dict.fromkeys(caps, 0)
            batch.append(look)
            counts[look.kind] += 1
        if batch:
            batches.append(batch)
        return batches

    def add(self, looks):
        """Queue looks. Returns a note saying when they arrive, or None when
        they all go out this turn."""
        start = len(self.pending)
        self.pending.extend(looks)
        turns = [n for n, batch in enumerate(self._batches(self.pending))
                 for _ in batch][start:]
        if not turns or turns[-1] == 0:
            return None
        first, last = turns[0], turns[-1]
        head = (f"That is more than one request carries (at most "
                f"{self.max_images} images and {self.max_videos} videos), so "
                f"queued files arrive one batch per turn")
        if first == 0:
            plural = "s" if last != 1 else ""
            return (f"{head}: {turns.count(0)} of these in the message below, "
                    f"the rest over the next {last} turn{plural}.")
        if first == last:
            plural = "s" if first != 1 else ""
            return f"{head}: these arrive {first} turn{plural} from now."
        return f"{head}: these arrive over turns {first}-{last} from now."

    def release(self):
        """This turn's batch: queued looks, in order, within both caps."""
        batches = self._batches(self.pending)
        return [self.pending.popleft() for _ in batches[0]] if batches else []

    def queued(self):
        """What is still waiting, as text, or None when nothing is."""
        parts = []
        for kind in ("image", "video"):
            n = sum(look.kind == kind for look in self.pending)
            if n:
                parts.append(f"{n} more {kind}{'s' if n != 1 else ''}")
        return " and ".join(parts) or None

    def clear(self):
        self.pending.clear()


def stub(looks, error=None, note=None):
    """The text that goes back as the tool result. The API will not carry media
    in a tool message, so this only announces what follows."""
    if not looks:
        return f"view failed: {error}"
    described = ", ".join(f"{look.path} ({look.note})" for look in looks)
    plural = "s" if len(looks) != 1 else ""
    if note:
        text = f"Queued {len(looks)} item{plural}: {described}. {note}"
    else:
        text = f"Attached below: {described}. {len(looks)} item{plural} in the next message."
    if error:
        text += f" Could not view: {error}"
    return text


def user_message(looks, queued=None):
    """The message that actually carries the media into context."""
    header = "Contents of " + ", ".join(look.path for look in looks)
    if queued:
        header += f" ({queued} queued for the following turns)"
    content = [{"type": "text", "text": header + ":"}]
    for look in looks:
        content.extend(look.blocks)
    return {"role": "user", "content": content}


def count_media(messages):
    """(images, videos) across the whole history - what the API's limits count."""
    kinds = [part.get("type") for message in messages
             if isinstance(message.get("content"), list)
             for part in message["content"]]
    return kinds.count("image_url"), kinds.count("video_url")


def evict(messages, keep_last=6, max_images=None, max_videos=None, marker=EVICTED):
    """Drop media blocks from older turns, leaving a text marker in their place.

    Three bounds, newest media kept first: the last `keep_last` looks (messages
    carrying media), and at most `max_images` image blocks and `max_videos`
    video blocks in total - the API counts each per request, and a single look
    can carry dozens.

    History is resent every turn, so without this a dozen looks quietly consume
    the context window. Replacing them with a marker rather than deleting the
    message keeps the transcript honest, and makes the model's response to
    forgetting - whether it takes notes or just looks again - something we
    measure rather than something we solve for it.
    """
    caps = {"image_url": max_images, "video_url": max_videos}
    kept = dict.fromkeys(caps, 0)
    looks = evicted = 0

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if not any(part.get("type") in MEDIA_TYPES for part in content):
            continue

        looks += 1
        replacement = list(content)
        for i in reversed(range(len(content))):
            kind = content[i].get("type")
            if kind not in MEDIA_TYPES:
                continue
            over = caps[kind] is not None and kept[kind] >= caps[kind]
            if looks <= keep_last and not over:
                kept[kind] += 1
                continue
            replacement[i] = {"type": "text", "text": marker}
            evicted += 1
        message["content"] = replacement

    return evicted
