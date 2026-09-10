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

# The API refuses a request carrying more than 50 images (400). History is resent
# every turn, so that bounds everything in context, not just the latest look.
# The default leaves headroom below it.
MAX_IMAGES = 45

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
    """Looks waiting to go out, released at most `max_images` images per turn.

    One request carries a bounded number of images, so a view of more than that
    cannot be answered in a single message. Refusing it would make the model
    re-plan around a transport limit; instead the looks are paged, one batch per
    turn, and eviction keeps each request under the limit. Videos ride along
    uncounted - the API's limit is on images.
    """

    def __init__(self, max_images=MAX_IMAGES):
        self.max_images = max_images
        self.pending = deque()

    def __bool__(self):
        return bool(self.pending)

    def images_pending(self):
        return sum(1 for look in self.pending if look.kind == "image")

    def add(self, looks):
        """Queue looks. Returns a note saying when they arrive, or None when
        they all go out this turn."""
        ahead = self.images_pending()
        images = sum(1 for look in looks if look.kind == "image")
        self.pending.extend(looks)
        if not images:
            return None
        first = ahead // self.max_images
        last = (ahead + images - 1) // self.max_images
        if last == 0:
            return None
        head = (f"More than {self.max_images} images are queued and a request "
                f"carries at most {self.max_images}, so they arrive "
                f"{self.max_images} per turn")
        if first == 0:
            now = min(images, self.max_images - ahead)
            plural = "s" if last != 1 else ""
            return (f"{head}: {now} of these in the message below, the rest over "
                    f"the next {last} turn{plural}.")
        if first == last:
            plural = "s" if first != 1 else ""
            return f"{head}: these arrive {first} turn{plural} from now."
        return f"{head}: these arrive over turns {first}-{last} from now."

    def release(self):
        """This turn's batch: queued looks, in order, up to the image limit."""
        batch, images = [], 0
        while self.pending:
            if self.pending[0].kind == "image":
                if images >= self.max_images:
                    break
                images += 1
            batch.append(self.pending.popleft())
        return batch

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


def user_message(looks, remaining=0):
    """The message that actually carries the media into context."""
    header = "Contents of " + ", ".join(look.path for look in looks)
    if remaining:
        plural = "s" if remaining != 1 else ""
        header += f" ({remaining} more image{plural} queued for the following turns)"
    content = [{"type": "text", "text": header + ":"}]
    for look in looks:
        content.extend(look.blocks)
    return {"role": "user", "content": content}


def count_images(messages):
    """Image blocks across the whole history - what the API's limit counts."""
    return sum(1 for message in messages
               if isinstance(message.get("content"), list)
               for part in message["content"] if part.get("type") == "image_url")


def evict(messages, keep_last=6, max_images=None, marker=EVICTED):
    """Drop media blocks from older turns, leaving a text marker in their place.

    Two bounds, newest media kept first: the last `keep_last` looks (messages
    carrying media), and at most `max_images` image blocks in total - the API
    counts images per request, and a single look can carry dozens of them.

    History is resent every turn, so without this a dozen looks quietly consume
    the context window. Replacing them with a marker rather than deleting the
    message keeps the transcript honest, and makes the model's response to
    forgetting - whether it takes notes or just looks again - something we
    measure rather than something we solve for it.
    """
    looks = images = evicted = 0

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
            over = (kind == "image_url" and max_images is not None
                    and images >= max_images)
            if looks <= keep_last and not over:
                images += kind == "image_url"
                continue
            replacement[i] = {"type": "text", "text": marker}
            evicted += 1
        message["content"] = replacement

    return evicted
