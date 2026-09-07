"""Build the index the planner reads: what is said and what is shown, per segment.

This is the whole point of the pipeline. ffmpeg alone is blind - it can cut
anywhere but knows nothing about what is at 04:32. Once every segment carries a
transcript and a description of the picture, choosing *what* to cut becomes a
text problem the model can actually reason about.
"""
import concurrent.futures
import io
import json
import subprocess
import tempfile
from pathlib import Path

from montage.llm import ask

DESCRIBE_PROMPT = """You are logging footage for an editor.

Describe what a camera sees in these {count} frames, sampled evenly from one
{duration:.1f} second shot. Cover: the subject, the action, the setting, the
shot size (wide/medium/close), and the mood.

Write 1-2 sentences. Describe only what is visible - no speculation about story
or intent. This description is used to search the footage later, so be concrete
and specific.{speech}"""


def sample_frames(video_path, start, end, count=3):
    """Grab `count` JPEG frames spread across a span, as PIL images."""
    from PIL import Image

    span = max(end - start, 0.1)
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(count):
            # Sample inside the span, avoiding the exact boundaries where a
            # frame often lands on the cut itself.
            offset = start + span * (i + 0.5) / count
            out = Path(tmp) / f"f{i}.jpg"
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{offset:.3f}", "-i", str(video_path),
                 "-frames:v", "1", "-vf", "scale=512:-2", "-y", str(out)],
                capture_output=True,
            )
            if result.returncode == 0 and out.exists():
                images.append(Image.open(io.BytesIO(out.read_bytes())))
    return images


def transcript_for(segment, lines):
    """Narration overlapping this segment, from the whole-file transcript."""
    spoken = [
        text for start, end, text in lines
        if end > segment.start and start < segment.end
    ]
    return " ".join(spoken).strip()


def describe_segment(video_path, segment, speech, frames=3):
    images = sample_frames(video_path, segment.start, segment.end, frames)
    if not images:
        return "(no frames could be read from this segment)"

    speech_note = (
        f"\n\nThe narration over this shot is: \"{speech}\". Mention how the "
        f"picture relates to it only if the connection is visible."
        if speech else ""
    )
    prompt = DESCRIBE_PROMPT.format(
        count=len(images), duration=segment.duration, speech=speech_note
    )
    return ask(prompt, images=images)


def build_index(video_path, segments, transcript_lines, frames=3, workers=4,
                progress=None):
    """Describe every segment, in parallel. Returns a list of dicts."""
    entries = [
        {
            "index": s.index,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "duration": round(s.duration, 3),
            "speech": transcript_for(s, transcript_lines),
            "description": None,
        }
        for s in segments
    ]
    by_index = {s.index: s for s in segments}

    def work(entry):
        try:
            entry["description"] = describe_segment(
                video_path, by_index[entry["index"]], entry["speech"], frames
            )
        except Exception as e:
            # One failed segment should not sink an hour of indexing.
            entry["description"] = f"(description unavailable: {e})"
        return entry

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, entries):
            done += 1
            if progress:
                progress(done, len(entries))
    return entries


def save(entries, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
