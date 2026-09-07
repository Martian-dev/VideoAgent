"""Split a montage into segments at its own scene changes.

ffmpeg's scene-detection filter reports every frame whose content differs enough
from the previous one. Cutting there means segments line up with real edits in
the source rather than an arbitrary grid, so a described segment is one shot
rather than a blur of three.
"""
import json
import re
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class Segment:
    index: int
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


def probe_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def scene_changes(video_path, threshold=0.35):
    """Timestamps where the picture changes materially, via ffmpeg scdet."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vf", f"scdet=threshold={threshold * 100}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # scdet logs one "lavfi.scd.time: <seconds>" line per detected change.
    times = [float(t) for t in re.findall(r"scd\.time:\s*([0-9.]+)", result.stderr)]
    return sorted(set(times))


def segment_video(video_path, threshold=0.35, min_duration=1.5, max_duration=20.0):
    """Return Segments covering the whole video.

    Scene cuts closer together than `min_duration` are merged - a flurry of cuts
    during a fast sequence would otherwise produce segments too short to describe
    or to use in an edit. Anything longer than `max_duration` is split so a
    single long take still yields several searchable spans.
    """
    total = probe_duration(video_path)
    cuts = [t for t in scene_changes(video_path, threshold) if 0 < t < total]

    boundaries = [0.0]
    for cut in cuts:
        if cut - boundaries[-1] >= min_duration:
            boundaries.append(cut)
    if total - boundaries[-1] < min_duration and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(total)

    segments, index = [], 0
    for start, end in zip(boundaries, boundaries[1:]):
        span = end - start
        # Split over-long takes into equal parts rather than one huge segment.
        parts = max(1, int(span // max_duration) + (1 if span % max_duration else 0))
        step = span / parts
        for part in range(parts):
            segments.append(Segment(index, start + part * step,
                                    start + (part + 1) * step))
            index += 1
    return segments


def save(segments, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(s) for s in segments], handle, indent=2)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [Segment(**d) for d in json.load(handle)]
