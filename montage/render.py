"""Execute an edit decision list with ffmpeg: cut, join, title, subtitle.

Nothing here makes creative decisions. It takes the EDL and produces the file.
"""
import os
import subprocess
import tempfile
from pathlib import Path

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(preferred=None):
    for candidate in ([preferred] if preferred else []) + FONT_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        "No font found for on-screen text. Install one "
        "(apt install fonts-dejavu-core) or pass --font."
    )


def _escape_drawtext(text):
    r"""drawtext treats ':' as a separator, '%' as strftime and '\' as escape."""
    return (text.replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("%", "\\%")
                .replace("'", "’"))


def _escape_filter_path(path):
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def _srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()[-800:]}")


def extract_cut(video_path, start, end, out_path, text=None, font=None,
                width=None):
    """Cut one span, re-encoding so the joins are frame-accurate.

    Stream copy would snap each cut to the nearest keyframe, which drifts the
    audio out of sync with the picture across a dozen joins.
    """
    filters = []
    if width:
        filters.append(f"scale={width}:-2")
    if text:
        filters.append(
            "drawtext=" + ":".join([
                f"fontfile='{font or find_font()}'",
                f"text='{_escape_drawtext(text)}'",
                "fontsize=h/16",
                "fontcolor=white",
                "x=(w-tw)/2",
                "y=h-th-h*0.10",
                "shadowcolor=black@0.7", "shadowx=2", "shadowy=2",
            ])
        )

    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video_path),
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        # A uniform timebase and pixel format keeps concat from rejecting parts.
        "-video_track_timescale", "90000", "-pix_fmt", "yuv420p",
        "-y", str(out_path),
    ]
    _run(cmd)


def concat(parts, out_path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for part in parts:
            handle.write(f"file '{os.path.abspath(part)}'\n")
        list_path = handle.name
    try:
        _run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", list_path, "-c", "copy", "-movflags", "+faststart",
              "-y", str(out_path)])
    finally:
        os.unlink(list_path)


def write_srt(cues, path):
    """cues: [(start, end, text)] on the finished timeline."""
    with open(path, "w", encoding="utf-8") as handle:
        for i, (start, end, text) in enumerate(cues, start=1):
            handle.write(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n\n")
    return path


def burn_subtitles(video_path, srt_path, out_path, font_size=24):
    style = (f"FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
             f"Outline=2,MarginV=40,BorderStyle=1")
    _run([
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"subtitles='{_escape_filter_path(srt_path)}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy",
        "-movflags", "+faststart", "-y", str(out_path),
    ])


def render(video_path, cuts, out_path, work_dir, font=None, width=None,
           progress=None):
    """Render an EDL. Returns (out_path, cue_spans) where cue_spans maps each
    cut onto the finished timeline, for subtitling."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    parts, spans, elapsed = [], [], 0.0
    for n, cut in enumerate(cuts):
        part = work_dir / f"cut_{n:04d}.mp4"
        extract_cut(video_path, cut["start"], cut["end"], part,
                    text=cut.get("text"), font=font, width=width)
        parts.append(part)
        duration = cut["end"] - cut["start"]
        spans.append({**cut, "out_start": elapsed, "out_end": elapsed + duration})
        elapsed += duration
        if progress:
            progress(n + 1, len(cuts))

    concat(parts, out_path)
    return out_path, spans
