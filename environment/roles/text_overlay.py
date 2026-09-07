import json
import os
import subprocess
from typing import List

from pydantic import BaseModel, Field

from environment.agents.base import BaseTool
from environment.config.runtime import get_font

# Where a caption sits in the frame, as ffmpeg x/y expressions. 'tw'/'th' are the
# rendered text box dimensions, 'w'/'h' the video's.
POSITIONS = {
    "top": ("(w-tw)/2", "h*0.08"),
    "center": ("(w-tw)/2", "(h-th)/2"),
    "bottom": ("(w-tw)/2", "h-th-h*0.08"),
    "top-left": ("w*0.05", "h*0.08"),
    "top-right": ("w-tw-w*0.05", "h*0.08"),
    "bottom-left": ("w*0.05", "h-th-h*0.08"),
    "bottom-right": ("w-tw-w*0.05", "h-th-h*0.08"),
}


def _escape_drawtext(text: str) -> str:
    r"""Escape text for ffmpeg's drawtext filter.

    drawtext treats ':' as an option separator, '%' as a strftime expansion and
    '\' as an escape, so all three must be neutralised before the value is
    single-quoted into the filtergraph.
    """
    return (
        text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("%", "\\%")
            .replace("'", "’")  # a literal ' cannot survive the quoting
    )


class TextOverlay(BaseTool):
    """
    Draw text onto the video picture: titles, lower thirds, chapter cards, credits
    or a watermark. Each overlay may appear for the whole video or only between a
    start and end time. For spoken-word captions use SubtitleGenerator and
    SubtitleBurner instead - this role is for authored text, not transcription.
    """

    class InputSchema(BaseTool.BaseInputSchema):
        video_path: str = Field(
            ...,
            description="File path to the source video"
        )
        overlays: str = Field(
            ...,
            description=(
                "JSON list of overlays. Each item takes 'text' plus optional "
                "'start' and 'end' in seconds, 'position' (top, center, bottom, "
                "top-left, top-right, bottom-left, bottom-right), 'font_size', "
                "'font_color', 'box' (true for a background plate) and 'box_color'. "
                "Example: [{\"text\": \"CHAPTER ONE\", \"start\": 0, \"end\": 3, "
                "\"position\": \"center\", \"font_size\": 72}]"
            )
        )
        output_path: str = Field(
            None,
            description="Output video path. Defaults to the input path with a '_titled' suffix."
        )
        font_file: str = Field(
            None,
            description="Path to a .ttf font. Defaults to a system sans-serif."
        )

    class OutputSchema(BaseModel):
        video_path: str = Field(
            ...,
            description="File path to the video with text drawn on it"
        )
        overlay_count: int = Field(
            ...,
            description="Number of text overlays drawn"
        )

    def _parse_overlays(self, raw) -> List[dict]:
        if isinstance(raw, list):
            return raw
        return json.loads(raw)

    def _build_drawtext(self, overlay: dict, font_file: str) -> str:
        text = str(overlay.get("text", "")).strip()
        if not text:
            return ""

        position = overlay.get("position", "bottom")
        x, y = POSITIONS.get(position, POSITIONS["bottom"])

        options = [
            f"fontfile='{font_file}'",
            f"text='{_escape_drawtext(text)}'",
            f"fontsize={int(overlay.get('font_size', 48))}",
            f"fontcolor={overlay.get('font_color', 'white')}",
            f"x={x}",
            f"y={y}",
            # A shadow keeps text readable over busy footage without a plate.
            "shadowcolor=black@0.6",
            "shadowx=2",
            "shadowy=2",
        ]

        if overlay.get("box"):
            options.append("box=1")
            options.append(f"boxcolor={overlay.get('box_color', 'black@0.5')}")
            options.append("boxborderw=12")

        start, end = overlay.get("start"), overlay.get("end")
        if start is not None or end is not None:
            start = float(start) if start is not None else 0.0
            if end is not None:
                condition = f"between(t,{start},{float(end)})"
            else:
                condition = f"gte(t,{start})"
            options.append(f"enable='{condition}'")

        return "drawtext=" + ":".join(options)

    def execute(self, **kwargs):
        params = self.InputSchema(**kwargs)
        print("Parameters validated successfully")

        video_path = params.video_path
        if not os.path.exists(video_path):
            print(f"Input video not found: {video_path}")
            return

        try:
            overlays = self._parse_overlays(params.overlays)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Could not parse 'overlays' as JSON: {e}")
            return
        if not overlays:
            print("No overlays supplied; nothing to draw.")
            return

        try:
            font_file = get_font(params.font_file)
        except FileNotFoundError as e:
            print(str(e))
            return

        filters = [f for f in (self._build_drawtext(o, font_file) for o in overlays) if f]
        if not filters:
            print("Every overlay had empty text; nothing to draw.")
            return

        output_path = params.output_path
        if not output_path:
            stem, ext = os.path.splitext(video_path)
            output_path = f"{stem}_titled{ext or '.mp4'}"
        if os.path.abspath(output_path) == os.path.abspath(video_path):
            print("output_path must differ from video_path")
            return

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", ",".join(filters),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-y", output_path,
        ]
        print(f"Drawing {len(filters)} overlay(s) onto {video_path}")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Text rendered to: {output_path}")
            return {"video_path": output_path, "overlay_count": len(filters)}
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr.decode()[-500:]}")
            return
        except Exception as e:
            print(f"Text overlay failed: {e}")
            return
