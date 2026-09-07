import os
import subprocess

from pydantic import BaseModel, Field

from environment.agents.base import BaseTool


def _escape_filter_path(path: str) -> str:
    r"""Escape a path for use inside an ffmpeg filtergraph argument.

    The filter parser strips one layer of escaping, so ':' (drive letters,
    Windows paths) and '\' both need escaping, and the whole value is then
    single-quoted by the caller.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


class SubtitleBurner(BaseTool):
    """
    Render subtitles permanently onto the video picture (open captions), reading
    cues from a .srt or .ass file. Use this when the text must be visible on any
    player, for example on social platforms that ignore sidecar subtitle tracks.
    Pair with SubtitleGenerator, which produces the .srt from the spoken audio.
    """

    class InputSchema(BaseTool.BaseInputSchema):
        video_path: str = Field(
            ...,
            description="File path to the source video"
        )
        srt_path: str = Field(
            ...,
            description="File path to the .srt or .ass subtitle file to render"
        )
        output_path: str = Field(
            None,
            description="Output video path. Defaults to the input path with a '_subtitled' suffix."
        )
        font_size: int = Field(
            24,
            description="Subtitle font size in points at the video's own resolution"
        )
        font_color: str = Field(
            "white",
            description="Subtitle fill colour, e.g. 'white' or '#FFE9A8'"
        )
        outline_width: int = Field(
            2,
            description="Outline thickness in pixels. Keeps text legible over bright footage."
        )
        margin_bottom: int = Field(
            40,
            description="Distance from the bottom edge of the frame, in pixels"
        )

    class OutputSchema(BaseModel):
        video_path: str = Field(
            ...,
            description="File path to the video with subtitles rendered in"
        )

    def _ass_color(self, color: str) -> str:
        """libass wants &HBBGGRR - note the byte order is reversed from HTML."""
        named = {
            "white": "FFFFFF", "black": "000000", "yellow": "FFFF00",
            "red": "FF0000", "green": "00FF00", "blue": "0000FF",
        }
        value = named.get(color.lower(), color.lstrip("#"))
        if len(value) != 6:
            value = "FFFFFF"
        return f"&H{value[4:6]}{value[2:4]}{value[0:2]}"

    def execute(self, **kwargs):
        params = self.InputSchema(**kwargs)
        print("Parameters validated successfully")

        video_path = params.video_path
        srt_path = params.srt_path

        for label, path in (("video", video_path), ("subtitle file", srt_path)):
            if not os.path.exists(path):
                print(f"Input {label} not found: {path}")
                return

        output_path = params.output_path
        if not output_path:
            stem, ext = os.path.splitext(video_path)
            output_path = f"{stem}_subtitled{ext or '.mp4'}"
        # Writing over the input mid-encode truncates it.
        if os.path.abspath(output_path) == os.path.abspath(video_path):
            print("output_path must differ from video_path")
            return

        style = ",".join([
            f"FontSize={params.font_size}",
            f"PrimaryColour={self._ass_color(params.font_color)}",
            f"OutlineColour={self._ass_color('black')}",
            f"Outline={params.outline_width}",
            f"MarginV={params.margin_bottom}",
            "BorderStyle=1",
        ])
        subtitle_filter = (
            f"subtitles='{_escape_filter_path(srt_path)}':force_style='{style}'"
        )

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-y", output_path,
        ]
        print(f"Executing command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Subtitles rendered to: {output_path}")
            return {"video_path": output_path}
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr.decode()[-500:]}")
            return
        except Exception as e:
            print(f"Subtitle burn-in failed: {e}")
            return
