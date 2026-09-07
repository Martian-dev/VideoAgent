import os
import subprocess

from pydantic import BaseModel, Field

from environment.agents.base import BaseTool
from environment.config.transcribe import transcribe_segments


def _srt_time(seconds: float) -> str:
    """Seconds -> SRT timestamp, HH:MM:SS,mmm."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


class SubtitleGenerator(BaseTool):
    """
    Transcribe the speech in a video or audio file and write a standalone .srt
    subtitle file. Does not modify the video - pair it with SubtitleBurner to
    render the subtitles onto the picture, or ship the .srt as a sidecar file.
    """

    class InputSchema(BaseTool.BaseInputSchema):
        media_path: str = Field(
            ...,
            description="File path to the video or audio to transcribe"
        )
        srt_path: str = Field(
            None,
            description="Output path for the .srt file. Defaults to the input path with a .srt extension."
        )
        language: str = Field(
            None,
            description="Spoken language hint, e.g. 'en'. Leave unset to auto-detect."
        )
        max_line_chars: int = Field(
            42,
            description="Soft character limit per subtitle line before wrapping. 42 suits most players."
        )

    class OutputSchema(BaseModel):
        srt_path: str = Field(
            ...,
            description="File path to the generated .srt subtitle file"
        )
        cue_count: int = Field(
            ...,
            description="Number of subtitle cues written"
        )

    def _wrap(self, text: str, limit: int) -> str:
        words = text.split()
        lines, current = [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        # Two lines is the practical maximum before subtitles cover the frame.
        if len(lines) > 2:
            lines = [" ".join(lines[:-1]), lines[-1]]
        return "\n".join(lines)

    def execute(self, **kwargs):
        params = self.InputSchema(**kwargs)
        print("Parameters validated successfully")

        media_path = params.media_path
        if not os.path.exists(media_path):
            print(f"Input not found: {media_path}")
            return

        srt_path = params.srt_path or os.path.splitext(media_path)[0] + ".srt"

        try:
            cues = transcribe_segments(media_path, language=params.language)

            if not cues:
                print("No speech found; writing an empty subtitle file.")

            with open(srt_path, "w", encoding="utf-8") as handle:
                for index, (start, end, text) in enumerate(cues, start=1):
                    handle.write(f"{index}\n")
                    handle.write(f"{_srt_time(start)} --> {_srt_time(end)}\n")
                    handle.write(self._wrap(text, params.max_line_chars) + "\n\n")

            print(f"Wrote {len(cues)} cues to {srt_path}")
            return {"srt_path": srt_path, "cue_count": len(cues)}

        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed while extracting audio: {e.stderr.decode()[-400:]}")
            return
        except Exception as e:
            print(f"Subtitle generation failed: {e}")
            return
