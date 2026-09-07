import json
import os
import re
import subprocess

from pydantic import BaseModel, Field

from environment.agents.base import BaseTool
from environment.config.llm import gpt
from environment.config.transcribe import transcribe_segments


class VoiceoverTimeline(BaseTool):
    """
    Build the edit timeline from a source video's own voice-over, with no speech
    synthesis. Transcribes the narration, selects which lines to keep for the
    requested edit (a trailer, a recap, a highlight cut), trims the original audio
    down to just those lines, and writes the scene file and timestamp file that
    VideoSearcher and VideoEditor consume.

    Use this when the narration already exists in the source footage. It is the
    counterpart to VoiceGenerator, which synthesises new narration instead.
    RhythmDetector is the alternative spine for music-led edits - it finds no cut
    points in speech, so it is not a substitute here.
    """

    class InputSchema(BaseTool.BaseInputSchema):
        media_path: str = Field(
            ...,
            description="File path to the source video containing the voice-over"
        )
        reqs: str = Field(
            ...,
            description="What to build, in the user's words, e.g. 'a 60 second trailer that sets up the conflict'"
        )
        audio_path: str = Field(
            None,
            description="Output path for the trimmed voice-over track. Defaults to dataset/video_edit/audio_analysis/gen_audio.wav"
        )
        video_scene_path: str = Field(
            None,
            description="Output path for the scene file consumed by VideoSearcher"
        )
        timestamp_path: str = Field(
            None,
            description="Output path for the cut-point file consumed by VideoEditor"
        )
        target_duration: float = Field(
            None,
            description="Approximate length of the finished edit in seconds. Leave unset to keep the whole voice-over."
        )

    class OutputSchema(BaseModel):
        audio_path: str = Field(..., description="File path to the trimmed voice-over track")
        video_scene_path: str = Field(..., description="File path to the scene file for VideoSearcher")
        timestamp_path: str = Field(..., description="File path to the cut-point file for VideoEditor")
        duration: float = Field(..., description="Duration of the trimmed voice-over, in seconds")
        line_count: int = Field(..., description="Number of narration lines kept")

    def _default_paths(self, params):
        base = os.path.join("dataset", "video_edit")
        audio = params.audio_path or os.path.join(base, "audio_analysis", "gen_audio.wav")
        scenes = params.video_scene_path or os.path.join(base, "scene_output", "video_scene.json")
        stamps = params.timestamp_path or os.path.join(base, "audio_analysis", "cut_points.json")
        for path in (audio, scenes, stamps):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return audio, scenes, stamps

    def _select_lines(self, lines, reqs, target_duration):
        """Ask the model which narration lines to keep. Falls back to all of them."""
        if target_duration is None:
            return list(range(len(lines)))

        catalogue = "\n".join(
            f"{i}: [{s:.1f}s-{e:.1f}s, {e - s:.1f}s] {t}" for i, (s, e, t) in enumerate(lines)
        )
        prompt = (
            f"You are cutting a video down to: {reqs}\n"
            f"Target length: about {target_duration:.0f} seconds.\n\n"
            f"Below are the narration lines from the source, with their durations.\n"
            f"Choose the subset that best delivers the requested edit. Keep the "
            f"original order. The kept durations should add up to roughly the "
            f"target length.\n\n{catalogue}\n\n"
            f"Reply with only a JSON array of the line numbers to keep, e.g. [0, 4, 5, 9]."
        )

        try:
            response = gpt(user=prompt)
            text = (response.choices[0].message.content or "").strip()
            match = re.search(r"\[[\d,\s]*\]", text)
            if not match:
                raise ValueError(f"no JSON array in model reply: {text[:120]}")
            kept = [i for i in json.loads(match.group(0)) if 0 <= i < len(lines)]
            if not kept:
                raise ValueError("model selected no lines")
            return sorted(set(kept))
        except Exception as e:
            print(f"Line selection failed ({e}); keeping the whole voice-over.")
            return list(range(len(lines)))

    def _describe_scenes(self, kept_lines, reqs):
        """One on-screen description per kept line, for VideoSearcher to match."""
        numbered = "\n".join(f"{i + 1}. {t}" for i, (_, _, t) in enumerate(kept_lines))
        prompt = (
            f"You are the editor on: {reqs}\n\n"
            f"For each narration line below, describe the footage that should be on "
            f"screen while it plays. Describe visible action, setting and subject - "
            f"these descriptions are used to search the source footage, so they must "
            f"describe what a camera sees, not abstract ideas.\n\n{numbered}\n\n"
            f"Reply with exactly {len(kept_lines)} descriptions. Begin each one with "
            f"///// and write nothing else."
        )

        try:
            response = gpt(user=prompt)
            text = (response.choices[0].message.content or "").strip()
            sections = [s.strip() for s in text.split("/////") if s.strip()]
            if len(sections) >= len(kept_lines):
                return sections[:len(kept_lines)]
            print(f"Model returned {len(sections)} scenes for {len(kept_lines)} lines; "
                  f"padding with the narration text.")
            sections += [t for (_, _, t) in kept_lines[len(sections):]]
            return sections
        except Exception as e:
            print(f"Scene description failed ({e}); falling back to narration text.")
            return [t for (_, _, t) in kept_lines]

    def _trim_audio(self, media_path, kept_lines, audio_path):
        """Concatenate just the kept spans of the original narration."""
        # aselect keeps the chosen spans; asetpts rebuilds a gapless timeline.
        spans = "+".join(f"between(t,{s},{e})" for s, e, _ in kept_lines)
        cmd = [
            "ffmpeg", "-i", media_path, "-vn",
            "-af", f"aselect='{spans}',asetpts=N/SR/TB",
            "-ac", "1", "-ar", "22050", "-y", audio_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            check=True, capture_output=True, text=True,
        )
        return float(probe.stdout.strip())

    def execute(self, **kwargs):
        params = self.InputSchema(**kwargs)
        print("Parameters validated successfully")

        if not os.path.exists(params.media_path):
            print(f"Source not found: {params.media_path}")
            return

        audio_path, scene_path, stamp_path = self._default_paths(params)

        try:
            lines = transcribe_segments(params.media_path)
            if not lines:
                print("No narration found in the source; nothing to build a timeline from.")
                return
            print(f"Transcribed {len(lines)} narration lines")

            kept_indices = self._select_lines(lines, params.reqs, params.target_duration)
            kept_lines = [lines[i] for i in kept_indices]
            print(f"Keeping {len(kept_lines)} of {len(lines)} lines")

            duration = self._trim_audio(params.media_path, kept_lines, audio_path)
            print(f"Trimmed voice-over: {duration:.2f}s -> {audio_path}")

            scenes = self._describe_scenes(kept_lines, params.reqs)
            with open(scene_path, "w", encoding="utf-8") as handle:
                json.dump({"segment_scene": "".join(f"/////{s}\n" for s in scenes)},
                          handle, indent=2, ensure_ascii=False)

            # Cut points live on the trimmed timeline, not the source timeline:
            # VideoEditor reads consecutive timestamps as the periods to fill.
            chunks, elapsed = [], 0.0
            for (start, end, text), scene in zip(kept_lines, scenes):
                elapsed += (end - start)
                chunks.append({"timestamp": round(elapsed, 3), "content": text})
            with open(stamp_path, "w", encoding="utf-8") as handle:
                json.dump({"sentence_data": {"chunks": chunks}}, handle,
                          indent=2, ensure_ascii=False)

            print(f"Wrote {len(scenes)} scenes -> {scene_path}")
            print(f"Wrote {len(chunks)} cut points -> {stamp_path}")

            return {
                "audio_path": audio_path,
                "video_scene_path": scene_path,
                "timestamp_path": stamp_path,
                "duration": round(duration, 3),
                "line_count": len(kept_lines),
            }

        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr.decode()[-400:]}")
            return
        except Exception as e:
            print(f"Timeline build failed: {e}")
            return
