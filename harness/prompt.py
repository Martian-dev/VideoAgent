"""The system prompt.

Deliberately bare. It states what exists and what the tools do, and says nothing
about how to approach the work - no editorial guidance, no suggested workflow,
no hints about what makes an edit good. The task instruction supplies the goal;
everything the model does to reach it is the thing being measured, so any advice
here would contaminate the result.
"""

SYSTEM = """You are working in a Linux container with a shell.

TOOLS
  shell(command)  Run a shell command. Returns stdout, stderr and exit code.
  view(path)      Put an image or video file into your context so you can see it.
                  Images and video only. Audio cannot be viewed.
  submit(path)    Declare the file at `path` your finished result and stop.

AVAILABLE COMMANDS
  ffmpeg, ffprobe   media inspection and manipulation
  transcribe FILE   speech-to-text with timestamps, for audio or video
  standard POSIX utilities (ls, cat, mkdir, cp, mv, rm, head, grep, ...)

ENVIRONMENT
  Source files:  {source_dir}
  Working dir:   {work_dir}   (writable scratch space)
  Output:        {output_path}

LIMITS
  {max_seconds} seconds of wall-clock. Each view is limited to {max_view_mb} MB;
  larger files must be made smaller with ffmpeg first.

You cannot hear audio. `transcribe` is how you access speech.
Call submit(path) when the result is ready."""


def build(source_dir, work_dir, output_path, max_seconds, max_view_mb):
    return SYSTEM.format(
        source_dir=source_dir, work_dir=work_dir, output_path=output_path,
        max_seconds=max_seconds, max_view_mb=max_view_mb,
    )
