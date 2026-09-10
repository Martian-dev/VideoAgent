"""The system prompt.

Deliberately bare. It states what exists and what the tools do, and says nothing
about how to approach the work - no editorial guidance, no suggested workflow,
no hints about what makes an edit good. The task instruction supplies the goal;
everything the model does to reach it is the thing being measured, so any advice
here would contaminate the result.
"""
from harness.media import MAX_IMAGES

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
  Working dir:   {work_dir}   (writable scratch space)
{paths}
LIMITS
  {max_seconds} seconds of wall-clock. Each view is limited to {max_view_mb} MB;
  larger files must be made smaller with ffmpeg first. At most {max_images}
  images reach you per turn; a view of more arrives in batches over the
  following turns.

You cannot hear audio. `transcribe` is how you access speech.
Call submit(path) when the result is ready."""


def build(work_dir, max_seconds, max_view_mb, source_dir=None, output_path=None,
          max_images=MAX_IMAGES):
    """Render the system prompt.

    source_dir and output_path are optional and normally omitted. The task
    instruction states where the material is and where the result goes, so
    repeating it here is the harness restating the task - and in a real Harbor
    trial only the instruction is passed, with no way to supply paths anyway.
    Give them only when a caller genuinely has them out of band.
    """
    lines = []
    if source_dir:
        lines.append(f"  Source files:  {source_dir}")
    if output_path:
        lines.append(f"  Output:        {output_path}")
    if not lines:
        lines.append("  The task below names the files to work on and where the "
                     "result goes.")
    return SYSTEM.format(
        work_dir=work_dir, paths="\n".join(lines) + "\n",
        max_seconds=max_seconds, max_view_mb=max_view_mb, max_images=max_images,
    )
