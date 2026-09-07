"""montage - cut a finished video out of one montage, from an instruction.

    montage edit source.mp4 "a 60 second trailer" --duration 60 --subtitles
    montage index source.mp4          # build the index only
    montage doctor                    # check this machine can run it

Indexing is the slow part and is cached per source, so re-planning with a
different instruction is fast.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from montage import describe, plan, render, segment, transcribe
from montage.config import (API_BASE_URL, API_KEY_ENV, PROJECT_ROOT,
                            WHISPER_MODELS_DIR, env_source, model,
                            whisper_model)


def work_dir_for(video_path):
    """A stable per-source cache directory, keyed by path + size + mtime."""
    stat = Path(video_path).stat()
    key = f"{Path(video_path).resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    path = PROJECT_ROOT / ".montage" / f"{Path(video_path).stem}-{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bar(done, total, label):
    width = 28
    filled = int(width * done / total) if total else width
    sys.stdout.write(f"\r  {label} [{'#' * filled:<{width}}] {done}/{total}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def build_index(video_path, workers=4, frames=3, force=False):
    """Segment, transcribe and describe. Cached; returns the index entries."""
    work = work_dir_for(video_path)
    index_path = work / "index.json"
    if index_path.exists() and not force:
        entries = describe.load(index_path)
        print(f"  using cached index ({len(entries)} segments) -> {index_path}")
        return entries, work

    print("  segmenting at scene changes ...", flush=True)
    segments = segment.segment_video(video_path)
    total = sum(s.duration for s in segments)
    print(f"  {len(segments)} segments over {total:.1f}s")

    print(f"  transcribing with whisper.cpp ({whisper_model()}) ...", flush=True)
    started = time.time()
    lines = transcribe.transcribe_segments(video_path)
    print(f"  {len(lines)} narration lines in {time.time() - started:.1f}s")

    print(f"  describing segments with {model()} ...", flush=True)
    started = time.time()
    entries = describe.build_index(
        video_path, segments, lines, frames=frames, workers=workers,
        progress=lambda d, t: _bar(d, t, "described"),
    )
    print(f"  described {len(entries)} segments in {time.time() - started:.1f}s")

    describe.save(entries, index_path)
    with open(work / "transcript.json", "w", encoding="utf-8") as handle:
        json.dump([{"start": s, "end": e, "text": t} for s, e, t in lines],
                  handle, indent=2, ensure_ascii=False)
    return entries, work


def cmd_index(args):
    print(f"Indexing {args.video}")
    entries, work = build_index(args.video, args.workers, args.frames, args.force)
    print(f"\nIndex written to {work / 'index.json'}")
    for entry in entries[:5]:
        print(f"  [{entry['index']}] {entry['start']:.1f}-{entry['end']:.1f}s "
              f"{entry['description'][:90]}")
    if len(entries) > 5:
        print(f"  ... {len(entries) - 5} more")
    return 0


def cmd_edit(args):
    video = Path(args.video)
    if not video.exists():
        print(f"No such file: {video}")
        return 1

    print(f"Source: {video}")
    entries, work = build_index(video, args.workers, args.frames, args.force)

    print(f"\n  planning the edit with {model()} ...", flush=True)
    cuts, reasoning = plan.plan_edit(entries, args.instruction, args.duration)
    plan.save(cuts, reasoning, work / "edl.json")
    planned = sum(c["end"] - c["start"] for c in cuts)
    print(f"  {len(cuts)} cuts, {planned:.1f}s")
    if reasoning:
        print(f"  {reasoning}")
    for cut in cuts:
        label = f'  "{cut["text"]}"' if cut["text"] else ""
        print(f"    [{cut['index']:>3}] {cut['start']:7.1f}-{cut['end']:.1f}s{label}")

    out = Path(args.output or work / "final.mp4")
    print(f"\n  rendering ...", flush=True)
    started = time.time()
    rendered, spans = render.render(
        video, cuts, out, work / "parts", font=args.font, width=args.width,
        progress=lambda d, t: _bar(d, t, "cut"),
    )

    if args.subtitles:
        print("  transcribing the finished cut for subtitles ...", flush=True)
        cues = transcribe.transcribe_segments(rendered)
        if cues:
            srt = render.write_srt(cues, work / "final.srt")
            subbed = out.with_name(out.stem + "_subtitled" + out.suffix)
            render.burn_subtitles(rendered, srt, subbed)
            rendered = subbed
            print(f"  burned {len(cues)} subtitle cues")
        else:
            print("  no speech found in the cut; skipping subtitles")

    print(f"  rendered in {time.time() - started:.1f}s")
    print(f"\nDone: {rendered}")
    return 0


def cmd_doctor(args):
    checks = []

    def record(name, ok, detail, fix=""):
        checks.append((name, ok, detail, fix))

    version = sys.version_info
    record("python", version >= (3, 10), f"{version.major}.{version.minor}.{version.micro}",
           "needs 3.10+")

    ffmpeg = shutil.which("ffmpeg")
    record("ffmpeg", bool(ffmpeg),
           subprocess_version(ffmpeg) if ffmpeg else "not on PATH",
           "apt install ffmpeg")
    record("ffprobe", bool(shutil.which("ffprobe")),
           shutil.which("ffprobe") or "not on PATH", "apt install ffmpeg")

    try:
        import pywhispercpp  # noqa: F401
        record("whisper.cpp", True, "importable")
    except ImportError as e:
        record("whisper.cpp", False, str(e)[:50], "make setup")

    ggml = WHISPER_MODELS_DIR / f"ggml-{whisper_model()}.bin"
    record("whisper model", ggml.exists(),
           f"{whisper_model()}, {ggml.stat().st_size / 1e6:.0f} MB" if ggml.exists()
           else f"{whisper_model()} missing", "make models")

    try:
        render.find_font()
        record("font", True, Path(render.find_font()).name)
    except FileNotFoundError:
        record("font", False, "none found", "apt install fonts-dejavu-core")

    import os
    key = os.environ.get(API_KEY_ENV)
    record(API_KEY_ENV, bool(key),
           f"{key[:8]}...{key[-4:]} (from {env_source(API_KEY_ENV)})" if key
           else "not set", "put it in .env")

    if key and args.api:
        try:
            from montage.llm import ask
            reply = ask("Reply with exactly: OK")
            record("model api", bool(reply), f"{model()} -> {reply[:12]!r}",
                   "" if reply else "empty reply")
        except Exception as e:
            record("model api", False, f"{type(e).__name__}: {str(e)[:60]}",
                   "check the key and the model id for your region")

    width = max(len(n) for n, *_ in checks)
    print()
    for name, ok, detail, _ in checks:
        print(f"  {'✓' if ok else '✗'} {name:<{width}}  {detail}")
    failed = [(n, f) for n, ok, _, f in checks if not ok and f]
    if failed:
        print("\n  To fix:")
        for name, fix in failed:
            print(f"    {name:<{width}}  {fix}")
    print(f"\n  {len(checks) - len([1 for _, ok, _, _ in checks if not ok])}"
          f"/{len(checks)} checks passed")
    return 1 if any(not ok for _, ok, _, _ in checks) else 0


def subprocess_version(exe):
    import subprocess
    out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0].split(" Copyright")[0] if out else "present"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="montage", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=4,
                        help="parallel description calls (default 4)")
    common.add_argument("--frames", type=int, default=3,
                        help="frames sampled per segment (default 3)")
    common.add_argument("--force", action="store_true",
                        help="rebuild the index even if one is cached")

    p_index = sub.add_parser("index", parents=[common], help="build the segment index only")
    p_index.add_argument("video")
    p_index.set_defaults(func=cmd_index)

    p_edit = sub.add_parser("edit", parents=[common], help="cut a video from an instruction")
    p_edit.add_argument("video")
    p_edit.add_argument("instruction")
    p_edit.add_argument("--duration", type=float, help="target length in seconds")
    p_edit.add_argument("--output", "-o", help="output path")
    p_edit.add_argument("--subtitles", action="store_true", help="burn in subtitles")
    p_edit.add_argument("--font", help="path to a .ttf for on-screen text")
    p_edit.add_argument("--width", type=int, help="scale output to this width")
    p_edit.set_defaults(func=cmd_edit)

    p_doctor = sub.add_parser("doctor", help="check this machine can run it")
    p_doctor.add_argument("--api", action="store_true", help="also make one live API call")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
