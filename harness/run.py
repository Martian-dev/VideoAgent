"""CLI entry point: harness.run --source X --instruction Y"""
import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

from harness import budget as budget_mod
from harness import loop, media, prompt, tools, trace


def harness_version():
    """What the ATIF trajectory records as the agent's version.

    ATIF requires one. Whoever installs the harness writes a HARNESS_VERSION
    file beside the package - the Harbor adapter puts the source revision in it
    - so a trajectory can be traced back to the code that produced it.
    """
    from_env = os.environ.get("HARNESS_VERSION")
    if from_env:
        return from_env.strip()
    try:
        version = (Path(__file__).resolve().parent.parent
                   / "HARNESS_VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    return version or "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    parser.add_argument("--source-dir", default=None,
                        help="optional; normally the instruction names the source")
    parser.add_argument("--instruction", required=True, help="path to the task instruction")
    parser.add_argument("--work-dir", default="/app/work", help="writable scratch space")
    parser.add_argument("--output", default=None,
                        help="optional; where an accepted submission is delivered. "
                             "Omit and the submitted path is the deliverable.")
    parser.add_argument("--runs-dir", default="/app/runs",
                        help="where trace.jsonl and trajectory.json go, kept out "
                             "of the graded output directory")
    parser.add_argument("--max-seconds", type=int, default=3600,
                        help="wall-clock bound (default 1 hour)")
    parser.add_argument("--max-cost", type=float, default=6.0,
                        help="USD spend cap")
    parser.add_argument("--max-errors", type=int, default=10,
                        help="consecutive failing tool calls before giving up")
    parser.add_argument("--no-rewind", action="store_true",
                        help="end on the first stall instead of rewinding once")
    parser.add_argument("--max-view-mb", type=int, default=24)
    parser.add_argument("--keep-looks", type=int, default=4,
                        help="how many recent looks stay in context")
    parser.add_argument("--max-images", type=int, default=media.MAX_IMAGES,
                        help="images per request; the API refuses more than 50, "
                             "and larger views are delivered in batches")
    parser.add_argument("--max-videos", type=int, default=media.MAX_VIDEOS,
                        help="videos per request; the API refuses more than 5")
    parser.add_argument("--api-retries", type=int, default=loop.API_RETRIES,
                        help="retries of a failed model request, with exponential "
                             "backoff, before the run fails")
    parser.add_argument("--retry-backoff", type=float, default=loop.RETRY_BACKOFF_S,
                        help="seconds before the first retry; doubles for each after")
    parser.add_argument("--request-timeout", type=float, default=loop.REQUEST_TIMEOUT_S,
                        help="seconds one model request may take, capped at the time left")
    parser.add_argument("--command-timeout", type=float, default=tools.COMMAND_TIMEOUT_S,
                        help="seconds one shell command may run, capped at the time left")
    parser.add_argument("--model", help="override the model id")
    parser.add_argument("--base-url", help="override the API base url")
    parser.add_argument("--api-key-env", default="META_AI_API_KEY")
    parser.add_argument("--agent-version", default=None,
                        help="version recorded in the ATIF trajectory (default: "
                             "$HARNESS_VERSION, else the HARNESS_VERSION file)")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"No API key: set {args.api_key_env}")
        return 2

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    instruction = Path(args.instruction).read_text(encoding="utf-8").strip()
    system = prompt.build(
        work_dir=args.work_dir, max_seconds=args.max_seconds,
        max_view_mb=args.max_view_mb, max_images=args.max_images,
        max_videos=args.max_videos, source_dir=args.source_dir, output_path=args.output,
    )

    # max_retries=0: the loop retries failed requests itself, and the SDK's own
    # retries would silently multiply that count.
    client = OpenAI(api_key=api_key,
                    base_url=args.base_url or os.environ.get(
                        "MONTAGE_BASE_URL", "https://api.meta.ai/v1"),
                    timeout=args.request_timeout, max_retries=0)
    model = args.model or os.environ.get("MONTAGE_MODEL", "muse-spark-1.2-contributor")

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    trace_path = runs_dir / "trace.jsonl"
    # ATIF, at the name anything that reads trajectories looks for.
    trajectory_path = runs_dir / "trajectory.json"
    # The chat messages as they went to the API, media redacted.
    messages_path = runs_dir / "messages.json"
    recorder = trace.Trace(trace_path)
    run = loop.Run(
        client=client, model=model, system=system, instruction=instruction,
        work_dir=work_dir, output_path=args.output,
        budget=budget_mod.Budget(max_seconds=args.max_seconds,
                                 max_cost_usd=args.max_cost,
                                 max_errors=args.max_errors),
        trace=recorder, max_view_bytes=args.max_view_mb * 1024 * 1024,
        keep_last_looks=args.keep_looks, max_images=args.max_images,
        max_videos=args.max_videos, allow_rewind=not args.no_rewind,
        trajectory_path=trajectory_path, messages_path=messages_path,
        agent_version=args.agent_version or harness_version(),
        api_retries=args.api_retries,
        retry_backoff=args.retry_backoff, request_timeout=args.request_timeout,
        command_timeout=args.command_timeout,
    )

    print(f"model={model}  max_seconds={args.max_seconds}  "
          f"max_cost=${args.max_cost}")
    print(f"runs_dir={runs_dir}")
    submitted, termination = run.run()
    # The score and the reason stay separate: a run that timed out is not the
    # same as an edit that was made and judged badly.
    print(f"\nsubmitted={bool(submitted)}  termination={termination}")
    print(f"path={submitted}")
    if run.delivered_elsewhere:
        print(f"delivered_elsewhere=True  (harness copied it to {args.output})")
    print(f"trace summary: {recorder.summary()}")
    print(f"trace={trace_path}  trajectory={trajectory_path}  "
          f"messages={messages_path}")
    return 0 if submitted else 1


if __name__ == "__main__":
    sys.exit(main())
