"""CLI entry point: harness.run --source X --instruction Y"""
import argparse
import sys
from pathlib import Path

from openai import OpenAI

from harness import budget as budget_mod
from harness import loop, prompt, trace


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    parser.add_argument("--source-dir", required=True, help="directory holding the source media")
    parser.add_argument("--instruction", required=True, help="path to the task instruction")
    parser.add_argument("--work-dir", default="/app/work", help="writable scratch space")
    parser.add_argument("--output", default="/app/output/final_video.mp4")
    parser.add_argument("--trace", default="/app/output/trace.jsonl")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=20_000_000,
                        help="cumulative spend cap; a cost guard, not a context limit")
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--max-view-mb", type=int, default=24)
    parser.add_argument("--keep-looks", type=int, default=4,
                        help="how many recent looks stay in context")
    parser.add_argument("--model", help="override the model id")
    parser.add_argument("--base-url", help="override the API base url")
    parser.add_argument("--api-key-env", default="META_AI_API_KEY")
    args = parser.parse_args(argv)

    import os
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"No API key: set {args.api_key_env}")
        return 2

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    instruction = Path(args.instruction).read_text(encoding="utf-8").strip()
    system = prompt.build(
        source_dir=args.source_dir, work_dir=args.work_dir,
        output_path=args.output, max_turns=args.max_turns,
        max_view_mb=args.max_view_mb,
    )

    client = OpenAI(api_key=api_key,
                    base_url=args.base_url or os.environ.get(
                        "MONTAGE_BASE_URL", "https://api.meta.ai/v1"))
    model = args.model or os.environ.get("MONTAGE_MODEL", "muse-spark-1.2-contributor")

    recorder = trace.Trace(args.trace)
    run = loop.Run(
        client=client, model=model, system=system, instruction=instruction,
        work_dir=work_dir, output_path=args.output,
        budget=budget_mod.Budget(args.max_turns, args.max_tokens, args.max_seconds),
        trace=recorder, max_view_bytes=args.max_view_mb * 1024 * 1024,
        keep_last_looks=args.keep_looks,
    )

    print(f"model={model}  max_turns={args.max_turns}  trace={args.trace}")
    submitted, stop_reason = run.run()
    print(f"\nstop_reason={stop_reason}  submitted={submitted}")
    print(f"trace summary: {recorder.summary()}")
    return 0 if submitted else 1


if __name__ == "__main__":
    sys.exit(main())
