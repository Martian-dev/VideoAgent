#!/usr/bin/env python3
"""Check that this machine can actually run VideoAgent, and say how to fix
whatever cannot. Exits non-zero if any required check fails.

    python scripts/doctor.py           # environment only
    python scripts/doctor.py --api     # also make one live Meta Model API call
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.config.runtime import (  # noqa: E402
    IMAGEBIND_BYTES, IMAGEBIND_CHECKPOINT, env_source, get_device, load_env,
)
from environment.config.transcribe import (  # noqa: E402
    DEFAULT_MODEL as WHISPER_MODEL, MODELS_DIR as WHISPER_MODELS_DIR,
)

load_env()

GB = 1024 ** 3
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(name, status, detail, fix=""):
    results.append((name, status, detail, fix))


def check_python():
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) in {(3, 10), (3, 11)}:
        record("python", PASS, detail)
    else:
        record("python", FAIL, detail,
               "pins need 3.10/3.11 - run `make setup` to build .venv")


def check_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        record("ffmpeg", FAIL, "not on PATH", "apt install ffmpeg")
        return
    out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout
    record("ffmpeg", PASS, out.splitlines()[0].split(" Copyright")[0])


def check_torch():
    try:
        import torch
    except ImportError:
        record("torch", FAIL, "not installed", "make setup")
        return None
    record("torch", PASS, f"{torch.__version__} (cuda build {torch.version.cuda})")
    return torch


def check_gpu(torch):
    """The trap this fork exists to catch: a torch build with no kernels for
    the installed GPU. Upstream pins torch 2.3.1+cu121, which has no sm_120,
    so Blackwell cards install cleanly and then fail at the first .cuda()."""
    if torch is None:
        return
    if not torch.cuda.is_available():
        record("gpu", WARN, "no CUDA device; will run on CPU (slow but works)",
               "set VIDEOAGENT_DEVICE=cpu to silence this")
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    mem = torch.cuda.get_device_properties(0).total_memory / GB
    detail = f"{name} {device_arch}, {mem:.1f} GB"

    if device_arch in supported:
        record("gpu", PASS, detail)
    else:
        record("gpu", FAIL, f"{detail}; this torch build supports {' '.join(supported)}",
               "reinstall torch with a CUDA version covering this arch (make setup)")


def check_imagebind():
    try:
        import imagebind  # noqa: F401
        record("imagebind pkg", PASS, "importable")
    except ImportError as e:
        record("imagebind pkg", FAIL, str(e)[:60], "make setup")


def check_whispercpp():
    try:
        import pywhispercpp  # noqa: F401
        record("whisper.cpp pkg", PASS, "importable")
    except ImportError as e:
        record("whisper.cpp pkg", FAIL, str(e)[:60], "make setup")


def check_weights():
    if IMAGEBIND_CHECKPOINT.exists():
        size = IMAGEBIND_CHECKPOINT.stat().st_size
        if size == IMAGEBIND_BYTES:
            record("imagebind weights", PASS, f"{size/GB:.2f} GB")
        else:
            record("imagebind weights", FAIL,
                   f"{size/GB:.2f} GB, expected {IMAGEBIND_BYTES/GB:.2f} GB",
                   "make models  (resumes)")
    else:
        record("imagebind weights", FAIL, "missing", "make models")

    ggml = WHISPER_MODELS_DIR / f"ggml-{WHISPER_MODEL}.bin"
    if ggml.exists():
        record("whisper.cpp model", PASS,
               f"{WHISPER_MODEL}, {ggml.stat().st_size/GB:.2f} GB")
    else:
        record("whisper.cpp model", FAIL, f"{WHISPER_MODEL} missing", "make models")


def check_key():
    key = os.environ.get("META_AI_API_KEY")
    if key:
        record("META_AI_API_KEY", PASS,
               f"{key[:8]}...{key[-4:]} (from {env_source('META_AI_API_KEY')})")
    else:
        record("META_AI_API_KEY", FAIL, "not set",
               "put it in .env (see .env.example) or export it")


def check_api():
    try:
        from environment.config.llm import claude, resolve_model
        response = claude(user="Reply with exactly: OK")
        text = (response.choices[0].message.content or "").strip()
        record("model api", PASS if text else FAIL,
               f"{resolve_model('claude')} -> {text[:20]!r}",
               "" if text else "model returned empty content; check account access")
    except Exception as e:
        record("model api", FAIL, f"{type(e).__name__}: {str(e)[:70]}",
               "check META_AI_API_KEY and llm.base_url in config.yml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="store_true",
                        help="also make one live API call (costs a few tokens)")
    args = parser.parse_args()

    check_python()
    check_ffmpeg()
    torch = check_torch()
    check_gpu(torch)
    check_imagebind()
    check_whispercpp()
    check_weights()
    check_key()
    if args.api:
        check_api()

    width = max(len(n) for n, *_ in results)
    print()
    for name, status, detail, _ in results:
        mark = {PASS: "✓", WARN: "!", FAIL: "✗"}[status]
        print(f"  {mark} {name:<{width}}  {detail}")

    problems = [(n, f) for n, s, _, f in results if s == FAIL and f]
    if problems:
        print("\n  To fix:")
        for name, fix in problems:
            print(f"    {name:<{width}}  {fix}")

    failed = sum(1 for _, s, _, _ in results if s == FAIL)
    print(f"\n  device: {get_device()}   {len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
