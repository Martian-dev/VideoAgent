#!/usr/bin/env python3
"""Download the model weights VideoAgent needs. Idempotent and resumable:
re-running skips anything already complete.

    python scripts/fetch_models.py              # everything for the lean path
    python scripts/fetch_models.py --only whisper
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.config.runtime import (  # noqa: E402
    IMAGEBIND_BYTES,
    IMAGEBIND_CHECKPOINT,
    IMAGEBIND_URL,
)
from environment.config.transcribe import (  # noqa: E402
    DEFAULT_MODEL as WHISPER_MODEL,
    MODELS_DIR as WHISPER_MODELS_DIR,
    model_is_present,
)

GB = 1024 ** 3


def _progress(done, total, label):
    if not total:
        return
    pct = done / total * 100
    bar = "#" * int(pct / 2.5)
    sys.stdout.write(f"\r  {label} [{bar:<40}] {pct:5.1f}%  {done/GB:.2f}/{total/GB:.2f} GB")
    sys.stdout.flush()


def fetch_imagebind():
    dest = IMAGEBIND_CHECKPOINT
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == IMAGEBIND_BYTES:
        print(f"  ImageBind already complete ({IMAGEBIND_BYTES/GB:.2f} GB) -> {dest}")
        return

    have = dest.stat().st_size if dest.exists() else 0
    if have > IMAGEBIND_BYTES:
        print(f"  {dest} is larger than expected; removing and restarting")
        dest.unlink()
        have = 0

    request = urllib.request.Request(IMAGEBIND_URL)
    if have:
        print(f"  resuming from {have/GB:.2f} GB")
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request, timeout=60) as response:
        # A server that ignores Range replies 200; restart rather than corrupt.
        if have and response.status != 206:
            print("  server ignored resume request, starting over")
            have = 0
        mode = "ab" if have else "wb"
        done = have
        with open(dest, mode) as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                _progress(done, IMAGEBIND_BYTES, "ImageBind")
    print()

    size = dest.stat().st_size
    if size != IMAGEBIND_BYTES:
        raise SystemExit(
            f"ImageBind download incomplete: got {size} bytes, expected "
            f"{IMAGEBIND_BYTES}. Re-run to resume."
        )
    print(f"  ImageBind -> {dest}")


def fetch_whisper():
    """Pull the ggml weights for whisper.cpp. pywhispercpp fetches on first use;
    doing it here keeps the download out of the first transcription."""
    if model_is_present():
        print(f"  {WHISPER_MODEL} already present -> {WHISPER_MODELS_DIR}")
        return

    from environment.config.transcribe import get_model

    WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading ggml-{WHISPER_MODEL}.bin ...")
    get_model()  # downloads into MODELS_DIR, then loads once to validate

    if not model_is_present():
        raise SystemExit(f"whisper.cpp model missing after download: {WHISPER_MODEL}")
    print(f"  whisper.cpp -> {WHISPER_MODELS_DIR}/ggml-{WHISPER_MODEL}.bin")


TARGETS = {"imagebind": fetch_imagebind, "whisper": fetch_whisper}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(TARGETS), action="append",
                        help="fetch just this model (repeatable)")
    args = parser.parse_args()

    wanted = args.only or sorted(TARGETS)
    print(f"Fetching: {', '.join(wanted)}")
    for name in wanted:
        print(f"\n[{name}]")
        TARGETS[name]()
    print("\nAll requested models are in place. Run `make doctor` to verify.")


if __name__ == "__main__":
    main()
