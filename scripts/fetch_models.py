#!/usr/bin/env python3
"""Download the whisper.cpp model. Idempotent - a second run is a no-op."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage.config import WHISPER_MODELS_DIR, whisper_model  # noqa: E402
from montage.transcribe import get_model, model_is_present  # noqa: E402


def main():
    name = whisper_model()
    if model_is_present(name):
        path = WHISPER_MODELS_DIR / f"ggml-{name}.bin"
        print(f"  {name} already present ({path.stat().st_size / 1e6:.0f} MB)")
        return
    WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading ggml-{name}.bin ...")
    get_model(name)
    if not model_is_present(name):
        raise SystemExit(f"model missing after download: {name}")
    print(f"  {name} -> {WHISPER_MODELS_DIR}")


if __name__ == "__main__":
    main()
