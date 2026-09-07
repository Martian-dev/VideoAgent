"""Speech-to-text via whisper.cpp.

CPU-only by design: no torch, no CUDA, prebuilt wheels. Everything in the
project transcribes through `transcribe_segments()`.
"""
import contextlib
import os
import subprocess
import tempfile
import threading

from montage.config import WHISPER_MODELS_DIR, whisper_model

MODELS_DIR = WHISPER_MODELS_DIR

_lock = threading.Lock()
_models = {}


@contextlib.contextmanager
def _quiet():
    """whisper.cpp writes its model banner straight to the C-level stderr, which
    Python's logging cannot filter. Redirect fd 2 for the duration."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def get_model(name=None):
    """Load a whisper.cpp model, at most once per name per process."""
    name = name or whisper_model()
    if name in _models:
        return _models[name]

    with _lock:
        if name in _models:
            return _models[name]

        from pywhispercpp.model import Model

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with _quiet():
            _models[name] = Model(
                name,
                models_dir=str(MODELS_DIR),
                print_realtime=False,
                print_progress=False,
                print_timestamps=False,
            )
        return _models[name]


def model_is_present(name=None):
    """True when the model file is already on disk, so nothing will be downloaded."""
    name = name or whisper_model()
    return (MODELS_DIR / f"ggml-{name}.bin").is_file()


def to_wav16k(media_path):
    """whisper.cpp wants 16 kHz mono PCM; let ffmpeg decode whatever container."""
    handle, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    subprocess.run(
        ["ffmpeg", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", "-y", wav_path],
        check=True, capture_output=True,
    )
    return wav_path


def transcribe_segments(media_path, language=None, model_name=None):
    """Transcribe any media file.

    Returns [(start_seconds, end_seconds, text)]. whisper.cpp reports times in
    centiseconds, so they are converted here and callers can work in seconds.
    """
    wav_path = to_wav16k(media_path)
    try:
        model = get_model(model_name)
        kwargs = {"language": language} if language else {}
        with _quiet():
            segments = model.transcribe(wav_path, **kwargs)

        lines = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            start = segment.t0 / 100.0
            end = segment.t1 / 100.0
            if end <= start:
                end = start + 0.5
            lines.append((start, end, text))
        return lines
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
