"""Speech-to-text via whisper.cpp.

The deployment target is CPU-only servers, where whisper.cpp is far faster than
the transformers implementation and carries no torch dependency of its own.
Everything in the project transcribes through `transcribe_segments()`.
"""
import contextlib
import os
import subprocess
import tempfile
import threading

from environment.config.runtime import PROJECT_ROOT

# Measured on the deployment target (16-core EPYC-Rome, avx2, no avx512),
# transcribing 122s of speech:
#
#   large-v3-turbo-q5_0   291.2s    0.4x realtime   15 segments
#   small.en               39.0s    3.1x realtime   32 segments
#   base.en                11.6s   10.6x realtime   32 segments
#
# large-v3-turbo is a GPU-era choice and unusable here - a 1-hour source would
# take 2.5 hours. small.en keeps real accuracy headroom for noisy montage audio
# at 3.1x (1 hour of source in ~19 minutes). Drop to base.en if throughput
# matters more than accuracy; use the non-".en" names for non-English audio.
DEFAULT_MODEL = os.environ.get("VIDEOAGENT_WHISPER_MODEL", "small.en")
MODELS_DIR = PROJECT_ROOT / "tools" / "whispercpp-models"

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
    name = name or DEFAULT_MODEL
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
    name = name or DEFAULT_MODEL
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
