"""Shared runtime facts: where the project lives, what device to use, and the
lazily-cached heavy models. Import this instead of recomputing paths per module.
"""
import os
import threading
from pathlib import Path

# environment/config/runtime.py -> project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# imagebind_model.imagebind_huge() resolves this path relative to the cwd.
IMAGEBIND_CHECKPOINT = PROJECT_ROOT / ".checkpoints" / "imagebind_huge.pth"

IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
IMAGEBIND_BYTES = 4803584173


# Names of variables this project sourced from .env, for diagnostics.
_from_env_file = set()


def load_env():
    """Load <root>/.env, overriding the ambient environment.

    .env wins deliberately. A stale global export of the same name is otherwise
    invisible and silently beats the key you just put in the project file - which
    is exactly how you end up debugging a 404 against the wrong account.
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not value:
            continue
        os.environ[key] = value
        _from_env_file.add(key)
    return True


def env_source(name):
    """Where the current value of `name` came from: '.env', 'shell', or None."""
    if name in _from_env_file:
        return ".env"
    return "shell" if os.environ.get(name) else None


# Roles whose execute() needs the TTS/voice-cloning models. They import fine on a
# lean install (the heavy imports are deferred), so without an explicit gate the
# router happily plans around a tool that dies mid-run.
VOICE_ROLES = {
    "VoiceGenerator", "TTSInfer", "SVCSingle", "SVCCoverist",
    "StandUpSynth", "CrossTalkSynth",
}

COSYVOICE_MODEL = PROJECT_ROOT / "tools" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"


def voice_stack_available():
    """True when the TTS models are actually on disk."""
    return COSYVOICE_MODEL.is_dir()


def get_device():
    """Torch device string. Override with VIDEOAGENT_DEVICE=cpu to force CPU."""
    override = os.environ.get("VIDEOAGENT_DEVICE")
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_imagebind_lock = threading.Lock()
_imagebind_model = None


def get_imagebind():
    """Return the shared ImageBind encoder, loading it at most once per process.

    The upstream code called imagebind_huge(pretrained=True).cuda() inside both
    upsert() and query(), so every indexing batch and every search reloaded a
    4.5 GB checkpoint from disk. Cache it and honour VIDEOAGENT_DEVICE.
    """
    global _imagebind_model
    if _imagebind_model is not None:
        return _imagebind_model

    with _imagebind_lock:
        if _imagebind_model is not None:
            return _imagebind_model

        if not IMAGEBIND_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"ImageBind checkpoint missing at {IMAGEBIND_CHECKPOINT}.\n"
                "Run `make models` to download it."
            )

        from imagebind.models import imagebind_model

        # imagebind_huge() resolves ".checkpoints/imagebind_huge.pth" relative to
        # the process cwd, so pin cwd to the project root for the load only.
        previous_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT)
            model = imagebind_model.imagebind_huge(pretrained=True)
        finally:
            os.chdir(previous_cwd)

        model = model.to(get_device())
        model.eval()
        _imagebind_model = model
        return _imagebind_model


# Candidate default fonts for ffmpeg drawtext, in preference order.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def get_font(preferred=None):
    """Absolute path to a usable TTF for drawtext. Raises if none is found,
    because ffmpeg's own error for a missing font is close to unreadable."""
    candidates = ([preferred] if preferred else []) + _FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        "No usable font found for text rendering. Install one "
        "(apt install fonts-dejavu-core) or pass font_file explicitly."
    )
