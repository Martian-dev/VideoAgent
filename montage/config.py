"""Runtime configuration: paths, credentials, model ids.

Everything is env-driven with working defaults. A .env file in the project root
is loaded first and overrides the ambient environment - a stale global export of
the same name is otherwise invisible and silently wins, which is exactly how you
end up debugging a 404 against the wrong account.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_MODELS_DIR = PROJECT_ROOT / "tools" / "whispercpp-models"

API_BASE_URL = "https://api.meta.ai/v1"
API_KEY_ENV = "META_AI_API_KEY"

# muse-spark-1.2-contributor rather than 1.3: the Meta Model API does not serve
# every model in every region. An EU host's /v1/models lists only 1.2 and below,
# and asking for 1.3 there returns model_not_found.
DEFAULT_MODEL = "muse-spark-1.2-contributor"

# whisper.cpp model. Measured on a 16-core EPYC-Rome (avx2, no avx512) over 122s
# of speech: large-v3-turbo-q5_0 0.4x realtime, small.en 3.1x, base.en 10.6x.
# small.en keeps accuracy headroom for noisy montage audio at a usable speed.
DEFAULT_WHISPER_MODEL = "small.en"

_from_env_file = set()


def load_env():
    """Load <root>/.env, overriding the ambient environment."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value:
            os.environ[key] = value
            _from_env_file.add(key)
    return True


load_env()


def env_source(name):
    """Where the current value came from: '.env', 'shell', or None."""
    if name in _from_env_file:
        return ".env"
    return "shell" if os.environ.get(name) else None


def api_key():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"No API key. Put {API_KEY_ENV} in .env (see .env.example) or export it."
        )
    return key


def model():
    return os.environ.get("MONTAGE_MODEL") or DEFAULT_MODEL


def whisper_model():
    return os.environ.get("MONTAGE_WHISPER_MODEL") or DEFAULT_WHISPER_MODEL
