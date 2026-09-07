import os

from openai import OpenAI

from environment.config.config import config

# Default model used for every role when the config does not override it.
# Muse Spark is text-out, natively perceives video/images/documents, has a 1M
# context window and supports tool calling + JSON responses, which covers every
# slot VideoAgent needs (router, editor, QA, captioning).
#
# 1.2-contributor rather than 1.3, because the Meta Model API does not serve
# every model in every region: an EU host's /v1/models lists only 1.2 and below,
# and asking for 1.3 there returns model_not_found. 1.2 is available everywhere
# we deploy. Set VIDEOAGENT_MODEL=muse-spark-1.3-contributor to use 1.3 where it
# is offered.
#
# Served by the Meta Model API directly (https://ai.developer.meta.com/docs), which
# is drop-in compatible with the OpenAI SDK. Note the bare model id -- no vendor
# prefix, unlike the aggregator-style "meta/muse-spark-1.2-contributor".
DEFAULT_MODEL = "muse-spark-1.2-contributor"
DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# Env var consulted when a role has no api_key in config.yml.
API_KEY_ENV = "META_AI_API_KEY"


def _setting(model_prefix, key, default=None):
    """Read llm.<prefix>_<key>, falling back to llm.<key>, then to `default`.

    Empty strings in the yaml count as unset so a half-filled config still boots.
    """
    llm_config = config.get('llm') or {}
    for candidate in (f"{model_prefix}_{key}" if model_prefix else None, key):
        if candidate and llm_config.get(candidate) not in (None, ''):
            return llm_config[candidate]
    return default


def resolve_model(model_prefix, model=None):
    """Pick the model id for a role.

    Order, most specific first:
      explicit argument > llm.<role>_model > $VIDEOAGENT_MODEL > llm.model > DEFAULT_MODEL

    $VIDEOAGENT_MODEL sits above the shared config value on purpose. The Meta
    Model API does not serve every model in every region - an EU host sees only
    muse-spark-1.2 and below - so one checkout must run different model ids in
    different places without editing tracked files. A per-role pin in config.yml
    still wins, so you can still hold one role on a specific model.
    """
    if model is not None:
        return model

    llm_config = config.get('llm') or {}
    per_role = llm_config.get(f"{model_prefix}_model")
    if per_role not in (None, ''):
        return per_role

    from_env = os.environ.get('VIDEOAGENT_MODEL')
    if from_env:
        return from_env

    shared = llm_config.get('model')
    return shared if shared not in (None, '') else DEFAULT_MODEL


def get_client(model_prefix=None):
    """Get OpenAI client with appropriate credentials based on model prefix"""
    api_key = _setting(model_prefix, 'api_key') or os.environ.get(API_KEY_ENV)
    base_url = _setting(model_prefix, 'base_url', DEFAULT_BASE_URL)

    if not api_key:
        raise RuntimeError(
            f"No API key for '{model_prefix}'. Set llm.{model_prefix}_api_key in "
            f"environment/config/config.yml, or export {API_KEY_ENV}."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def _build_messages(system=None, user=None, messages=None):
    if messages is not None:
        return messages

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if user is not None:
        messages.append({"role": "user", "content": user})
    return messages


def _complete(model_prefix, model=None, system=None, user=None, messages=None, **kwargs):
    client = get_client(model_prefix)
    return client.chat.completions.create(
        model=resolve_model(model_prefix, model),
        messages=_build_messages(system, user, messages),
        temperature=1,
        **kwargs,
    )


def deepseek(model=None, system=None, user=None, messages=None):
    try:
        return _complete(
            "deepseek", model, system, user, messages,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Muse Spark honours json_object, but a role pointed at another model may
        # not; the callers already prompt for JSON, so retry rather than hard-fail.
        if 'response_format' not in str(exc) and 'json' not in str(exc).lower():
            raise
        return _complete("deepseek", model, system, user, messages)


def claude(model=None, system=None, user=None, messages=None):
    return _complete("claude", model, system, user, messages)


def gemini(model=None, system=None, user=None, messages=None):
    return _complete("gemini", model, system, user, messages)


def gpt(model=None, system=None, user=None, messages=None):
    return _complete("gpt", model, system, user, messages)
