"""One entry point to Muse. Text in, text out, with optional images.

The Meta Model API is drop-in compatible with the OpenAI SDK, so this is a thin
wrapper over chat.completions - the value it adds is retries, image encoding and
tolerant JSON extraction, which every caller in this project needs.
"""
import base64
import io
import json
import re
import time

from openai import OpenAI

from montage.config import API_BASE_URL, api_key, model

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key(), base_url=API_BASE_URL)
    return _client


def image_part(image):
    """A PIL image as an OpenAI-format image_url part."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def ask(prompt, images=(), retries=3):
    """Send a prompt (optionally with images) and return the reply text."""
    content = [{"type": "text", "text": prompt}]
    content.extend(image_part(im) for im in images)

    last_error = None
    for attempt in range(retries):
        try:
            response = client().chat.completions.create(
                model=model(),
                messages=[{"role": "user", "content": content if images else prompt}],
                temperature=1,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Muse call failed after {retries} attempts: {last_error}")


def ask_json(prompt, images=(), retries=3):
    """Ask for JSON and parse it, tolerating prose or code fences around it."""
    text = ask(prompt, images, retries)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Models wrap JSON in ```json fences or a sentence often enough to handle it.
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in reply: {text[:200]}")
