"""A thin client for the Meta Model API: Muse Spark with structured (schema-constrained) JSON output.

Request formats are documented in docs/api-notes.md.
"""

from __future__ import annotations

import base64
import io
import os
import random
import time
from collections.abc import Sequence
from typing import Any, TypeVar

import openai
from PIL import Image
from pydantic import BaseModel

from ..core.errors import describe

BASE_URL = "https://api.meta.ai/v1"
MUSE_SPARK_MODEL = "muse-spark-1.3"

T = TypeVar("T", bound=BaseModel)


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (see .env.example)")
    return key


def image_data_url(image: Image.Image, max_side: int = 1280, quality: int = 88) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------- Muse Spark ----------


def ask_structured(
    prompt: str,
    images: Sequence[Image.Image],
    schema: type[T],
    *,
    reasoning_effort: str = "low",
    retries: int = 5,
) -> tuple[T, dict[str, Any]]:
    """One Chat Completions call constrained to `schema`; returns the parsed model and usage.

    Retries connection errors, 5xx, replies that fail validation, and 404s. The docs
    say not to retry 404, but during a 24-clip batch run the API returned
    `404 model not found` for about 1 in 6 calls of a model that answered the
    identical request moments later (docs/api-notes.md, 2026-10-26), so a 404 is
    retried with jittered exponential backoff (about 90 s in total) before it is treated as real.
    """
    client = openai.OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=3)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_data_url(image)}} for image in images]
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MUSE_SPARK_MODEL,
                reasoning_effort=reasoning_effort,
                messages=[{"role": "user", "content": content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                },
            )
            parsed = schema.model_validate_json(response.choices[0].message.content or "")
            usage = response.usage.model_dump() if response.usage else {}
            return parsed, usage
        except (openai.APIConnectionError, openai.InternalServerError, openai.NotFoundError, ValueError) as error:
            # ValueError covers a reply that fails Pydantic validation (e.g. out-of-range values).
            last_error = error
            # 404 bursts cluster under concurrency, so back off long enough to outlast one.
            time.sleep(min(30.0, 3.0 * 2**attempt) * random.uniform(0.75, 1.25))
    raise RuntimeError(f"Muse Spark failed after {retries + 1} attempts: {describe(last_error)}")
