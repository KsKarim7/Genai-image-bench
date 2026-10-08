"""Provider adapters.

Each provider hides its own auth, request shape and response parsing behind a single
`generate()` coroutine returning a GenerationResult. Adding a provider means adding
one class; the runner, scorer and report never learn provider-specific details.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx

# Estimated USD per image from published pricing at time of writing.
# These are ESTIMATES used for relative comparison, not measured billing.
COST_PER_IMAGE = {
    "gemini-flash-image": 0.0,      # free tier
    "pollinations": 0.0,
    "huggingface": 0.0,             # free inference tier
}


@dataclass
class GenerationResult:
    provider: str
    prompt_id: str
    repeat: int
    ok: bool
    latency_s: float
    cost_usd: float = 0.0
    image_path: Optional[str] = None
    # Failures are recorded, never silently dropped. The reason matters:
    # a rate limit is an operational problem, a refusal is a content-policy
    # problem, a parse error is an integration problem. Collapsing them into
    # "failed" throws away the distinction a pipeline decision depends on.
    error_kind: Optional[str] = None     # rate_limit | timeout | refused | parse | http | unknown
    error_detail: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderUnavailable(Exception):
    """Raised at construction when required credentials are missing."""


class BaseProvider:
    name: str = "base"
    # Conservative default; free tiers rate-limit aggressively.
    max_concurrency: int = 2

    def available(self) -> bool:
        return True

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> bytes:
        """Return raw image bytes, or raise. Subclasses implement."""
        raise NotImplementedError

    @staticmethod
    def classify_error(exc: Exception) -> tuple[str, str]:
        """Map an exception to (error_kind, detail)."""
        if isinstance(exc, httpx.TimeoutException):
            return "timeout", str(exc) or "request timed out"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return "rate_limit", f"HTTP 429: {exc.response.text[:200]}"
            if code in (400, 403) and "safety" in exc.response.text.lower():
                return "refused", exc.response.text[:200]
            return "http", f"HTTP {code}: {exc.response.text[:200]}"
        if isinstance(exc, (KeyError, IndexError, ValueError)):
            return "parse", f"{type(exc).__name__}: {exc}"
        return "unknown", f"{type(exc).__name__}: {exc}"


class GeminiFlashImage(BaseProvider):
    """Google Gemini 2.5 Flash Image (codename Nano Banana) via generateContent.

    Returns image data inline as base64 in the candidate parts, alongside any text
    parts the model emits. We take the first inline image part and ignore prose.
    """

    name = "gemini-flash-image"
    model_id = "gemini-2.5-flash-image"
    max_concurrency = 2

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> bytes:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )
        resp = await client.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            # Prompt feedback carries the block reason when the request was refused.
            feedback = data.get("promptFeedback", {})
            raise ValueError(f"no candidates returned; feedback={feedback}")

        for part in candidates[0].get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])

        finish = candidates[0].get("finishReason", "unknown")
        raise ValueError(f"no inline image part in response (finishReason={finish})")


class Pollinations(BaseProvider):
    """Keyless endpoint. Included as a zero-friction baseline.

    No auth means no quota guarantees either, so treat its latency numbers as
    indicative only — they include whatever queueing the public endpoint is doing.
    """

    name = "pollinations"
    max_concurrency = 2

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> bytes:
        resp = await client.get(
            f"https://image.pollinations.ai/prompt/{httpx.URL(path=prompt).path.lstrip('/')}",
            params={"nologo": "true"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"expected image bytes, got content-type={content_type!r}")
        return resp.content


class HuggingFaceInference(BaseProvider):
    """Hugging Face Inference API. Model is configurable via HF_IMAGE_MODEL.

    Returns raw image bytes on success and a JSON error body otherwise, including
    the 503 'model loading' case which we surface rather than silently waiting out.
    """

    name = "huggingface"
    max_concurrency = 1

    def __init__(self) -> None:
        self.token = os.getenv("HF_TOKEN", "").strip()
        self.model = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

    def available(self) -> bool:
        return bool(self.token)

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> bytes:
        resp = await client.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"inputs": prompt},
        )
        if resp.status_code == 503:
            raise ValueError("model cold-starting (HTTP 503) — retry shortly")
        resp.raise_for_status()
        if not resp.headers.get("content-type", "").startswith("image/"):
            raise ValueError(f"expected image bytes, got: {resp.text[:200]}")
        return resp.content


ALL_PROVIDERS: list[type[BaseProvider]] = [
    GeminiFlashImage,
    Pollinations,
    HuggingFaceInference,
]


def load_available() -> list[BaseProvider]:
    """Instantiate every provider, skipping those without credentials.

    A missing key is a normal condition, not an error — the run proceeds with
    whatever is configured so the harness is useful before you have every key.
    """
    live: list[BaseProvider] = []
    for cls in ALL_PROVIDERS:
        try:
            inst = cls()
        except ProviderUnavailable as exc:
            print(f"  skip {cls.name}: {exc}")
            continue
        if not inst.available():
            print(f"  skip {cls.name}: no credentials configured")
            continue
        live.append(inst)
    return live
