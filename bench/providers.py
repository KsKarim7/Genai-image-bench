"""Provider adapters.

Each provider hides its own auth, request shape and response parsing behind a single
`generate()` coroutine returning a GenerationResult. Adding a provider means adding
one class; the runner, scorer and report never learn provider-specific details.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

# LIST prices, not measured spend: what a provider publishes per image on the tier
# this harness uses. Every configured provider is on a free tier, so these are zero
# and the report says "free tier" rather than "$0.00". README carries paid-tier list
# prices as a separate reference table, cited and dated.
LIST_PRICE_USD_PER_IMAGE = {
    "gemini-flash-image": 0.0,
    "pollinations": 0.0,
    "huggingface": 0.0,
}


@dataclass
class GenerationResult:
    provider: str
    prompt_id: str
    blind_id: str
    ok: bool
    latency_s: float
    cost_usd: float = 0.0
    image_path: Optional[str] = None
    # Failures are recorded, never silently dropped. The reason matters:
    # a rate limit is an operational problem, a refusal is a content-policy
    # problem, a parse error is an integration problem. Collapsing them into
    # "failed" throws away the distinction a pipeline decision depends on.
    # parse means an exchange we could not form or read; http_client is permanent,
    # http_server and network are transient.
    error_kind: Optional[str] = None     # timeout | network | rate_limit | payment_required
                                         # | refused | http_client | http_server | parse | unknown
    error_detail: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # results.json is the one run artifact the scoring module reads, so it must
    # carry no field that maps a blind id back to a provider.
    _UNBLINDING = ("blind_id", "image_path")

    def to_results_row(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in self._UNBLINDING}

    def to_blind_entry(self) -> dict:
        return {
            "provider": self.provider,
            "prompt_id": self.prompt_id,
            "image_path": self.image_path,
        }


@dataclass
class GeneratedImage:
    """One successful generation. meta carries whatever per-request diagnostics
    the adapter chose to surface; the runner stores it without interpreting it."""

    data: bytes
    content_type: str
    meta: dict = field(default_factory=dict)


def _payment_detail(response: httpx.Response) -> str:
    """x402 sends an empty JSON body; the challenge rides in a base64 header."""
    header = response.headers.get("payment-required")
    if not header:
        return response.text[:200] or "no body"
    try:
        challenge = json.loads(base64.b64decode(header + "=" * (-len(header) % 4)))
    except Exception:
        return f"undecodable payment-required header ({len(header)} bytes)"
    accepts = (challenge.get("accepts") or [{}])[0]
    asset = accepts.get("extra", {}).get("name") or accepts.get("asset", "?")
    return (
        f"{challenge.get('error', 'payment required')} - "
        f"{accepts.get('amount', '?')} {asset} on {accepts.get('network', '?')} "
        f"for {challenge.get('resource', {}).get('serviceName', 'resource')}"
    )


class ContentRefused(Exception):
    """The provider declined on content-policy grounds. Never worth retrying."""


# finishReason values meaning the model declined, as opposed to the call failing.
_REFUSAL_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "IMAGE_SAFETY",
    "RECITATION",
    "SPII",
}


class BaseProvider:
    name: str = "base"
    # Conservative default; free tiers rate-limit aggressively.
    max_concurrency: int = 2

    def available(self) -> bool:
        return True

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        """Return a GeneratedImage, or raise. Subclasses implement."""
        raise NotImplementedError

    @staticmethod
    def classify_error(exc: Exception) -> tuple[str, str]:
        """Map an exception to (error_kind, detail)."""
        if isinstance(exc, ContentRefused):
            return "refused", str(exc)
        # TimeoutException is itself a TransportError, so it must be tested first.
        if isinstance(exc, httpx.TimeoutException):
            return "timeout", str(exc) or "request timed out"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return "rate_limit", f"HTTP 429: {exc.response.text[:200]}"
            if code == 402:
                return "payment_required", f"HTTP 402: {_payment_detail(exc.response)}"
            kind = "http_server" if code >= 500 else "http_client"
            return kind, f"HTTP {code}: {exc.response.text[:200]}"
        if isinstance(exc, httpx.TransportError):
            return "network", f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (httpx.InvalidURL, KeyError, IndexError, ValueError)):
            return "parse", f"{type(exc).__name__}: {exc}"
        return "unknown", f"{type(exc).__name__}: {exc}"

    @staticmethod
    def retry_after_seconds(exc: Exception, cap: float = 120.0) -> float | None:
        """Delay a Retry-After header asks for, capped so a bad value cannot stall a run."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        raw = (exc.response.headers.get("retry-after") or "").strip()
        if not raw:
            return None
        try:
            return min(max(float(int(raw)), 0.0), cap)
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return min(max((when - datetime.now(timezone.utc)).total_seconds(), 0.0), cap)


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

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
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
            feedback = data.get("promptFeedback", {})
            reason = feedback.get("blockReason")
            if reason:
                raise ContentRefused(f"blockReason={reason}")
            raise ValueError(f"no candidates returned; feedback={feedback}")

        for part in candidates[0].get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return GeneratedImage(
                    base64.b64decode(inline["data"]),
                    mime,
                    {"model_version": data.get("modelVersion")},
                )

        finish = candidates[0].get("finishReason", "unknown")
        if finish in _REFUSAL_FINISH_REASONS:
            raise ContentRefused(f"finishReason={finish}")
        raise ValueError(f"no inline image part in response (finishReason={finish})")


class Pollinations(BaseProvider):
    """Keyless endpoint. Included as a zero-friction baseline.

    No auth means no quota guarantees either, so treat its latency numbers as
    indicative only — they include whatever queueing the public endpoint is doing.
    """

    name = "pollinations"
    # Measured: 6 simultaneous requests gave 5x HTTP 402 and one timeout, zero
    # successes; the same prompts sequentially succeed. The gate is on rate.
    max_concurrency = 1

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        # quote(safe="") and not httpx.URL(path=...): the latter leaves "/"
        # unescaped so a prompt can inject path segments, and raises InvalidURL
        # on "?" or "#".
        resp = await client.get(
            "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt, safe=""),
            params={"nologo": "true"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"expected image bytes, got content-type={content_type!r}")
        # A cache hit bypasses the rate gate and returns in a fraction of the
        # generation time, so a run that does not record this cannot tell a
        # latency measurement from a CDN read.
        return GeneratedImage(
            resp.content,
            content_type,
            {
                "x_cache": resp.headers.get("x-cache"),
                "model": resp.headers.get("x-model-used"),
            },
        )


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

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        resp = await client.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"inputs": prompt},
        )
        if resp.status_code == 503:
            raise ValueError("model cold-starting (HTTP 503) — retry shortly")
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"expected image bytes, got: {resp.text[:200]}")
        return GeneratedImage(resp.content, content_type, {"model": self.model})


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
        inst = cls()
        if not inst.available():
            print(f"  skip {cls.name}: no credentials configured")
            continue
        live.append(inst)
    return live
