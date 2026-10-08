"""Provider adapters.

One class per provider, hiding its auth, request shape and response parsing behind
generate(). Nothing downstream learns provider-specific details.
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

# List prices, not measured spend. Standard tier, never batch: batch halves the
# Gemini price and destroys the latency measurement. README has the cited table.
LIST_PRICE_USD_PER_IMAGE = {
    "gemini-3.1-flash-lite-image": 0.0336,
    "gemini-3.1-flash-image": 0.067,
    "pollinations": 0.0,
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
    # http_client is permanent, http_server and network transient; parse means an
    # exchange we could not form or read.
    error_kind: Optional[str] = None     # timeout | network | rate_limit | payment_required
                                         # | refused | http_client | http_server | parse | unknown
    error_detail: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # Defence in depth: results.json is not the scorer input any more, but it still
    # must not map a blind id to a provider.
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
    # Providers sharing a credential share a rate gate, so they share one semaphore.
    # None means this provider is alone on its own quota.
    quota_key: str | None = None

    def gate(self) -> str:
        return self.quota_key or self.name

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


def _refusal_reason(node: dict) -> str | None:
    """Only a known refusal value counts. The /interactions refusal shape is
    undocumented, so anything unrecognised stays a parse error with the body
    attached rather than being guessed into the wrong bucket."""
    for key in ("blockReason", "block_reason", "finishReason", "finish_reason"):
        value = node.get(key)
        if isinstance(value, str) and value.upper() in _REFUSAL_FINISH_REASONS:
            return f"{key}={value}"
    feedback = node.get("promptFeedback") or node.get("prompt_feedback") or {}
    reason = feedback.get("blockReason") or feedback.get("block_reason")
    return f"blockReason={reason}" if reason else None


class _GeminiImage(BaseProvider):
    """Shared adapter for the Gemini 3.1 image models.

    They are not on models/{id}:generateContent: they take a response_format on
    /v1beta/interactions and return the image under interaction.outputImage.

    Both models run at 1K. Lite supports nothing else, and holding the full model
    there too keeps the Lite-against-full comparison about the model rather than
    about output resolution.
    """

    model_id = ""
    image_size = "1K"
    aspect_ratio = "1:1"
    max_concurrency = 2
    quota_key = "google-ai-studio"

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        resp = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "model": self.model_id,
                "input": [{"type": "text", "text": prompt}],
                "response_format": {
                    "type": "image",
                    "aspect_ratio": self.aspect_ratio,
                    "image_size": self.image_size,
                },
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        interaction = payload.get("interaction") or payload

        image = (
            interaction.get("outputImage")
            or interaction.get("output_image")
            or {}
        )
        if image.get("data"):
            return GeneratedImage(
                base64.b64decode(image["data"]),
                image.get("mimeType") or image.get("mime_type") or "image/png",
                {
                    "model_version": interaction.get("model") or self.model_id,
                    "interaction_id": interaction.get("id"),
                    "image_size": self.image_size,
                },
            )

        for step in interaction.get("steps") or []:
            for part in step.get("content") or []:
                if part.get("type") == "image" and part.get("data"):
                    return GeneratedImage(
                        base64.b64decode(part["data"]),
                        part.get("mimeType") or "image/png",
                        {"model_version": self.model_id, "image_size": self.image_size},
                    )

        reason = _refusal_reason(interaction) or _refusal_reason(payload)
        if reason:
            raise ContentRefused(reason)
        raise ValueError(f"no image in interaction response: {str(payload)[:200]}")


class GeminiFlashLiteImage(_GeminiImage):
    name = "gemini-3.1-flash-lite-image"
    model_id = "gemini-3.1-flash-lite-image"


class GeminiFlashImage(_GeminiImage):
    name = "gemini-3.1-flash-image"
    model_id = "gemini-3.1-flash-image"


class Pollinations(BaseProvider):
    """Keyless endpoint, kept as a zero-friction baseline."""

    name = "pollinations"
    # Measured: 6 simultaneous requests gave 5x HTTP 402 and one timeout, zero
    # successes; the same prompts sequentially succeed. The gate is on rate.
    max_concurrency = 1

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        # quote(safe="") not httpx.URL(path=...): that leaves "/" unescaped so a
        # prompt can inject path segments, and raises InvalidURL on "?" or "#".
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
        # generation time, so without this a CDN read looks like a measurement.
        return GeneratedImage(
            resp.content,
            content_type,
            {
                "x_cache": resp.headers.get("x-cache"),
                "model": resp.headers.get("x-model-used"),
            },
        )


ALL_PROVIDERS: list[type[BaseProvider]] = [
    GeminiFlashLiteImage,
    GeminiFlashImage,
    Pollinations,
]


def load_available() -> list[BaseProvider]:
    """Instantiate every provider, skipping those with no credentials."""
    live: list[BaseProvider] = []
    for cls in ALL_PROVIDERS:
        inst = cls()
        if not inst.available():
            print(f"  skip {cls.name}: no credentials configured")
            continue
        live.append(inst)
    return live
