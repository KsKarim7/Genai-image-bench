"""Provider adapters.

One class per provider, hiding its auth, request shape and response parsing behind
generate(). Nothing downstream learns provider-specific details.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

# List prices, not measured spend. Every provider here runs inside a free
# allocation, so these are zero and the report says "free tier"; the README
# reference table carries the published rates behind them.
LIST_PRICE_USD_PER_IMAGE = {
    "pollinations": 0.0,
    "cloudflare-flux-1-schnell": 0.0,
    "cloudflare-flux-2-klein-4b": 0.0,
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
                                         # | not_entitled | refused | backend_failure
                                         # | http_client | http_server | parse | unknown
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


_MAGIC = (
    (bytes([0xFF, 0xD8, 0xFF]), "image/jpeg"),
    (bytes([0x89]) + b"PNG", "image/png"),
    (b"RIFF", "image/webp"),
    (b"GIF8", "image/gif"),
)


def sniff_image_type(data: bytes) -> str:
    """Read the format off the bytes.

    Workers AI returns bare base64 with no mime type, and output format is a
    reported datum here, so it is sniffed rather than assumed.
    """
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    return "application/octet-stream"


# Google answers an un-entitled model with 429 and "limit: 0 ... on Free Tier".
# A zero ceiling is an entitlement failure wearing a rate-limit code, and waiting
# does not move it.
_ZERO_QUOTA = re.compile(r"limit:\s*0(?!\d)", re.IGNORECASE)


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
    """A provider declined on content-policy grounds. Never worth retrying."""


class BackendFailure(Exception):
    """The request was accepted and the provider inference backend failed.

    Transient on its merits: the request is well-formed, nothing about it needs
    changing, and the same request may well succeed. Worth retrying.
    """


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
        if isinstance(exc, BackendFailure):
            return "backend_failure", str(exc)
        # TimeoutException is itself a TransportError, so it must be tested first.
        if isinstance(exc, httpx.TimeoutException):
            return "timeout", str(exc) or "request timed out"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                body = exc.response.text[:400]
                if _ZERO_QUOTA.search(body):
                    return "not_entitled", f"HTTP 429: {body}"
                return "rate_limit", f"HTTP 429: {body}"
            if code == 402:
                return "payment_required", f"HTTP 402: {_payment_detail(exc.response)}"
            kind = "http_server" if code >= 500 else "http_client"
            return kind, f"HTTP {code}: {exc.response.text[:400]}"
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


class Pollinations(BaseProvider):
    """Keyless endpoint, kept as a zero-friction baseline."""

    name = "pollinations"
    # Measured: 6 simultaneous requests gave 5x HTTP 402 and one timeout, zero
    # successes; the same prompts sequentially succeed. The gate is on rate.
    max_concurrency = 1

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        # A fresh seed per request does two things. It defeats the year-long
        # immutable cache, so a run measures generation rather than a CDN read, and
        # it is honoured: two independent generations with one seed returned
        # byte-identical images, so recording it makes the request re-issuable.
        #
        # No width or height. The endpoint honours the requested aspect and clamps
        # to roughly 0.59 megapixels, so asking for 1024x1024 yields 768x768 while
        # asking for width alone yields 886x665 -- a worse aspect mismatch against
        # the square Workers AI outputs than the resolution gap it would fix.
        seed = random.randrange(2**31)
        # quote(safe="") not httpx.URL(path=...): that leaves "/" unescaped so a
        # prompt can inject path segments, and raises InvalidURL on "?" or "#".
        resp = await client.get(
            "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt, safe=""),
            params={"nologo": "true", "seed": str(seed)},
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
                "seed": seed,
            },
        )


# Workers AI overloads error code 8007 across unrelated conditions: a blocked
# prompt (HTTP 400) and a failed inference (HTTP 409) both carry it. The message is
# what separates them, so these match on text and ignore the code. One is permanent
# and one is transient, and both would otherwise land in http_client.
def _workers_messages(response: httpx.Response) -> list[str]:
    try:
        errors = response.json().get("errors") or []
    except ValueError:
        return []
    return [error.get("message") or "" for error in errors]


def _workers_refusal(response: httpx.Response) -> str | None:
    for message in _workers_messages(response):
        if "nsfw" in message.lower():
            return message[:300]
    return None


def _workers_backend_failure(response: httpx.Response) -> str | None:
    for message in _workers_messages(response):
        if "prediction failed" in message.lower():
            return message[:300]
    return None


class _WorkersAI(BaseProvider):
    """Cloudflare Workers AI.

    10,000 Neurons per day at no charge and no payment method, shared across every
    model, so all of them sit behind one quota gate. Models disagree on how the
    request is encoded and agree on the response: JSON carrying the image as
    base64, with HTTP 200 plus success=false for routing and model errors.
    """

    model_id = ""
    quota_key = "cloudflare-workers-ai"
    max_concurrency = 2

    def __init__(self) -> None:
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

    def available(self) -> bool:
        return bool(self.account_id and self.api_token)

    def request_kwargs(self, prompt: str) -> dict:
        raise NotImplementedError

    def request_meta(self) -> dict:
        return {}

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> GeneratedImage:
        resp = await client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model_id}",
            headers={"Authorization": f"Bearer {self.api_token}"},
            **self.request_kwargs(prompt),
        )
        if resp.status_code >= 400:
            refusal = _workers_refusal(resp)
            if refusal:
                raise ContentRefused(refusal)
            backend = _workers_backend_failure(resp)
            if backend:
                raise BackendFailure(backend)
        resp.raise_for_status()
        payload = resp.json()

        # HTTP 200 with success=false would otherwise read as a malformed response.
        if payload.get("success") is False:
            raise ValueError(f"workers ai reported failure: {payload.get('errors')}")

        encoded = (payload.get("result") or {}).get("image")
        if not encoded:
            raise ValueError(f"no image in response: {str(payload)[:200]}")

        data = base64.b64decode(encoded)
        return GeneratedImage(
            data,
            sniff_image_type(data),
            {"model": self.model_id, **self.request_meta()},
        )


class CloudflareFluxSchnell(_WorkersAI):
    """FLUX.1 [schnell]: 4.80 Neurons per 512x512 tile plus 9.60 per step."""

    name = "cloudflare-flux-1-schnell"
    model_id = "@cf/black-forest-labs/flux-1-schnell"
    # Distilled to 4 steps; its own ceiling is 8.
    steps = 4

    def request_kwargs(self, prompt: str) -> dict:
        return {"json": {"prompt": prompt, "steps": self.steps}}

    def request_meta(self) -> dict:
        return {"steps": self.steps}


class CloudflareFluxKlein4b(_WorkersAI):
    """FLUX.2 [klein] 4B: 26.05 Neurons per output 512x512 tile.

    Rejects a JSON body -- its input schema requires a multipart envelope, and
    multipart/form-data with a prompt field is what it accepts. Established by
    probing, because the docs carry no request example for it.
    """

    name = "cloudflare-flux-2-klein-4b"
    model_id = "@cf/black-forest-labs/flux-2-klein-4b"

    def request_kwargs(self, prompt: str) -> dict:
        # files= makes httpx send multipart; (None, value) is a plain field.
        return {"files": {"prompt": (None, prompt)}}


ALL_PROVIDERS: list[type[BaseProvider]] = [
    Pollinations,
    CloudflareFluxSchnell,
    CloudflareFluxKlein4b,
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
