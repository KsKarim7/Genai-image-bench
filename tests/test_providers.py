"""Adapter request and response handling, against mocked transports.

These assert the shape each adapter sends and what it does with the shapes it can
get back. They are not a substitute for a real call: an adapter is only trusted
once it has generated against the live API. They exist so that the shape, once
established, stays established.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import unittest

import httpx

from bench.providers import (
    CloudflareFluxSchnell,
    ContentRefused,
    GeminiFlashImage,
    GeminiFlashLiteImage,
    sniff_image_type,
)

JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"payload"
PNG = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]) + b"payload"


def _run(provider, handler, prompt="a plain grey square"):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await provider.generate(client, prompt)
    return asyncio.run(go())


class SniffImageType(unittest.TestCase):
    def test_known_formats(self):
        self.assertEqual(sniff_image_type(JPEG), "image/jpeg")
        self.assertEqual(sniff_image_type(PNG), "image/png")
        self.assertEqual(sniff_image_type(b"RIFF....WEBP"), "image/webp")
        self.assertEqual(sniff_image_type(b"GIF89a"), "image/gif")

    def test_unknown_is_not_claimed_to_be_an_image(self):
        self.assertEqual(sniff_image_type(b"not an image"), "application/octet-stream")


class CloudflareAdapter(unittest.TestCase):
    def setUp(self):
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acct123"
        os.environ["CLOUDFLARE_API_TOKEN"] = "tok456"
        self.provider = CloudflareFluxSchnell()
        self.seen = {}

    def _ok_handler(self, data=JPEG):
        def handler(request: httpx.Request) -> httpx.Response:
            self.seen["url"] = str(request.url)
            self.seen["auth"] = request.headers.get("authorization")
            self.seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "result": {"image": base64.b64encode(data).decode()},
                "success": True, "errors": [], "messages": [],
            })
        return handler

    def test_available_requires_both_credentials(self):
        self.assertTrue(self.provider.available())
        os.environ["CLOUDFLARE_API_TOKEN"] = ""
        self.assertFalse(CloudflareFluxSchnell().available())
        os.environ["CLOUDFLARE_API_TOKEN"] = "tok456"
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""
        self.assertFalse(CloudflareFluxSchnell().available())

    def test_request_shape(self):
        _run(self.provider, self._ok_handler())
        self.assertEqual(
            self.seen["url"],
            "https://api.cloudflare.com/client/v4/accounts/acct123"
            "/ai/run/@cf/black-forest-labs/flux-1-schnell",
        )
        self.assertEqual(self.seen["auth"], "Bearer tok456")
        self.assertEqual(self.seen["body"], {"prompt": "a plain grey square", "steps": 4})

    def test_decodes_base64_and_sniffs_the_format(self):
        image = _run(self.provider, self._ok_handler(JPEG))
        self.assertEqual(image.data, JPEG)
        self.assertEqual(image.content_type, "image/jpeg")
        self.assertEqual(image.meta["steps"], 4)
        self.assertEqual(image.meta["model"], "@cf/black-forest-labs/flux-1-schnell")

    def test_format_is_read_from_bytes_not_assumed(self):
        image = _run(self.provider, self._ok_handler(PNG))
        self.assertEqual(image.content_type, "image/png")

    def test_http_200_with_success_false_is_not_treated_as_an_image(self):
        def handler(request):
            return httpx.Response(200, json={
                "success": False,
                "errors": [{"code": 7003, "message": "Could not route to /ai/run"}],
                "result": None,
            })
        with self.assertRaises(ValueError) as caught:
            _run(self.provider, handler)
        self.assertIn("7003", str(caught.exception))
        self.assertEqual(self.provider.classify_error(caught.exception)[0], "parse")

    def test_missing_image_field_raises_with_the_body(self):
        def handler(request):
            return httpx.Response(200, json={"result": {}, "success": True})
        with self.assertRaises(ValueError):
            _run(self.provider, handler)

    def test_rate_limit_is_retryable_and_bad_token_is_not(self):
        from bench.runner import RETRYABLE
        for status, expected in ((429, "rate_limit"), (401, "http_client"),
                                 (500, "http_server")):
            def handler(request, status=status):
                return httpx.Response(status, json={"success": False, "errors": []})
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                _run(self.provider, handler)
            kind, _ = self.provider.classify_error(caught.exception)
            self.assertEqual(kind, expected, f"HTTP {status}")
        self.assertIn("rate_limit", RETRYABLE)
        self.assertNotIn("http_client", RETRYABLE)


class GeminiAdapter(unittest.TestCase):
    def setUp(self):
        os.environ["GOOGLE_API_KEY"] = "gkey"
        self.seen = {}

    def _handler(self, data=PNG, mime="image/png"):
        def handler(request: httpx.Request) -> httpx.Response:
            self.seen["url"] = str(request.url)
            self.seen["key"] = request.headers.get("x-goog-api-key")
            self.seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"interaction": {
                "id": "ix_1", "model": self.seen["body"]["model"],
                "outputImage": {"data": base64.b64encode(data).decode(), "mimeType": mime},
            }})
        return handler

    def test_both_models_post_to_the_interactions_endpoint(self):
        for cls, model in ((GeminiFlashLiteImage, "gemini-3.1-flash-lite-image"),
                           (GeminiFlashImage, "gemini-3.1-flash-image")):
            image = _run(cls(), self._handler())
            self.assertEqual(
                self.seen["url"],
                "https://generativelanguage.googleapis.com/v1beta/interactions",
            )
            self.assertEqual(self.seen["key"], "gkey")
            self.assertEqual(self.seen["body"]["model"], model)
            self.assertEqual(self.seen["body"]["input"],
                             [{"type": "text", "text": "a plain grey square"}])
            self.assertEqual(self.seen["body"]["response_format"],
                             {"type": "image", "aspect_ratio": "1:1", "image_size": "1K"})
            self.assertEqual(image.meta["model_version"], model)
            self.assertEqual(image.meta["image_size"], "1K")

    def test_both_models_share_one_quota_gate(self):
        self.assertEqual(GeminiFlashLiteImage().gate(), GeminiFlashImage().gate())

    def test_snake_case_response_is_accepted(self):
        def handler(request):
            return httpx.Response(200, json={"interaction": {
                "output_image": {"data": base64.b64encode(JPEG).decode(),
                                 "mime_type": "image/jpeg"}}})
        image = _run(GeminiFlashLiteImage(), handler)
        self.assertEqual(image.content_type, "image/jpeg")

    def test_steps_fallback_is_accepted(self):
        def handler(request):
            return httpx.Response(200, json={"interaction": {"steps": [
                {"type": "model_output",
                 "content": [{"type": "image", "data": base64.b64encode(PNG).decode()}]}]}})
        self.assertEqual(_run(GeminiFlashLiteImage(), handler).data, PNG)

    def test_recognised_refusal_becomes_refused_and_is_not_retried(self):
        from bench.runner import RETRYABLE
        def handler(request):
            return httpx.Response(200, json={
                "interaction": {"finishReason": "PROHIBITED_CONTENT"}})
        with self.assertRaises(ContentRefused) as caught:
            _run(GeminiFlashLiteImage(), handler)
        self.assertEqual(
            GeminiFlashLiteImage.classify_error(caught.exception)[0], "refused")
        self.assertNotIn("refused", RETRYABLE)

    def test_unrecognised_response_is_not_guessed_into_refused(self):
        def handler(request):
            return httpx.Response(200, json={"interaction": {"id": "ix_2"}})
        with self.assertRaises(ValueError) as caught:
            _run(GeminiFlashLiteImage(), handler)
        self.assertEqual(
            GeminiFlashLiteImage.classify_error(caught.exception)[0], "parse")


if __name__ == "__main__":
    unittest.main()
