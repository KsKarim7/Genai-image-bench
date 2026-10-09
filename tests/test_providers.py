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
    BaseProvider,
    CloudflareFluxKlein4b,
    CloudflareFluxSchnell,
    ContentRefused,
    Pollinations,
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


class ErrorTaxonomy(unittest.TestCase):
    """The kinds that are not reachable from any current adapter still have to
    behave, because the runner decides retries from them."""

    @staticmethod
    def _status(code, body=""):
        response = httpx.Response(code, text=body,
                                  request=httpx.Request("GET", "https://example.test"))
        return httpx.HTTPStatusError("e", request=response.request, response=response)

    def test_zero_quota_429_is_not_a_rate_limit(self):
        from bench.runner import RETRYABLE
        body = ('{"error":{"message":"Rate limit exceeded for model x '
                '(limit: 0 requests per day on Free Tier)."}}')
        kind, detail = BaseProvider.classify_error(self._status(429, body))
        self.assertEqual(kind, "not_entitled")
        self.assertNotIn(kind, RETRYABLE)
        self.assertIn("limit: 0", detail)

    def test_zero_quota_on_a_token_metric_too(self):
        body = '{"error":{"message":"... (limit: 0 input tokens per minute on Free Tier)."}}'
        self.assertEqual(BaseProvider.classify_error(self._status(429, body))[0],
                         "not_entitled")

    def test_a_real_ceiling_stays_retryable(self):
        from bench.runner import RETRYABLE
        body = '{"error":{"message":"... (limit: 1500 requests per day)."}}'
        kind, _ = BaseProvider.classify_error(self._status(429, body))
        self.assertEqual(kind, "rate_limit")
        self.assertIn(kind, RETRYABLE)

    def test_a_zero_prefixed_ceiling_is_not_mistaken_for_zero(self):
        body = '{"error":{"message":"... (limit: 05 requests per day)."}}'
        self.assertEqual(BaseProvider.classify_error(self._status(429, body))[0],
                         "rate_limit")

    def test_a_429_with_no_limit_text_stays_a_rate_limit(self):
        self.assertEqual(BaseProvider.classify_error(self._status(429, "slow down"))[0],
                         "rate_limit")

    def test_content_refused_maps_to_refused_and_is_never_retried(self):
        from bench.runner import RETRYABLE
        kind, detail = BaseProvider.classify_error(ContentRefused("blockReason=SAFETY"))
        self.assertEqual(kind, "refused")
        self.assertNotIn(kind, RETRYABLE)
        self.assertIn("SAFETY", detail)

    def test_error_bodies_are_kept_long_enough_to_diagnose(self):
        body = "x" * 600
        _, detail = BaseProvider.classify_error(self._status(400, body))
        self.assertGreater(len(detail), 300)


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


class PollinationsSeeding(unittest.TestCase):
    """The seed defeats a year-long immutable cache, so a run measures generation
    rather than a CDN read, and it is honoured, so recording it makes the request
    re-issuable. No width or height: the endpoint honours the requested aspect and
    clamps to about 0.59 MP, so asking for a square 1024 gives 768 and asking for
    width alone gives 886x665."""

    def setUp(self):
        self.provider = Pollinations()
        self.seen = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(dict(request.url.params))
        return httpx.Response(200, content=JPEG, headers={
            "content-type": "image/jpeg", "x-cache": "MISS", "x-model-used": "sana"})

    def test_every_request_carries_a_seed(self):
        image = _run(self.provider, self._handler)
        self.assertIn("seed", self.seen[0])
        self.assertTrue(self.seen[0]["seed"].isdigit())
        self.assertEqual(str(image.meta["seed"]), self.seen[0]["seed"])

    def test_the_seed_changes_between_requests(self):
        _run(self.provider, self._handler)
        _run(self.provider, self._handler)
        self.assertNotEqual(self.seen[0]["seed"], self.seen[1]["seed"])

    def test_no_size_parameters_are_sent(self):
        _run(self.provider, self._handler)
        self.assertNotIn("width", self.seen[0])
        self.assertNotIn("height", self.seen[0])

    def test_cache_status_and_model_still_recorded(self):
        image = _run(self.provider, self._handler)
        self.assertEqual(image.meta["x_cache"], "MISS")
        self.assertEqual(image.meta["model"], "sana")

    def test_prompt_is_percent_encoded_into_the_path(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, content=JPEG,
                                  headers={"content-type": "image/jpeg"})

        _run(self.provider, handler, "a/b slash and ? and #")
        self.assertNotIn("/b", seen["path"].split("/prompt/", 1)[1])
        self.assertIn("%2F", seen["path"])


class WorkersAiContentRefusal(unittest.TestCase):
    """flux-1-schnell refused a benign character-sheet prompt as NSFW in run
    20261009-053831. It arrived as HTTP 400 and classified as http_client, which
    loses the distinction between a malformed request and a policy refusal."""

    REAL_BODY = {
        "errors": [{
            "message": "AiError: AiError: Input prompt contains NSFW content. "
                       "(bd40edba-ba52-4d2d-8952-af3471313f4e)",
            "code": 8007,
        }],
        "success": False, "result": {}, "messages": [],
    }

    def setUp(self):
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acct123"
        os.environ["CLOUDFLARE_API_TOKEN"] = "tok456"
        self.provider = CloudflareFluxSchnell()

    def _call(self, payload, status):
        def handler(request):
            return httpx.Response(status, json=payload)
        return _run(self.provider, handler)

    def test_the_real_nsfw_body_becomes_a_refusal(self):
        from bench.runner import RETRYABLE
        with self.assertRaises(ContentRefused) as caught:
            self._call(self.REAL_BODY, 400)
        kind, detail = self.provider.classify_error(caught.exception)
        self.assertEqual(kind, "refused")
        self.assertNotIn(kind, RETRYABLE)
        self.assertIn("NSFW", detail)

    def test_an_nsfw_message_without_the_code_is_still_a_refusal(self):
        body = {"errors": [{"message": "blocked: nsfw content detected", "code": 9999}],
                "success": False}
        with self.assertRaises(ContentRefused):
            self._call(body, 400)

    def test_a_non_refusal_400_stays_http_client(self):
        body = {"errors": [{"message": "AiError: Bad input: required properties at "
                                       "'/' are 'multipart'", "code": 5006}],
                "success": False}
        with self.assertRaises(httpx.HTTPStatusError) as caught:
            self._call(body, 400)
        self.assertEqual(self.provider.classify_error(caught.exception)[0], "http_client")

    def test_a_400_with_an_unparseable_body_stays_http_client(self):
        def handler(request):
            return httpx.Response(400, text="<html>gateway</html>")
        with self.assertRaises(httpx.HTTPStatusError) as caught:
            _run(self.provider, handler)
        self.assertEqual(self.provider.classify_error(caught.exception)[0], "http_client")

    def test_other_statuses_are_untouched_by_the_refusal_check(self):
        for status, expected in ((401, "http_client"), (429, "rate_limit"),
                                 (500, "http_server")):
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                self._call({"errors": []}, status)
            self.assertEqual(
                self.provider.classify_error(caught.exception)[0], expected, status)


class KleinSendsMultipart(unittest.TestCase):
    """klein-4b rejects a JSON body with "required properties at '/' are
    'multipart'". The encoding was established by probing the live API, so it is
    pinned here."""

    def setUp(self):
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acct123"
        os.environ["CLOUDFLARE_API_TOKEN"] = "tok456"
        self.provider = CloudflareFluxKlein4b()
        self.seen = {}

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.seen["content_type"] = request.headers.get("content-type", "")
        self.seen["raw"] = request.content
        self.seen["url"] = str(request.url)
        return httpx.Response(200, json={
            "result": {"image": base64.b64encode(JPEG).decode()},
            "success": True, "errors": [], "messages": [],
        })

    def test_request_is_multipart_form_data_with_a_prompt_field(self):
        image = _run(self.provider, self._handler, "a plain grey square")
        self.assertTrue(self.seen["content_type"].startswith("multipart/form-data"))
        self.assertIn(b'name="prompt"', self.seen["raw"])
        self.assertIn(b"a plain grey square", self.seen["raw"])
        self.assertNotIn(b'{"prompt"', self.seen["raw"])
        self.assertEqual(image.data, JPEG)

    def test_endpoint_names_the_klein_model(self):
        _run(self.provider, self._handler)
        self.assertTrue(self.seen["url"].endswith(
            "/ai/run/@cf/black-forest-labs/flux-2-klein-4b"))

    def test_meta_records_the_model_and_no_step_count(self):
        image = _run(self.provider, self._handler)
        self.assertEqual(image.meta["model"], "@cf/black-forest-labs/flux-2-klein-4b")
        self.assertNotIn("steps", image.meta)

    def test_schnell_still_sends_json_not_multipart(self):
        provider = CloudflareFluxSchnell()
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["content_type"] = request.headers.get("content-type", "")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "result": {"image": base64.b64encode(JPEG).decode()}, "success": True})

        _run(provider, handler, "a cat")
        self.assertEqual(seen["content_type"], "application/json")
        self.assertEqual(seen["body"], {"prompt": "a cat", "steps": 4})

    def test_both_workers_models_share_one_quota_gate(self):
        self.assertEqual(CloudflareFluxSchnell().gate(), CloudflareFluxKlein4b().gate())
        self.assertEqual(CloudflareFluxKlein4b().gate(), "cloudflare-workers-ai")


if __name__ == "__main__":
    unittest.main()
