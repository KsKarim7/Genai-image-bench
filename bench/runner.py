"""Async execution of the prompt suite across providers.

Concurrency is capped per provider because the rate gate is per key. Measured: six
simultaneous Pollinations requests returned five HTTP 402s and a timeout and zero
images, where the same prompts issued sequentially all succeeded. Providers are
independent coroutines with their own clients, so the cap is not about one provider
starving another -- there is nothing shared to contend for.

Retries are bounded and backed off. A request that exhausts them is recorded as a
failure with its reason, since failure rate is part of what is being measured.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from .providers import (
    BaseProvider,
    GenerationResult,
    COST_PER_IMAGE,
    load_available,
)

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
MAX_ATTEMPTS = 3
# payment_required is retryable on evidence: the x402 gate is driven by request rate,
# and backoff recovered 3 of 12 outputs in run 20261008-162624. http_client (4xx) and
# refused are permanent, so retrying them only burns quota and delays the failure.
RETRYABLE = {"rate_limit", "payment_required", "timeout", "network", "http_server"}


def load_prompts(config_path: Path, axis: str | None = None) -> tuple[dict, list[dict]]:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    prompts = cfg["prompts"]
    if axis:
        prompts = [p for p in prompts if p["axis"] == axis]
        if not prompts:
            raise SystemExit(f"no prompts found for axis {axis!r}")
    return cfg["axes"], prompts


async def _attempt(
    provider: BaseProvider,
    client: httpx.AsyncClient,
    prompt_spec: dict,
    blind_id: str,
    images_dir: Path,
) -> GenerationResult:
    prompt_id = prompt_spec["id"]
    prompt_text = " ".join(prompt_spec["prompt"].split())

    last_kind, last_detail = "unknown", "no attempt made"
    attempts_made = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_made = attempt
        started = time.perf_counter()
        try:
            image = await provider.generate(client, prompt_text)
            latency = time.perf_counter() - started

            # The blind id IS the filename, with no suffix: a .jpg beside a .png
            # partitions the set by provider. Real format is recorded in meta and
            # the scorer views images through an <img> tag, which sniffs content.
            filename = blind_id
            (images_dir / filename).write_bytes(image.data)

            return GenerationResult(
                provider=provider.name,
                prompt_id=prompt_id,
                blind_id=blind_id,
                ok=True,
                latency_s=round(latency, 3),
                cost_usd=COST_PER_IMAGE.get(provider.name, 0.0),
                image_path=f"images/{filename}",
                meta={
                    "attempts": attempt,
                    "bytes": len(image.data),
                    "content_type": image.content_type.split(";")[0].strip(),
                    # Nested, not merged: an adapter key cannot shadow one of ours.
                    "provider_meta": image.meta,
                },
            )
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            latency = time.perf_counter() - started
            last_kind, last_detail = provider.classify_error(exc)

            if last_kind not in RETRYABLE or attempt == MAX_ATTEMPTS:
                break

            # Exponential backoff with jitter. Jitter matters because every provider
            # starts its suite at the same instant; without it the retries collide
            # on the same schedule and we re-trigger the rate limit we backed off from.
            # A server that says when to come back beats guessing with our own curve.
            hinted = provider.retry_after_seconds(exc)
            delay = hinted if hinted is not None else (2 ** attempt) + random.uniform(0, 1.5)
            print(f"    {provider.name}/{prompt_id}: {last_kind}, retry in {delay:.1f}s")
            await asyncio.sleep(delay)

    return GenerationResult(
        provider=provider.name,
        prompt_id=prompt_id,
        blind_id=blind_id,
        ok=False,
        latency_s=round(latency, 3),
        error_kind=last_kind,
        error_detail=last_detail,
        meta={"attempts": attempts_made},
    )


async def _run_provider(
    provider: BaseProvider,
    prompts: list[dict],
    images_dir: Path,
) -> list[GenerationResult]:
    # The slot is deliberately held across the backoff sleep. When the limit is a
    # rate gate, not issuing the next request is the entire point; releasing it
    # here would let a sibling prompt fire into the gate we just backed off from.
    sem = asyncio.Semaphore(provider.max_concurrency)
    results: list[GenerationResult] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:

        async def one(spec: dict) -> None:
            async with sem:
                blind_id = uuid.uuid4().hex[:10]
                res = await _attempt(provider, client, spec, blind_id, images_dir)
                status = "ok" if res.ok else f"FAIL({res.error_kind})"
                print(f"  {provider.name:<20} {spec['id']:<8}{res.latency_s:>6.2f}s  {status}")
                results.append(res)

        await asyncio.gather(*(one(spec) for spec in prompts))

    return results


def _group_key(spec: dict) -> str | None:
    return spec.get("consistency_group") or spec.get("style_group")


def build_units(prompts: list[dict], results: list[GenerationResult]) -> list[dict]:
    """Group successful outputs into scoring units.

    Prompts sharing a consistency_group or style_group are one unit, judged as a
    set: "same individual as cc_01" cannot be answered one shuffled image at a
    time. A unit never spans providers, so presenting the set keeps it blind.
    Members are ordered as the suite declares them, so the reference comes first.
    """
    specs = {p["id"]: p for p in prompts}
    order = {p["id"]: i for i, p in enumerate(prompts)}
    units: dict[tuple, dict] = {}

    for res in results:
        if not res.ok:
            continue
        spec = specs[res.prompt_id]
        group = _group_key(spec)
        key = (res.provider, group) if group else (res.provider, res.prompt_id)
        unit = units.setdefault(
            key,
            {
                "unit_id": uuid.uuid4().hex[:10],
                "provider": res.provider,
                "axis": spec["axis"],
                "group": group,
                "members": [],
            },
        )
        unit["members"].append(res)

    for unit in units.values():
        unit["members"].sort(key=lambda r: order[r.prompt_id])
    return list(units.values())


def build_blind_map(units: list[dict]) -> dict[str, dict]:
    """The only artifact that knows which provider produced what.

    images maps an image id to its provider; units maps a scoring unit to the
    provider and the image ids it covers. Nothing the scorer reads holds either.
    """
    return {
        "images": {
            res.blind_id: res.to_blind_entry() for u in units for res in u["members"]
        },
        "units": {
            u["unit_id"]: {
                "provider": u["provider"],
                "axis": u["axis"],
                "group": u["group"],
                "image_ids": [res.blind_id for res in u["members"]],
            }
            for u in units
        },
    }


def build_scoring_manifest(
    run_id: str, axes: dict, prompts: list[dict], units: list[dict]
) -> dict:
    """The scorer's only input.

    Carries no provider and nothing that joins to one: no bytes, no latency, no
    content type, no prompt id. One entry per scoring decision.
    """
    specs = {p["id"]: p for p in prompts}
    return {
        "run_id": run_id,
        "units": [
            {
                "unit_id": u["unit_id"],
                "axis": u["axis"],
                "scale": axes[u["axis"]]["scale"],
                "images": [res.image_path for res in u["members"]],
                "prompts": [
                    {
                        "prompt": " ".join(specs[res.prompt_id]["prompt"].split()),
                        "checks": specs[res.prompt_id]["checks"],
                    }
                    for res in u["members"]
                ],
            }
            for u in units
        ],
    }


async def run(config_path: Path, runs_dir: Path, axis: str | None = None) -> str:
    axes, prompts = load_prompts(config_path, axis)

    providers = load_available()
    if not providers:
        raise SystemExit(
            "No providers available. Pollinations needs no key, so if you are seeing "
            "this, check that bench/providers.py loaded correctly."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / run_id
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nrun {run_id}")
    print(f"  providers: {', '.join(p.name for p in providers)}")
    print(f"  prompts:   {len(prompts)}")

    all_results: list[GenerationResult] = []
    # Providers run concurrently with each other; each polices its own rate limit.
    gathered = await asyncio.gather(
        *(_run_provider(p, prompts, images_dir) for p in providers)
    )
    for batch in gathered:
        all_results.extend(batch)

    units = build_units(prompts, all_results)
    blind_map = build_blind_map(units)

    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "axes": axes,
                "prompts": prompts,
                "results": [r.to_results_row() for r in all_results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "blind_map.json").write_text(
        json.dumps(blind_map, indent=2), encoding="utf-8"
    )
    (run_dir / "scoring_manifest.json").write_text(
        json.dumps(build_scoring_manifest(run_id, axes, prompts, units), indent=2),
        encoding="utf-8",
    )

    ok = sum(1 for r in all_results if r.ok)
    sets = sum(1 for u in units if len(u["members"]) > 1)
    print(f"\n  {ok}/{len(all_results)} succeeded")
    print(f"  {len(units)} scoring unit(s)" + (f", {sets} scored as sets" if sets else ""))
    print(f"  written to {run_dir}")
    print(f"\n  next: python run.py score {run_id}")
    return run_id
