"""Async execution of the prompt suite across providers.

Concurrency is capped per provider rather than globally, because free tiers rate-limit
per key and one aggressive provider would otherwise starve the others. Retries use
exponential backoff and are bounded; a request that exhausts its retries is recorded
as a failure with its reason rather than dropped, since failure rate is part of what
this benchmark is measuring.
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

from .providers import BaseProvider, GenerationResult, COST_PER_IMAGE, load_available

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
MAX_ATTEMPTS = 3
RETRYABLE = {"rate_limit", "timeout", "http"}


def load_prompts(config_path: Path, axis: str | None = None) -> tuple[dict, list[dict]]:
    with config_path.open() as fh:
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
    repeat: int,
    images_dir: Path,
) -> GenerationResult:
    prompt_id = prompt_spec["id"]
    prompt_text = " ".join(prompt_spec["prompt"].split())

    last_kind, last_detail = "unknown", "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            data = await provider.generate(client, prompt_text)
            latency = time.perf_counter() - started

            filename = f"{provider.name}__{prompt_id}__r{repeat}.png"
            (images_dir / filename).write_bytes(data)

            return GenerationResult(
                provider=provider.name,
                prompt_id=prompt_id,
                repeat=repeat,
                ok=True,
                latency_s=round(latency, 3),
                cost_usd=COST_PER_IMAGE.get(provider.name, 0.0),
                image_path=f"images/{filename}",
                meta={"attempts": attempt, "bytes": len(data)},
            )
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            latency = time.perf_counter() - started
            last_kind, last_detail = provider.classify_error(exc)

            if last_kind not in RETRYABLE or attempt == MAX_ATTEMPTS:
                break

            # Exponential backoff with jitter. Jitter matters because every provider
            # starts its suite at the same instant; without it the retries collide
            # on the same schedule and we re-trigger the rate limit we backed off from.
            delay = (2 ** attempt) + random.uniform(0, 1.5)
            print(f"    {provider.name}/{prompt_id} r{repeat}: {last_kind}, retry in {delay:.1f}s")
            await asyncio.sleep(delay)

    return GenerationResult(
        provider=provider.name,
        prompt_id=prompt_id,
        repeat=repeat,
        ok=False,
        latency_s=round(latency, 3),
        error_kind=last_kind,
        error_detail=last_detail,
        meta={"attempts": MAX_ATTEMPTS if last_kind in RETRYABLE else 1},
    )


async def _run_provider(
    provider: BaseProvider,
    prompts: list[dict],
    repeats: int,
    images_dir: Path,
) -> list[GenerationResult]:
    sem = asyncio.Semaphore(provider.max_concurrency)
    results: list[GenerationResult] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:

        async def one(spec: dict, repeat: int) -> None:
            async with sem:
                res = await _attempt(provider, client, spec, repeat, images_dir)
                status = "ok" if res.ok else f"FAIL({res.error_kind})"
                print(f"  {provider.name:<20} {spec['id']} r{repeat}  {res.latency_s:>6.2f}s  {status}")
                results.append(res)

        await asyncio.gather(
            *(one(spec, r) for spec in prompts for r in range(1, repeats + 1))
        )

    return results


def build_blind_map(results: list[GenerationResult]) -> dict[str, dict]:
    """Assign an opaque id to every successful output.

    The scorer is handed only these ids. The mapping back to provider names is written
    to a separate file that the scoring step never reads, so the person scoring cannot
    see which model produced what even if they are the same person who ran the
    benchmark. This is the one design decision the whole project rests on.
    """
    blind: dict[str, dict] = {}
    for res in results:
        if not res.ok:
            continue
        blind_id = uuid.uuid4().hex[:10]
        blind[blind_id] = {
            "provider": res.provider,
            "prompt_id": res.prompt_id,
            "repeat": res.repeat,
            "image_path": res.image_path,
        }
    return blind


async def run(
    config_path: Path,
    runs_dir: Path,
    axis: str | None = None,
    repeats: int = 1,
) -> str:
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
    print(f"  prompts:   {len(prompts)}  x {repeats} repeat(s)\n")

    all_results: list[GenerationResult] = []
    # Providers run concurrently with each other; each polices its own rate limit.
    gathered = await asyncio.gather(
        *(_run_provider(p, prompts, repeats, images_dir) for p in providers)
    )
    for batch in gathered:
        all_results.extend(batch)

    blind_map = build_blind_map(all_results)

    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "axes": axes,
                "prompts": prompts,
                "repeats": repeats,
                "results": [r.to_dict() for r in all_results],
            },
            indent=2,
        )
    )
    (run_dir / "blind_map.json").write_text(json.dumps(blind_map, indent=2))

    ok = sum(1 for r in all_results if r.ok)
    print(f"\n  {ok}/{len(all_results)} succeeded")
    print(f"  written to {run_dir}")
    print(f"\n  next: python run.py score {run_id}")
    return run_id
