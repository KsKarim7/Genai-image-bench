"""Async execution of the prompt suite across providers.

Concurrency is capped per provider because the rate gate is per key: six simultaneous
Pollinations requests returned five 402s, a timeout and no images, where the same
prompts issued sequentially all succeeded.

A request that exhausts its bounded retries is recorded as a failure with its reason,
since failure rate is part of what is being measured.
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
    LIST_PRICE_USD_PER_IMAGE,
    load_available,
)

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
MAX_ATTEMPTS = 3
# payment_required is retryable on evidence: the x402 gate is rate-driven and backoff
# recovered 3 of 12 outputs in run 20261008-162624. 4xx and refusals are permanent.
RETRYABLE = {"rate_limit", "payment_required", "timeout", "network", "http_server"}


def load_prompts(config_path: Path, axis: str | None = None) -> tuple[dict, list[dict]]:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    axes, prompts = cfg["axes"], cfg["prompts"]
    groups = cfg.get("groups") or {}

    # Fail here, not part-way through a scoring session.
    missing = [name for name, spec in axes.items() if not spec.get("anchors")]
    if missing:
        raise SystemExit(f"axes with no scorer anchors: {', '.join(missing)}")
    undeclared = sorted({p["axis"] for p in prompts} - set(axes))
    if undeclared:
        raise SystemExit(f"prompts name undeclared axes: {', '.join(undeclared)}")

    used = {g for p in prompts if (g := _group_key(p))}
    unknown = sorted(used - set(groups))
    if unknown:
        raise SystemExit(f"prompts name undeclared groups: {', '.join(unknown)}")
    for name in sorted(used):
        if not groups[name].get("checks"):
            raise SystemExit(f"group {name!r} declares no checks")
        declared = groups[name].get("axis")
        members = {p["axis"] for p in prompts if _group_key(p) == name}
        if declared and members != {declared}:
            raise SystemExit(
                f"group {name!r} declares axis {declared!r} but its prompts use {sorted(members)}"
            )

    if axis:
        prompts = [p for p in prompts if p["axis"] == axis]
        if not prompts:
            raise SystemExit(f"no prompts found for axis {axis!r}")
    return {"axes": axes, "groups": groups}, prompts


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

            # The blind id is the whole filename: a .jpg beside a .png partitions
            # the set by provider. Real format goes in meta.
            filename = blind_id
            (images_dir / filename).write_bytes(image.data)

            return GenerationResult(
                provider=provider.name,
                prompt_id=prompt_id,
                blind_id=blind_id,
                ok=True,
                latency_s=round(latency, 3),
                cost_usd=LIST_PRICE_USD_PER_IMAGE.get(provider.name, 0.0),
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


def build_gates(providers: list[BaseProvider]) -> dict[str, asyncio.Semaphore]:
    """One semaphore per credential, not per provider.

    The rate gate is per key, so two adapters on one key have to share a cap rather
    than get one each. Where they disagree on max_concurrency the lower wins.
    """
    limits: dict[str, int] = {}
    for provider in providers:
        key = provider.gate()
        limits[key] = min(limits.get(key, provider.max_concurrency), provider.max_concurrency)
    return {key: asyncio.Semaphore(limit) for key, limit in limits.items()}


async def _run_provider(
    provider: BaseProvider,
    prompts: list[dict],
    images_dir: Path,
    sem: asyncio.Semaphore,
) -> list[GenerationResult]:
    # The slot is held across the backoff sleep on purpose: when the limit is a rate
    # gate, not issuing the next request is the point. Do not "fix" this to release.
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


def _shared_gates(providers: list[BaseProvider]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for provider in providers:
        grouped.setdefault(provider.gate(), []).append(provider.name)
    return {key: names for key, names in grouped.items() if len(names) > 1}


def _group_key(spec: dict) -> str | None:
    return spec.get("consistency_group") or spec.get("style_group")


def build_units(prompts: list[dict], results: list[GenerationResult]) -> list[dict]:
    """Group successful outputs into scoring units.

    Prompts sharing a consistency_group or style_group are one unit, judged as a set:
    whether three images show one individual cannot be answered one shuffled image at
    a time. A unit never spans providers, so presenting the set keeps it blind.
    Members follow suite order, so a consistency group reference comes first.
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
    run_id: str, config: dict, prompts: list[dict], units: list[dict]
) -> dict:
    """The scorer's only input.

    Carries no provider and nothing that joins to one: no bytes, no latency, no
    content type, no prompt id. One entry per scoring decision, with set-level
    criteria stated once rather than repeated under each member.
    """
    axes, groups = config["axes"], config["groups"]
    specs = {p["id"]: p for p in prompts}
    return {
        "run_id": run_id,
        "units": [
            {
                "unit_id": u["unit_id"],
                "axis": u["axis"],
                "anchors": axes[u["axis"]]["anchors"],
                "set_checks": (groups.get(u["group"]) or {}).get("checks", []),
                "images": [res.image_path for res in u["members"]],
                "prompts": [
                    {
                        "prompt": " ".join(specs[res.prompt_id]["prompt"].split()),
                        "checks": specs[res.prompt_id].get("checks") or [],
                    }
                    for res in u["members"]
                ],
            }
            for u in units
        ],
    }


async def run(config_path: Path, runs_dir: Path, axis: str | None = None) -> str:
    config, prompts = load_prompts(config_path, axis)

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
    for key, members in _shared_gates(providers).items():
        print(f"  shared rate gate {key}: {', '.join(members)}")
    print(f"  prompts:   {len(prompts)}")

    all_results: list[GenerationResult] = []
    gates = build_gates(providers)
    gathered = await asyncio.gather(
        *(_run_provider(p, prompts, images_dir, gates[p.gate()]) for p in providers)
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
                "axes": config["axes"],
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
        json.dumps(build_scoring_manifest(run_id, config, prompts, units), indent=2),
        encoding="utf-8",
    )

    ok = sum(1 for r in all_results if r.ok)
    sets = sum(1 for u in units if len(u["members"]) > 1)
    print(f"\n  {ok}/{len(all_results)} succeeded")
    print(f"  {len(units)} scoring unit(s)" + (f", {sets} scored as sets" if sets else ""))
    print(f"  written to {run_dir}")
    print(f"\n  next: python run.py score {run_id}")
    return run_id
