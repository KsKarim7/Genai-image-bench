"""Unblinding and report generation.

This is the only module that joins scores back to provider identity. It emits a
self-contained HTML file: a summary table plus a prompt-by-provider image grid.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _p90(values: list[float]) -> float | None:
    """Linear interpolation. The previous nearest-rank floor, int(0.9*(n-1)),
    returned the minimum at n=2 and only reached a true p90 at n>=11, so it
    printed a p90 below the median."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = 0.9 * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (pos - low) * (ordered[high] - ordered[low]), 2)


def _is_degenerate(unit: dict) -> bool:
    """A grouped axis scored from one image.

    Derived rather than read from the artifact, so runs recorded before the flag
    existed are handled the same way. Such a score was given against set criteria
    that mostly could not be answered, so it is not comparable to a real set score
    and does not enter an axis mean.
    """
    return bool(unit.get("group")) and len(unit["image_ids"]) < 2


def _cache_state(row: dict) -> str | None:
    state = (row["meta"].get("provider_meta") or {}).get("x_cache")
    return state.upper() if isinstance(state, str) else None


def _overall_cell(o: dict, total_axes: int) -> str:
    """An overall over two axes is not comparable to one over four, so the coverage
    travels with the number instead of being left for the reader to infer."""
    score = o["overall_score"]
    if score is None:
        return "—"
    scored = o.get("axes_scored", total_axes)
    if scored < total_axes:
        return f"{score} <span class=\"qual\">{scored}/{total_axes} axes</span>"
    return str(score)


def _cost_cell(o: dict) -> str:
    if not o["succeeded"]:
        return "—"
    # Zero is the measured cost of a free tier, not a missing figure.
    return "free tier" if not o["est_cost_usd"] else f"${o['est_cost_usd']:.4f}"


def _cache_cell(o: dict) -> str:
    c = o["cache"]
    return f"{c['hit']} hit / {c['miss']} miss" if c else "—"


def build_report(run_dir: Path) -> Path:
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    blind_map = json.loads((run_dir / "blind_map.json").read_text(encoding="utf-8"))

    scores_path = run_dir / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
    if not scores:
        print("  note: no scores.json — report will show operational metrics only")

    results = payload["results"]
    prompts = payload["prompts"]
    # results.json carries no image_path; the blind map is the only join.
    images = blind_map["images"]
    image_paths = {
        (rec["provider"], rec["prompt_id"]): rec["image_path"]
        for rec in images.values()
    }
    prompts_by_id = {p["id"]: p for p in prompts}
    providers = sorted({r["provider"] for r in results})

    # --- operational metrics ------------------------------------------------
    ops: dict[str, dict] = {}
    for provider in providers:
        rows = [r for r in results if r["provider"] == provider]
        ok_rows = [r for r in rows if r["ok"]]
        failure_kinds: dict[str, int] = defaultdict(int)
        for r in rows:
            if not r["ok"]:
                failure_kinds[r["error_kind"] or "unknown"] += 1

        states = [_cache_state(r) for r in ok_rows]
        reports_cache = any(st is not None for st in states)
        hit_lat = [r["latency_s"] for r, st in zip(ok_rows, states) if st == "HIT"]
        miss_lat = [r["latency_s"] for r, st in zip(ok_rows, states) if st == "MISS"]
        # Where a provider reports cache status, only the misses measure it: a hit
        # is a CDN read. Where it reports none, every success counts as generated.
        gen_lat = miss_lat if reports_cache else [r["latency_s"] for r in ok_rows]

        ops[provider] = {
            "attempts": len(rows),
            "succeeded": len(ok_rows),
            "success_rate": round(100 * len(ok_rows) / len(rows), 1) if rows else 0.0,
            # Both rates, because the one a pipeline decision turns on is whether
            # retrying is affordable. Reporting only the post-retry figure would also
            # let a reclassification that enables retries flatter the result.
            "first_attempt_rate": (
                round(100 * sum(1 for r in ok_rows if r["meta"].get("attempts") == 1)
                      / len(rows), 1) if rows else 0.0
            ),
            "succeeded_first_attempt": sum(
                1 for r in ok_rows if r["meta"].get("attempts") == 1
            ),
            "median_generated_latency_s": _median(gen_lat),
            "max_generated_latency_s": round(max(gen_lat), 2) if gen_lat else None,
            "p90_generated_latency_s": _p90(gen_lat),
            "median_cache_read_s": _median(hit_lat),
            "cache": (
                {"hit": len(hit_lat), "miss": len(miss_lat)} if reports_cache else None
            ),
            "est_cost_usd": round(sum(r["cost_usd"] for r in ok_rows), 4),
            "formats": sorted(
                {r["meta"]["content_type"] for r in ok_rows if r["meta"].get("content_type")}
            ),
            "failures": dict(failure_kinds),
        }

    # --- quality scores, unblinded here and only here -----------------------
    by_provider_axis: dict[tuple[str, str], list[float]] = defaultdict(list)
    score_by_image: dict[tuple[str, str], dict] = {}
    excluded_by_provider: dict[str, int] = defaultdict(int)
    for unit_id, unit in blind_map["units"].items():
        entry = scores.get(unit_id)
        if not entry or entry.get("score") is None:
            continue
        degenerate = _is_degenerate(unit)
        if degenerate:
            # Kept visible in the grid, kept out of the arithmetic.
            excluded_by_provider[unit["provider"]] += 1
        else:
            # One score per unit, so a set of three images contributes one number.
            by_provider_axis[(unit["provider"], entry["axis"])].append(entry["score"])
        is_set = len(unit["image_ids"]) > 1
        for image_id in unit["image_ids"]:
            record = images[image_id]
            score_by_image[(record["provider"], record["prompt_id"])] = {
                **entry,
                "set": is_set,
                "excluded": degenerate,
            }

    axes = sorted({p["axis"] for p in prompts})
    for provider in providers:
        ops[provider]["axis_scores"] = {
            axis: _mean(by_provider_axis.get((provider, axis), [])) for axis in axes
        }
        # Unweighted mean across axes, not across units. prompt_fidelity and
        # text_rendering have three units each while the grouped axes have one, and
        # that ratio is a property of how the suite was written, not of the models.
        axis_means = [m for m in ops[provider]["axis_scores"].values() if m is not None]
        ops[provider]["overall_score"] = _mean(axis_means)
        ops[provider]["axes_scored"] = len(axis_means)
        ops[provider]["n_scored"] = sum(
            len(by_provider_axis.get((provider, axis), [])) for axis in axes
        )
        ops[provider]["excluded_singletons"] = excluded_by_provider.get(provider, 0)

    (run_dir / "summary.json").write_text(json.dumps(ops, indent=2), encoding="utf-8")
    html_path = run_dir / "report.html"
    excluded_total = sum(excluded_by_provider.values())
    html_path.write_text(
        _render_html(payload, ops, providers, axes, prompts, score_by_image,
                     image_paths, excluded_total),
        encoding="utf-8",
    )

    print(f"\n  {'provider':<22}{'ok%':>7}{'gen s':>8}  {'cache':>14}{'score':>8}  failures")
    for provider in providers:
        o = ops[provider]
        score = o["overall_score"]
        fails = ", ".join(f"{k}:{v}" for k, v in o["failures"].items()) or "-"
        gen = o["median_generated_latency_s"]
        print(
            f"  {provider:<22}{o['success_rate']:>6.1f}%"
            f"{(f'{gen:.2f}' if gen is not None else '-'):>8}"
            f"  {_cache_cell(o):>14}"
            f"{(f'{score:.2f}' if score is not None else '-'):>8}  {fails}"
        )
    if any(ops[p]["cache"] and ops[p]["median_generated_latency_s"] is None for p in providers):
        print(f"\n  note: a provider served only cache hits, so this run holds no")
        print("        latency measurement for it. Measure with fresh prompts.")
    print(f"\n  report → {html_path}")
    return html_path


def _render_html(payload, ops, providers, axes, prompts, score_by_image, image_paths,
                 excluded_total=0) -> str:
    def esc(value) -> str:
        return html.escape(str(value), quote=True)

    def cell(provider: str, prompt_id: str) -> str:
        match = next(
            (
                r for r in payload["results"]
                if r["provider"] == provider and r["prompt_id"] == prompt_id
            ),
            None,
        )
        if match is None:
            return '<td class="miss">—</td>'
        if not match["ok"]:
            return (
                f'<td class="fail"><span class="tag">{esc(match["error_kind"])}</span>'
                f'<div class="detail">{esc((match["error_detail"] or "")[:160])}</div></td>'
            )
        src = image_paths.get((provider, prompt_id))
        entry = score_by_image.get((provider, prompt_id))
        badge = ""
        if entry:
            if entry.get("excluded"):
                badge = (f'<span class="score excluded">{entry["score"]}/5 '
                         "not counted</span>")
            else:
                mark = " set" if entry.get("set") else ""
                badge = f'<span class="score">{entry["score"]}/5{mark}</span>' 
        note = f'<div class="detail">{esc(entry["note"])}</div>' if entry and entry.get("note") else ""
        return (
            f'<td><img src="{esc(src)}" loading="lazy" alt="">'
            f'<div class="meta">{match["latency_s"]:.1f}s {badge}</div>{note}</td>'
        )

    summary_rows = "".join(
        f"<tr><td class='name'>{esc(p)}</td>"
        f"<td>{ops[p]['succeeded_first_attempt']}/{ops[p]['attempts']} "
        f"({ops[p]['first_attempt_rate']}%)</td>"
        f"<td>{ops[p]['succeeded']}/{ops[p]['attempts']} ({ops[p]['success_rate']}%)</td>"
        f"<td>{ops[p]['median_generated_latency_s'] if ops[p]['median_generated_latency_s'] is not None else '—'}</td>"
        f"<td>{ops[p]['max_generated_latency_s'] if ops[p]['max_generated_latency_s'] is not None else '—'}</td>"
        f"<td>{_cache_cell(ops[p])}</td>"
        f"<td class='cdn'>{ops[p]['median_cache_read_s'] if ops[p]['median_cache_read_s'] is not None else '—'}</td>"
        f"<td>{_cost_cell(ops[p])}</td>"
        f"<td>{esc(', '.join(ops[p]['formats'])) or '—'}</td>"
        + "".join(
            f"<td>{ops[p]['axis_scores'].get(a) or '—'}</td>"
            for a in axes
        )
        + f"<td class='overall'>{_overall_cell(ops[p], len(axes))}</td></tr>"
        for p in providers
    )

    grid_rows = ""
    for axis in axes:
        axis_prompts = [p for p in prompts if p["axis"] == axis]
        grid_rows += f'<tr class="axis-row"><td colspan="{len(providers) + 1}">{esc(axis)}</td></tr>'
        for spec in axis_prompts:
            prompt_text = " ".join(spec["prompt"].split())
            grid_rows += (
                f'<tr><td class="prompt"><strong>{esc(spec["id"])}</strong>'
                f'<div class="detail">{esc(prompt_text)}</div></td>'
                + "".join(cell(p, spec["id"]) for p in providers)
                + "</tr>"
            )

    excluded_note = (
        f"{excluded_total} score(s) are shown struck through and excluded from every "
        "mean: a set-scored axis left with a single image by generation failures was "
        "judged against criteria about a set, most of which cannot be answered from "
        "one image. The score is the honest record of what was entered and is not a "
        "comparable measurement. "
    ) if excluded_total else ""

    axis_headers = "".join(f"<th>{esc(a)}</th>" for a in axes)
    provider_headers = "".join(f"<th>{esc(p)}</th>" for p in providers)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>genai-image-bench · {payload['run_id']}</title>
<style>
  :root {{ --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b66; --line:#e3e3df; --accent:#1f6f54; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#17171a; --fg:#ececea; --muted:#9a9a95; --line:#2e2e32; --accent:#5fcfa4; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--fg);
    font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:28px; }}
  h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:.07em;
    color:var(--muted); margin:36px 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left;
    vertical-align:top; }}
  th {{ background:color-mix(in srgb, var(--fg) 5%, transparent);
    font-weight:600; font-size:12px; }}
  .name,.overall {{ font-weight:600; }}
  .axis-row td {{ background:color-mix(in srgb, var(--accent) 12%, transparent);
    font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:12px; }}
  .prompt {{ width:230px; }}
  .detail {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  img {{ width:100%; max-width:230px; border-radius:4px; display:block; }}
  .meta {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .score {{ color:var(--accent); font-weight:600; }}
  .score.excluded {{ color:var(--muted); font-weight:400;
    text-decoration:line-through solid 1px; }}
  .qual {{ font-weight:400; text-transform:none; letter-spacing:0;
    color:var(--muted); font-size:11px; }}
  .cdn {{ color:var(--muted); }}
  .fail .tag {{ color:#c0392b; font-weight:600; font-size:12px; }}
  .miss {{ color:var(--muted); text-align:center; }}
  footer {{ margin-top:40px; padding-top:16px; border-top:1px solid var(--line);
    color:var(--muted); font-size:12px; }}
</style></head><body><div class="wrap">
<h1>genai-image-bench</h1>
<div class="sub">run {payload['run_id']} · generated {payload['generated_at']} ·
scored blind, unblinded at report time</div>

<h2>Summary</h2>
<table><thead><tr><th>provider</th>
<th>success<br><span class="qual">first attempt</span></th>
<th>success<br><span class="qual">with retries</span></th>
<th>median latency<br><span class="qual">generated</span></th>
<th>max latency<br><span class="qual">generated</span></th>
<th>cache</th>
<th>cache read<br><span class="qual">CDN, not the model</span></th>
<th>est. cost</th><th>format</th>{axis_headers}
<th>overall<br><span class="qual">unweighted mean of the axes</span></th></tr></thead>
<tbody>{summary_rows}</tbody></table>

<h2>Outputs</h2>
<table><thead><tr><th>prompt</th>{provider_headers}</tr></thead>
<tbody>{grid_rows}</tbody></table>

<footer>
{excluded_note}The per-axis columns are the result. "overall" is an unweighted mean across the four
axes, included only as a summary: it weights each axis equally regardless of how many
prompts the suite happens to contain for it, and an aggregate hides exactly the
per-axis differences this comparison exists to show.
Latency columns marked "generated" exclude cache hits; a cache read measures the
provider's CDN, not the model, so the two are never averaged. A provider that served
only cache hits has no latency measurement in this run.
Scores are from a single human scorer on a small prompt set and indicate direction,
not statistical significance. Cost is list price times successful images, not billing
read from an invoice. Free-tier endpoints may differ from paid tiers in resolution and
throughput, so latency here is not representative of paid performance.
</footer>
</div></body></html>"""
