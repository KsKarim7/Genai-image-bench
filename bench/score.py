"""Blind scoring pass.

Reads scoring_manifest.json and nothing else: the manifest carries no provider name
and no field that joins to one, so this module cannot leak what it never loads.
Scores are keyed by unit id; report.py does the unblinding.
"""

from __future__ import annotations

import html
import json
import random
import tempfile
import webbrowser
from pathlib import Path

QUIT = "quit"
SKIP = "skip"
# The one place the score range is stated.
SCORE_RANGE = (1, 5)

_VIEW_CSS = """
  :root { color-scheme: dark light; }
  body { font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;
         margin:0; padding:28px; background:#17171a; color:#ececea; }
  h1 { font-size:14px; text-transform:uppercase; letter-spacing:.08em;
       color:#9a9a95; margin:0 0 6px; }
  .anchors { color:#9a9a95; font-size:13px; margin-bottom:18px; }
  .setchecks { border:1px solid #2e2e32; border-radius:6px; padding:12px 16px;
               margin-bottom:22px; }
  .setchecks ul { margin:0; padding-left:18px; color:#c9c9c4; font-size:13px; }
  .grid { display:flex; flex-wrap:wrap; gap:22px; align-items:flex-start; }
  figure { margin:0; max-width:380px; }
  img { width:100%; border-radius:6px; display:block; background:#000; }
  figcaption { font-size:13px; color:#c9c9c4; margin-top:8px; }
  b { display:block; color:#5fcfa4; font-size:11px; text-transform:uppercase;
      letter-spacing:.07em; margin-bottom:4px; }
  p { margin:0 0 6px; }
  ul { margin:0; padding-left:18px; color:#9a9a95; font-size:12px; }
"""


def _prompt_for_score(low: int, high: int) -> int | str:
    while True:
        try:
            raw = input(f"    score [{low}-{high}, s=skip, q=quit]: ").strip().lower()
        except EOFError:
            print()
            return QUIT
        if raw == "q":
            return QUIT
        if raw == "s":
            return SKIP
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"      enter a number {low}-{high}, or s / q")


def _view_html(unit: dict, run_dir: Path) -> str:
    """Images carry no suffix on disk, so they are shown through <img>, which
    sniffs the bytes instead of trusting a filename."""
    total = len(unit["images"])
    set_checks = unit.get("set_checks") or []
    set_block = ""
    if set_checks:
        items = "".join(f"<li>{html.escape(c)}</li>" for c in set_checks)
        set_block = (
            f'<div class="setchecks"><b>criteria for the whole set</b>'
            f"<ul>{items}</ul></div>"
        )
    cards = []
    for index, (rel, spec) in enumerate(zip(unit["images"], unit["prompts"]), 1):
        uri = (run_dir / rel).resolve().as_uri()
        checks = "".join(f"<li>{html.escape(c)}</li>" for c in spec["checks"])
        label = f"image {index} of {total}" if total > 1 else "output"
        cards.append(
            f'<figure><img src="{uri}" alt=""><figcaption><b>{label}</b>'
            f"<p>{html.escape(spec['prompt'])}</p><ul>{checks}</ul>"
            "</figcaption></figure>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(unit['unit_id'])}</title>"
        "<style>" + _VIEW_CSS + "</style></head><body>"
        f"<h1>{html.escape(unit['axis'])}</h1>"
        f"<div class=\"anchors\">{html.escape(unit['anchors'])}</div>"
        + set_block
        + f"<div class=\"grid\">{''.join(cards)}</div>"
        "</body></html>"
    )


def score_run(run_dir: Path, open_images: bool = True) -> None:
    manifest_path = run_dir / "scoring_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no scoring_manifest.json in {run_dir} - rerun generate")

    units = json.loads(manifest_path.read_text(encoding="utf-8"))["units"]
    random.shuffle(units)

    scores_path = run_dir / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}

    remaining = [u for u in units if u["unit_id"] not in scores]
    if not remaining:
        print("every unit already scored. delete scores.json to redo.")
        return

    views = Path(tempfile.mkdtemp(prefix="bench-score-")) if open_images else None

    print()
    print(f"blind scoring - {len(remaining)} of {len(units)} units remaining")
    print("units are shuffled and anonymous. score what you see, not what you expect.")

    try:
        for index, unit in enumerate(remaining, 1):
            total = len(unit["images"])
            print()
            print(f"[{index}/{len(remaining)}]  unit {unit['unit_id']}"
                  + (f"  ({total} images, scored as a set)" if total > 1 else ""))
            print(f"  axis:    {unit['axis']}")
            print(f"  anchors: {unit['anchors']}")
            if unit.get("set_checks"):
                print("  criteria for the whole set:")
                for check in unit["set_checks"]:
                    print(f"    - {check}")
            for spec in unit["prompts"]:
                print(f"  prompt: {spec['prompt']}")
                for check in spec["checks"]:
                    print(f"    - {check}")

            if views is not None:
                view = views / f"{unit['unit_id']}.html"
                view.write_text(_view_html(unit, run_dir), encoding="utf-8")
                print(f"  view:   {view}")
                try:
                    webbrowser.open(view.resolve().as_uri())
                except Exception:
                    pass  # the path is printed above
            else:
                for rel in unit["images"]:
                    print(f"  image:  {run_dir / rel}")

            value = _prompt_for_score(*SCORE_RANGE)
            if value == QUIT:
                print()
                print("stopped. progress saved - rerun to continue.")
                break
            if value == SKIP:
                continue

            try:
                note = input("    note (optional, enter to skip): ").strip()
            except EOFError:
                note = ""
            scores[unit["unit_id"]] = {
                "score": value,
                "note": note or None,
                "axis": unit["axis"],
            }
            scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        # Every score is flushed as it is entered, so there is nothing to unwind.
        print()
        print()
        print("interrupted. progress saved - rerun to continue.")

    print()
    print(f"  {len(scores)}/{len(units)} units scored -> {scores_path}")
    print()
    print(f"  next: python run.py report {run_dir.name}")
