"""Blind scoring pass.

This module deliberately never opens blind_map.json. It presents outputs in shuffled
order, identified only by an opaque id, and writes scores keyed by that id. Provider
identity is rejoined later, in report.py.

Why bother: in an unblinded comparison the scorer knows which output came from the
model they expect to win, and that expectation moves the score. The effect is well
documented in human evaluation generally, and it is exactly the kind of measurement
artifact that makes published model comparisons hard to trust. Blinding costs about
twenty lines of code, so there is no good reason to skip it.
"""

from __future__ import annotations

import json
import random
import webbrowser
from pathlib import Path


def _prompt_for_int(label: str, low: int, high: int) -> int | None:
    while True:
        raw = input(f"    {label} [{low}-{high}, s=skip, q=quit]: ").strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return -1
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"      enter a number {low}-{high}, or s / q")


def score_run(run_dir: Path, open_images: bool = True) -> None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"no results.json in {run_dir}")

    payload = json.loads(results_path.read_text())
    prompts_by_id = {p["id"]: p for p in payload["prompts"]}
    axes = payload["axes"]

    blind_map_path = run_dir / "blind_map.json"
    if not blind_map_path.exists():
        raise SystemExit("blind_map.json missing — rerun generate")

    # We read the blind map ONLY for the image path and prompt id. Provider is
    # popped before anything reaches the screen, so an accidental print cannot
    # leak it. The unblinding happens in report.py and nowhere else.
    raw_map = json.loads(blind_map_path.read_text())
    items = []
    for blind_id, rec in raw_map.items():
        items.append(
            {
                "blind_id": blind_id,
                "prompt_id": rec["prompt_id"],
                "image_path": rec["image_path"],
            }
        )

    random.shuffle(items)

    scores_path = run_dir / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    remaining = [i for i in items if i["blind_id"] not in scores]
    if not remaining:
        print("every output already scored. delete scores.json to redo.")
        return

    print(f"\nblind scoring — {len(remaining)} outputs remaining")
    print("outputs are shuffled and anonymous. score what you see, not what you expect.\n")

    for idx, item in enumerate(remaining, 1):
        spec = prompts_by_id[item["prompt_id"]]
        axis = spec["axis"]
        axis_info = axes[axis]

        print(f"\n[{idx}/{len(remaining)}]  output {item['blind_id']}")
        print(f"  axis:   {axis}")
        print(f"  scale:  {axis_info['scale']}")
        print(f"  prompt: {' '.join(spec['prompt'].split())}")
        print("  checks:")
        for check in spec["checks"]:
            print(f"    - {check}")

        image_file = run_dir / item["image_path"]
        print(f"  image:  {image_file}")
        if open_images:
            try:
                webbrowser.open(image_file.resolve().as_uri())
            except Exception:
                pass  # viewing manually is fine; the path is printed above

        value = _prompt_for_int("score", 1, 5)
        if value is None:
            print("\nstopped. progress saved — rerun to continue.")
            break
        if value == -1:
            continue

        note = input("    note (optional, enter to skip): ").strip()
        scores[item["blind_id"]] = {"score": value, "note": note or None, "axis": axis}
        scores_path.write_text(json.dumps(scores, indent=2))

    print(f"\n  {len(scores)}/{len(items)} scored → {scores_path}")
    print(f"\n  next: python run.py report {run_dir.name}")
