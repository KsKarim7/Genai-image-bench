#!/usr/bin/env python3
"""Entry point.

    python run.py generate [--axis AXIS] [--repeats N]
    python run.py score  <run_id> [--no-open]
    python run.py report <run_id>
    python run.py runs
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
CONFIG = ROOT / "config" / "prompts.yaml"
RUNS = ROOT / "runs"


def resolve_run(run_id: str) -> Path:
    run_dir = RUNS / run_id
    if not run_dir.exists():
        available = sorted(p.name for p in RUNS.glob("*") if p.is_dir())
        raise SystemExit(
            f"no run {run_id!r}.\n"
            + ("available: " + ", ".join(available) if available else "no runs yet")
        )
    return run_dir


def main() -> None:
    # Windows stdout defaults to the locale codepage (cp1252 here), which cannot
    # encode the glyphs this CLI prints: redirecting output to a file or pipe
    # raises UnicodeEncodeError. The console itself is already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(prog="genai-image-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="run the prompt suite against all providers")
    gen.add_argument("--axis", help="limit to one axis (e.g. text_rendering)")
    gen.add_argument("--repeats", type=int, default=1,
                     help="generations per prompt per provider (default 1)")

    sc = sub.add_parser("score", help="blind scoring pass over a run")
    sc.add_argument("run_id")
    sc.add_argument("--no-open", action="store_true",
                    help="print image paths instead of opening them")

    rp = sub.add_parser("report", help="unblind scores and build report.html")
    rp.add_argument("run_id")

    sub.add_parser("runs", help="list runs")

    args = parser.parse_args()

    if args.command == "generate":
        from bench.runner import run
        asyncio.run(run(CONFIG, RUNS, axis=args.axis, repeats=args.repeats))

    elif args.command == "score":
        from bench.score import score_run
        score_run(resolve_run(args.run_id), open_images=not args.no_open)

    elif args.command == "report":
        from bench.report import build_report
        build_report(resolve_run(args.run_id))

    elif args.command == "runs":
        dirs = sorted((p for p in RUNS.glob("*") if p.is_dir()), reverse=True)
        if not dirs:
            print("no runs yet — python run.py generate")
            return
        for d in dirs:
            scored = "scored" if (d / "scores.json").exists() else "unscored"
            reported = " · reported" if (d / "report.html").exists() else ""
            print(f"  {d.name}  {scored}{reported}")


if __name__ == "__main__":
    sys.exit(main())
