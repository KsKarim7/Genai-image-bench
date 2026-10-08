"""Regression checks for the blinding boundary.

The central claim broke three times: the provider name sat in the image filename,
it was recoverable from results.json by matching meta.bytes to a file size on
disk, and it was recoverable from the file extension. All three survived a
careful read, so the property is asserted here rather than re-read.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml

from bench.providers import ALL_PROVIDERS, BaseProvider, GeneratedImage, GenerationResult
from bench.runner import _attempt
from bench.report import build_report
from bench.runner import build_blind_map, build_scoring_manifest, build_units
from bench.score import _view_html, score_run

ROOT = Path(__file__).resolve().parent.parent
FAKE_PROVIDERS = ("alpha", "beta")
ALL_NAMES = [c.name for c in ALL_PROVIDERS] + list(FAKE_PROVIDERS)
JOIN_KEYS = ("bytes", "latency", "content_type", "prompt_id", "provider", "x_cache")


def _load_suite():
    cfg = yaml.safe_load((ROOT / "config" / "prompts.yaml").read_text(encoding="utf-8"))
    return cfg["axes"], cfg["prompts"]


def _fake_results(prompts):
    """Two providers over the whole suite, with failures chosen so that one group
    is incomplete and one prompt has a single surviving provider."""
    results = []
    for provider in FAKE_PROVIDERS:
        for spec in prompts:
            ok = not (provider == "beta" and spec["id"] in {"cc_03", "tr_02"})
            blind_id = uuid.uuid4().hex[:10]
            results.append(
                GenerationResult(
                    provider=provider,
                    prompt_id=spec["id"],
                    blind_id=blind_id,
                    ok=ok,
                    latency_s=1.5,
                    image_path=f"images/{blind_id}" if ok else None,
                    error_kind=None if ok else "payment_required",
                    meta=(
                        {
                            "attempts": 1,
                            "bytes": 1234,
                            "content_type": "image/jpeg",
                            "provider_meta": {"x_cache": "MISS"},
                        }
                        if ok
                        else {"attempts": 3}
                    ),
                )
            )
    return results


def _make_run(root: Path):
    axes, prompts = _load_suite()
    results = _fake_results(prompts)
    units = build_units(prompts, results)

    run_dir = root / "20260101-000000"
    (run_dir / "images").mkdir(parents=True)
    # Distinct sizes, so a test asserting that no size leaks is not passing by
    # accident on identical files.
    for index, res in enumerate(r for r in results if r.ok):
        (run_dir / res.image_path).write_bytes(
            bytes([0xFF, 0xD8, 0xFF]) + b"x" * (4096 + index * 137)
        )

    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "axes": axes,
                "prompts": prompts,
                "results": [r.to_results_row() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "blind_map.json").write_text(
        json.dumps(build_blind_map(units), indent=2), encoding="utf-8"
    )
    (run_dir / "scoring_manifest.json").write_text(
        json.dumps(build_scoring_manifest(run_dir.name, axes, prompts, units), indent=2),
        encoding="utf-8",
    )
    return run_dir, units


def _drive_scorer(run_dir: Path, answers):
    """Run the scoring CLI non-interactively and capture everything it printed."""
    supply = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(supply)
        except StopIteration:
            raise EOFError

    real_input, builtins.input = builtins.input, fake_input
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            score_run(run_dir, open_images=False)
    finally:
        builtins.input = real_input
    return buffer.getvalue()


class ScorerSeesNoProvider(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_dir, self.units = _make_run(Path(tmp.name))
        self.manifest = json.loads(
            (self.run_dir / "scoring_manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_names_no_provider(self):
        blob = json.dumps(self.manifest)
        for name in ALL_NAMES:
            self.assertNotIn(name, blob)

    def test_manifest_carries_no_join_key(self):
        blob = json.dumps(self.manifest)
        for key in JOIN_KEYS:
            self.assertNotIn(key, blob, f"{key} would join the scorer view to a provider")

    def test_manifest_holds_no_value_matching_a_file_size(self):
        """The original leak: meta.bytes equalled the size on disk, so the
        provider fell out of an integer comparison."""

        def numbers(node):
            if isinstance(node, bool):
                return
            if isinstance(node, (int, float)):
                yield node
            elif isinstance(node, dict):
                for value in node.values():
                    yield from numbers(value)
            elif isinstance(node, list):
                for value in node:
                    yield from numbers(value)

        sizes = {p.stat().st_size for p in (self.run_dir / "images").iterdir()}
        self.assertGreater(len(sizes), 1, "fixture must not use identical file sizes")
        self.assertFalse(sizes & set(numbers(self.manifest)))

    def test_image_filenames_carry_no_suffix(self):
        for image in (self.run_dir / "images").iterdir():
            self.assertEqual(image.suffix, "", f"{image.name} reveals the response format")

    def test_results_rows_hold_no_blind_id_or_image_path(self):
        payload = json.loads((self.run_dir / "results.json").read_text(encoding="utf-8"))
        for row in payload["results"]:
            self.assertNotIn("blind_id", row)
            self.assertNotIn("image_path", row)

    def test_no_identifier_appears_in_results_json(self):
        blob = (self.run_dir / "results.json").read_text(encoding="utf-8")
        blind_map = json.loads((self.run_dir / "blind_map.json").read_text(encoding="utf-8"))
        for identifier in list(blind_map["images"]) + list(blind_map["units"]):
            self.assertNotIn(identifier, blob)

    def test_score_module_opens_only_the_manifest(self):
        source = (ROOT / "bench" / "score.py").read_text(encoding="utf-8")
        self.assertIn("scoring_manifest.json", source)
        self.assertNotIn("results.json", source)
        self.assertNotIn("blind_map.json", source)

    def test_console_output_names_no_provider(self):
        printed = _drive_scorer(self.run_dir, ["3", "", "4", "", "5", ""])
        for name in ALL_NAMES:
            self.assertNotIn(name, printed)

    def test_rendered_view_names_no_provider(self):
        for unit in self.manifest["units"]:
            page = _view_html(unit, self.run_dir)
            for name in ALL_NAMES:
                self.assertNotIn(name, page)

    def test_scores_file_holds_only_score_note_axis(self):
        _drive_scorer(self.run_dir, ["4", "a note"])
        saved = json.loads((self.run_dir / "scores.json").read_text(encoding="utf-8"))
        self.assertTrue(saved)
        for entry in saved.values():
            self.assertLessEqual(set(entry), {"score", "note", "axis"})


class UnitGrouping(unittest.TestCase):
    def setUp(self):
        self.axes, self.prompts = _load_suite()
        self.results = _fake_results(self.prompts)
        self.units = build_units(self.prompts, self.results)

    def test_no_unit_spans_providers(self):
        for unit in self.units:
            self.assertEqual(len({m.provider for m in unit["members"]}), 1)

    def test_group_members_follow_suite_order(self):
        order = {p["id"]: i for i, p in enumerate(self.prompts)}
        for unit in self.units:
            ids = [m.prompt_id for m in unit["members"]]
            self.assertEqual(ids, sorted(ids, key=lambda i: order[i]))

    def test_grouped_prompts_collapse_to_one_unit_per_provider(self):
        fox = [u for u in self.units if u["group"] == "fox_courier"]
        self.assertEqual(len(fox), 2)
        self.assertEqual(sorted(len(u["members"]) for u in fox), [2, 3])

    def test_ungrouped_prompts_stay_single(self):
        for unit in self.units:
            if unit["group"] is None:
                self.assertEqual(len(unit["members"]), 1)

    def test_failed_outputs_never_enter_a_unit(self):
        present = {(m.provider, m.prompt_id) for u in self.units for m in u["members"]}
        self.assertNotIn(("beta", "cc_03"), present)
        self.assertNotIn(("beta", "tr_02"), present)

    def test_unit_ids_are_unique(self):
        ids = [u["unit_id"] for u in self.units]
        self.assertEqual(len(ids), len(set(ids)))


class FilenameIsTheBlindId(unittest.TestCase):
    """Covers the code path that names files, not just the shape of the result.

    A mutation that renamed images back to provider__id passed every other test
    in this file, because the fixtures build image_path themselves.
    """

    def test_attempt_writes_the_blind_id_and_nothing_else(self):
        class Stub(BaseProvider):
            name = "stub-provider"

            async def generate(self, client, prompt):
                return GeneratedImage(
                    bytes([0xFF, 0xD8, 0xFF]) + b"payload",
                    "image/jpeg",
                    {"x_cache": "MISS"},
                )

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        images = Path(tmp.name)
        spec = {"id": "pf_01", "axis": "prompt_fidelity", "prompt": "a prompt", "checks": []}

        result = asyncio.run(_attempt(Stub(), None, spec, "deadbeef01", images))

        self.assertTrue(result.ok)
        self.assertEqual([p.name for p in images.iterdir()], ["deadbeef01"])
        self.assertEqual(result.image_path, "images/deadbeef01")
        self.assertNotIn("stub-provider", result.image_path)
        self.assertEqual(result.meta["content_type"], "image/jpeg")
        self.assertEqual(result.meta["provider_meta"]["x_cache"], "MISS")


class SetScoresCountOnce(unittest.TestCase):
    def test_a_grouped_set_contributes_one_number(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_dir, _ = _make_run(Path(tmp.name))

        blind_map = json.loads((run_dir / "blind_map.json").read_text(encoding="utf-8"))
        scores = {
            unit_id: {"score": 4, "note": None, "axis": unit["axis"]}
            for unit_id, unit in blind_map["units"].items()
        }
        (run_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            build_report(run_dir)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        alpha_units = sum(1 for u in blind_map["units"].values() if u["provider"] == "alpha")
        self.assertEqual(summary["alpha"]["n_scored"], alpha_units)
        self.assertLess(summary["alpha"]["n_scored"], summary["alpha"]["succeeded"])


if __name__ == "__main__":
    unittest.main()
