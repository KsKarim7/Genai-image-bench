"""Checks on loading the prompt suite.

The axis filter had no coverage, and a parameter shadow in the group validation
silently made every run use one axis regardless of --axis. The whole blinding
suite stayed green through it, so these assert the loader itself.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import yaml

from bench.runner import load_prompts

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITE = ROOT / "config" / "prompts.yaml"


def _write(cfg) -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "prompts.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tmp


def _suite() -> dict:
    return yaml.safe_load(SUITE.read_text(encoding="utf-8"))


class AxisFilter(unittest.TestCase):
    def test_no_axis_returns_the_whole_suite(self):
        _, prompts = load_prompts(SUITE)
        self.assertEqual(len(prompts), len(_suite()["prompts"]))

    def test_each_axis_returns_only_its_own_prompts(self):
        declared = _suite()["axes"]
        for axis in declared:
            _, prompts = load_prompts(SUITE, axis)
            self.assertTrue(prompts, f"{axis} returned nothing")
            self.assertEqual({p["axis"] for p in prompts}, {axis})

    def test_axes_partition_the_suite(self):
        seen = []
        for axis in _suite()["axes"]:
            seen += [p["id"] for p in load_prompts(SUITE, axis)[1]]
        self.assertEqual(sorted(seen), sorted(p["id"] for p in _suite()["prompts"]))

    def test_unknown_axis_exits(self):
        with self.assertRaises(SystemExit):
            load_prompts(SUITE, "no_such_axis")

    def test_config_bundle_carries_axes_and_groups(self):
        config, _ = load_prompts(SUITE)
        self.assertEqual(set(config), {"axes", "groups"})
        self.assertTrue(config["groups"])


class Validation(unittest.TestCase):
    def test_axis_without_anchors_exits(self):
        cfg = _suite()
        cfg["axes"]["prompt_fidelity"].pop("anchors")
        with self.assertRaises(SystemExit):
            load_prompts(_write(cfg))

    def test_prompt_naming_an_undeclared_axis_exits(self):
        cfg = _suite()
        cfg["prompts"][0]["axis"] = "invented"
        with self.assertRaises(SystemExit):
            load_prompts(_write(cfg))

    def test_prompt_naming_an_undeclared_group_exits(self):
        cfg = _suite()
        cfg["groups"].pop("fox_courier")
        with self.assertRaises(SystemExit):
            load_prompts(_write(cfg))

    def test_group_without_checks_exits(self):
        cfg = _suite()
        cfg["groups"]["fox_courier"]["checks"] = []
        with self.assertRaises(SystemExit):
            load_prompts(_write(cfg))

    def test_group_axis_disagreeing_with_its_members_exits(self):
        cfg = _suite()
        cfg["groups"]["fox_courier"]["axis"] = "style_adherence"
        with self.assertRaises(SystemExit):
            load_prompts(_write(cfg))

    def test_the_real_suite_validates(self):
        config, prompts = load_prompts(SUITE)
        self.assertTrue(prompts)
        for name, group in config["groups"].items():
            self.assertTrue(group["checks"], name)


class GroupedPromptsCarryNoOwnChecks(unittest.TestCase):
    def test_grouped_criteria_live_on_the_group(self):
        config, prompts = load_prompts(SUITE)
        for prompt in prompts:
            group = prompt.get("consistency_group") or prompt.get("style_group")
            if group:
                self.assertFalse(
                    prompt.get("checks"),
                    f"{prompt['id']} repeats criteria the {group} group already states",
                )


if __name__ == "__main__":
    unittest.main()
