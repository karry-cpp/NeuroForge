"""Tests for the entry analysis.

The point of these is the derivation, not the prose. The ids `targets_for`
returns are fed straight to `viewer.selectMany()`, so a silent change here
shows up as a brain that highlights the wrong parts - or nothing at all.
"""

import os
import unittest

os.environ.setdefault("NEUROFORGE_LLM", "off")

from neuroforge import analyse
from neuroforge.anatomy.anchors import NODE_TO_CORTEX, NODE_TO_STRUCTURE
from neuroforge.atlas import EDGES, PATHWAYS
from neuroforge.events import EVENTS_BY_ID


class TestDerivation(unittest.TestCase):

    def test_every_atlas_node_maps_somewhere(self):
        # A node that maps to neither is invisible: the analysis would claim
        # a pathway is engaged and then light nothing up for part of it.
        nodes = {n for e in EDGES for n in (e.src, e.dst)}
        unmapped = nodes - set(NODE_TO_CORTEX) - set(NODE_TO_STRUCTURE)
        self.assertEqual(unmapped, set())

    def test_rumination_engages_its_pathways(self):
        t = analyse.targets_for(["rumination"])
        ids = [p["id"] for p in t["pathways"]]
        self.assertIn("rumination", ids)
        self.assertIn("amygdala", t["structures"])

    def test_direction_is_signed_not_flattened(self):
        # "regulation +0.70" and "rumination -0.20" are different claims.
        t = analyse.targets_for(["name_emotion"])
        by_id = {p["id"]: p for p in t["pathways"]}
        self.assertEqual(by_id["regulation"]["direction"], "strengthen")
        self.assertEqual(by_id["rumination"]["direction"], "weaken")

    def test_unknown_event_is_dropped_not_coerced(self):
        t = analyse.targets_for(["definitely_not_an_event"])
        self.assertEqual(t["events"], [])
        self.assertEqual(t["pathways"], [])
        self.assertEqual(t["regions"], [])

    def test_strongest_engagement_wins_across_events(self):
        t = analyse.targets_for(["pause", "reappraisal"])
        reg = [p for p in t["pathways"] if p["id"] == "regulation"][0]
        expected = max(EVENTS_BY_ID["pause"].activations["regulation"],
                       EVENTS_BY_ID["reappraisal"].activations["regulation"])
        self.assertAlmostEqual(reg["amount"], expected, places=3)

    def test_pathways_ordered_by_magnitude(self):
        t = analyse.targets_for(["avoidance"])
        sizes = [abs(p["amount"]) for p in t["pathways"]]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_beneficial_is_not_the_raw_sign(self):
        # Strengthening the stress habit is +0.70 and still bad news. Colour
        # driven by sign alone painted it green in the bars and on the brain.
        t = analyse.targets_for(["reactive_outburst"])
        by_id = {p["id"]: p for p in t["pathways"]}
        habit = by_id["stress_habit"]
        self.assertGreater(habit["amount"], 0)
        self.assertFalse(habit["beneficial"])
        reg = by_id["regulation"]
        self.assertLess(reg["amount"], 0)
        self.assertFalse(reg["beneficial"])

    def test_practice_is_beneficial_both_ways(self):
        t = analyse.targets_for(["pause"])
        for p in t["pathways"]:
            self.assertTrue(p["beneficial"], p["id"])

    def test_every_named_pathway_is_real(self):
        for eid in EVENTS_BY_ID:
            for p in analyse.targets_for([eid])["pathways"]:
                self.assertIn(p["id"], PATHWAYS)


class TestRulesText(unittest.TestCase):

    def test_works_with_no_model(self):
        a = analyse.analyse("I kept replaying the argument.", ["rumination"])
        self.assertEqual(a["source"], "rules")
        self.assertTrue(a["text"])

    def test_no_model_is_not_reported_as_an_error(self):
        # The default state of the app is "nothing connected". Calling that a
        # failure in the UI would make a working install look broken.
        a = analyse.analyse("I paused.", ["pause"])
        self.assertEqual(a.get("note", ""), "")

    def test_empty_event_list_says_so(self):
        a = analyse.analyse("something vague", [])
        self.assertIn("no rule", a["text"].lower())

    def test_state_event_is_not_reported_as_unmatched(self):
        # sleep_poor carries no activations at all. Reading only pathways
        # made it come back as "nothing matched", which was simply untrue.
        a = analyse.analyse("Slept badly again.", ["sleep_poor"])
        self.assertNotIn("nothing in this entry", a["text"].lower())
        self.assertIn("sleep", a["text"].lower())

    def test_state_event_exposes_modulators(self):
        t = analyse.targets_for(["sleep_poor"])
        self.assertEqual(t["pathways"], [])
        kinds = {m["kind"] for m in t["modulators"]}
        self.assertIn("sleep", kinds)
        self.assertIn("stress", kinds)

    def test_rules_text_names_the_event(self):
        a = analyse.analyse("I paused before reacting.", ["pause"])
        self.assertIn("paused", a["text"].lower())


class TestCleaning(unittest.TestCase):

    def test_think_block_is_removed(self):
        out = analyse._clean("<think>reasoning here</think>The answer.")
        self.assertEqual(out, "The answer.")

    def test_markdown_and_markers_stripped(self):
        out = analyse._clean("**bold** [[show:amygdala]] text")
        self.assertNotIn("*", out)
        self.assertNotIn("[[", out)

    def test_smart_quotes_normalised(self):
        out = analyse._clean("the situation\u2019s safety")
        self.assertIn("situation's", out)


if __name__ == "__main__":
    unittest.main()
