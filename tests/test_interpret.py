"""Tests for the free-text interpreter."""

import os
import unittest

from neuroforge import discover
from neuroforge.events import EVENTS_BY_ID
from neuroforge.interpret import (Interpretation, _validate, classify_keywords,
                                  interpret)

_SAVED = None


def setUpModule():
    """Keep the suite hermetic.

    Auto-discovery means `interpret()` will happily call whatever model
    server the developer has open. That made these tests hit the network,
    take as long as the loaded model takes to generate, and assert against
    its output - the suite hung for minutes once a 9B was loaded where a
    0.5B had been. Nothing here is testing the model, so turn it off.
    """
    global _SAVED
    _SAVED = os.environ.get("NEUROFORGE_LLM")
    os.environ["NEUROFORGE_LLM"] = "off"
    discover.find(force=True)


def tearDownModule():
    if _SAVED is None:
        os.environ.pop("NEUROFORGE_LLM", None)
    else:
        os.environ["NEUROFORGE_LLM"] = _SAVED


class TestKeywordClassifier(unittest.TestCase):

    def _top(self, text):
        r = classify_keywords(text)
        return r.proposals[0].event if r.proposals else None

    def test_regulated_success(self):
        self.assertEqual(self._top("Boss criticised me and I stayed calm"),
                         "regulated_success")

    def test_negation_is_not_an_outburst(self):
        # The user's own example. A naive matcher sees "angry" and calls this
        # a reactive outburst - the exact opposite of what happened.
        self.assertEqual(self._top("He provoked me but I didn't get angry"),
                         "regulated_success")

    def test_outburst_still_detected(self):
        self.assertEqual(self._top("I lost my temper and shouted at him"),
                         "reactive_outburst")

    def test_rumination(self):
        self.assertEqual(self._top("I kept replaying the conversation"),
                         "rumination")

    def test_unmatched_returns_nothing_rather_than_guessing(self):
        r = classify_keywords("Tuesday. The weather was fine.")
        self.assertEqual(r.proposals, [])
        self.assertIn("No phrase", r.note)

    def test_confidence_is_capped_low(self):
        # A keyword hit is evidence a word appeared, not that it was meant.
        r = classify_keywords("I paused before reacting")
        self.assertLessEqual(r.proposals[0].confidence, 0.62)

    def test_intensity_responds_to_hedging(self):
        strong = classify_keywords("I meditated")
        weak = classify_keywords("I meditated a little")
        self.assertLess(weak.proposals[0].intensity,
                        strong.proposals[0].intensity)

    def test_every_cue_maps_to_a_real_event(self):
        from neuroforge.interpret import _CUES
        for eid in _CUES:
            self.assertIn(eid, EVENTS_BY_ID, f"{eid} is not a real event")


class TestValidation(unittest.TestCase):
    """The model is never trusted. These are the guards that enforce it."""

    def test_unknown_event_is_dropped_not_coerced(self):
        r = _validate({"proposals": [
            {"event": "rewired_my_amygdala", "intensity": 1, "confidence": 1}]},
            "llm:test")
        self.assertEqual(r.proposals, [])
        self.assertIn("discarded", r.note)

    def test_out_of_range_intensity_is_clamped(self):
        r = _validate({"proposals": [
            {"event": "pause", "intensity": 99, "confidence": 0.5}]},
            "llm:test")
        self.assertEqual(r.proposals[0].intensity, 1.5)

    def test_garbage_types_do_not_crash(self):
        r = _validate({"proposals": [
            {"event": "pause", "intensity": "lots", "confidence": None}]},
            "llm:test")
        self.assertEqual(r.proposals, [])

    def test_missing_proposals_key(self):
        self.assertEqual(_validate({}, "llm:test").proposals, [])

    def test_caps_at_three(self):
        r = _validate({"proposals": [
            {"event": "pause", "intensity": 1, "confidence": 0.5}] * 9},
            "llm:test")
        self.assertLessEqual(len(r.proposals), 3)


class TestInterpretEntryPoint(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(interpret("").proposals, [])

    def test_offline_path_is_used_when_no_llm_configured(self):
        # setUpModule forces NEUROFORGE_LLM=off. Without that this asserts a
        # property of the developer's desktop, not of the code.
        self.assertEqual(os.environ.get("NEUROFORGE_LLM"), "off")
        r = interpret("I stayed calm", allow_llm=True)
        self.assertEqual(r.source, "keyword")

    def test_returns_interpretation(self):
        self.assertIsInstance(interpret("I meditated"), Interpretation)


if __name__ == "__main__":
    unittest.main()
