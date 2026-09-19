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


class TestKeywordCoverage(unittest.TestCase):
    """The offline path is what a first-time user with no model actually gets.

    "angry" and "skipped" were once absent from the cue table, so the most
    obvious sentence anyone types matched nothing at all.
    """

    def top(self, text):
        r = interpret(text, allow_llm=False)
        return r.proposals[0].event if r.proposals else None

    def test_plain_anger_is_matched(self):
        for text in ("i got angry",
                     "I got angry at my manager",
                     "i was angry all afternoon",
                     "i lost it with him"):
            self.assertEqual(self.top(text), "reactive_outburst", text)

    def test_plain_avoidance_is_matched(self):
        for text in ("i skipped the meeting because i was dreading it",
                     "i bailed on dinner",
                     "i stayed home instead"):
            self.assertEqual(self.top(text), "avoidance", text)

    def test_negation_still_flips_the_meaning(self):
        # "didn't get angry" is a regulated success; a matcher that only saw
        # "angry" would record the opposite of what happened.
        self.assertEqual(self.top("i didn't get angry even though he pushed"),
                         "regulated_success")
        self.assertEqual(self.top("i didn't snap"), "regulated_success")
        self.assertNotEqual(self.top("i never got angry"), "reactive_outburst")

    def test_dreading_does_not_outrank_doing_it_anyway(self):
        # Both cues are present in "made the call i was dreading"; the longer,
        # more specific one has to win or exposure reads as avoidance.
        self.assertEqual(self.top("i made the call i was dreading"), "exposure")

    def test_confidence_stays_capped(self):
        for p in interpret("i got angry and shouted", allow_llm=False).proposals:
            self.assertLessEqual(p.confidence, 0.62)


class TestDiscoveryCachesAMiss(unittest.TestCase):
    """A machine with no model runner must not re-probe on every entry.

    The cache held Optional[dict] and tested `is not None`, so "no runner
    found" was indistinguishable from "nothing cached yet". Every interpret
    paid two socket timeouts - about three seconds - and the result was
    still correct, which is why it went unnoticed.
    """

    def setUp(self):
        self.runners = discover._RUNNERS
        self.get = discover._get_json
        self.env = os.environ.pop("NEUROFORGE_LLM", None)
        self.calls = 0

        def counting(url, timeout=0.5):
            self.calls += 1
            return None                      # nothing is listening

        discover._get_json = counting
        discover._RUNNERS = (
            ("ollama", "http://127.0.0.1:59999/v1", "http://x/a"),
            ("lmstudio", "http://127.0.0.1:59998/v1", "http://x/b"),
        )

    def tearDown(self):
        discover._RUNNERS = self.runners
        discover._get_json = self.get
        if self.env is not None:
            os.environ["NEUROFORGE_LLM"] = self.env
        discover.find(force=True)

    def test_miss_is_cached(self):
        self.assertIsNone(discover.find(force=True))
        probes = self.calls
        self.assertEqual(probes, len(discover._RUNNERS))
        for _ in range(5):
            self.assertIsNone(discover.find())
        self.assertEqual(self.calls, probes, "a cached miss was re-probed")


if __name__ == "__main__":
    unittest.main()
