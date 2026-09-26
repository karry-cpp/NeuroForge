"""Tests for the self-described starting point and the engagement read-out."""

import os
import tempfile
import unittest

os.environ.setdefault("NEUROFORGE_LLM", "off")

from neuroforge import engagement as E
from neuroforge import reactions as R
from neuroforge.engine import Simulation
from neuroforge.events import EVENTS_BY_ID

RUMINATOR = {"criticised": "replay", "plans": "chew", "dreaded": "brood",
             "night": "replay", "cutoff": "stew", "feeling": "push",
             "week": "lowenergy", "move": "rarely"}
REGULATED = {"criticised": "useful", "plans": "redirect", "dreaded": "steps",
             "night": "calm", "cutoff": "letgo", "feeling": "name",
             "week": "calm", "move": "often"}


def levels(sim):
    rows = E.engagement(sim.current.snapshot(), sim.warmup_days,
                        sim.warmup_days)
    return {r["part"]: r["level"] for r in rows}


class TestQuestions(unittest.TestCase):

    def test_every_option_names_real_events(self):
        for q in R.QUESTIONS:
            for o in q["options"]:
                for eid in o["events"]:
                    self.assertIn(eid, EVENTS_BY_ID, f"{q['id']}/{o['id']}")

    def test_unknown_answers_are_dropped_not_coerced(self):
        got = R.chosen({"criticised": "nope", "nosuch": "replay",
                        "plans": "chew"})
        self.assertEqual([(q["id"], o["id"]) for q, o, _ in got],
                         [("plans", "chew")])

    def test_every_option_and_event_has_a_plain_phrase(self):
        for q in R.QUESTIONS:
            for o in q["options"]:
                self.assertIn(f"{q['id']}/{o['id']}", R.BECAUSE)
        self.assertEqual(set(R.RARELY), set(EVENTS_BY_ID))
        self.assertEqual(set(E.MEANING), set(E.PARTS))


class TestBuild(unittest.TestCase):

    def test_same_answers_same_brain(self):
        a, b = R.build(RUMINATOR), R.build(RUMINATOR)
        self.assertEqual(a.current.snapshot(), b.current.snapshot())

    def test_profile_history_is_not_logged_practice(self):
        sim = R.build(REGULATED)
        self.assertEqual(sim.day, 0)
        self.assertEqual(sim.log, [])
        self.assertEqual(sim.streak(), 0)
        self.assertAlmostEqual(sim.progress, 0.0, places=6)
        self.assertEqual(sim.warmup_days, R.WEEKS * 7)

    def test_profile_survives_save_and_load(self):
        sim = R.build(RUMINATOR)
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        sim.save(path)
        back = Simulation.load(path)
        self.assertEqual(back.profile, sim.profile)
        self.assertEqual(back.warmup_days, sim.warmup_days)


class TestPatterns(unittest.TestCase):

    class _P:
        def __init__(self, event, intensity=1.0):
            self.event, self.intensity = event, intensity

    def test_note_keeps_only_known_events(self):
        n = R.note_from("x", [self._P("rumination"), self._P("made_up")])
        self.assertEqual(list(n["events"]), ["rumination"])
        self.assertIsNone(R.note_from("x", [self._P("made_up")]))

    def test_intensity_scales_the_rate(self):
        n = R.note_from("x", [self._P("rumination", 0.5)])
        self.assertAlmostEqual(n["events"]["rumination"], R.TEXT_RATE * 0.5)

    def test_a_pattern_moves_the_readout(self):
        base = levels(R.build(REGULATED))
        note = R.note_from("I go over things for hours",
                           [self._P("rumination", 1.5)])
        more = R.build(REGULATED, [note, dict(note), dict(note)])
        self.assertNotEqual(levels(more), base)

    def test_refining_keeps_everything_logged(self):
        sim = R.build(RUMINATOR)
        sim.log_event("pause")
        sim.advance_day(2)
        sim.log_event("reappraisal")
        note = R.note_from("x", [self._P("exercise")])
        again = R.build(RUMINATOR, [note], log=sim.log, day=sim.day)
        self.assertEqual(again.day, sim.day)
        self.assertEqual([(e.day, e.event_id) for e in again.log],
                         [(e.day, e.event_id) for e in sim.log])

    def test_old_single_description_still_reads(self):
        old = {"answers": {}, "text": "hi", "text_events": ["rumination"]}
        self.assertEqual(list(R.notes_of(old)[0]["events"]), ["rumination"])


class TestEngagement(unittest.TestCase):

    def test_default_brain_is_typical_everywhere(self):
        self.assertEqual(set(levels(Simulation(seed=0)).values()), {"typical"})

    def test_saying_nothing_moves_nothing(self):
        # Skipped questions count as typical, so an empty profile IS typical.
        self.assertEqual(set(levels(R.build({})).values()), {"typical"})

    def test_typical_splits_into_balanced_and_unknown(self):
        ans = {"criticised": "replay"}
        sim = R.build(ans)
        rows = {r["part"]: r for r in E.engagement(
            sim.current.snapshot(), sim.warmup_days, sim.warmup_days,
            R.sources(ans), R.profile_rates(ans))}
        # Rumination never acts on the new-habit pathway through NAcc.
        self.assertEqual(rows["accumbens"]["basis"], "unknown")
        for r in rows.values():
            if r["level"] != "typical":
                self.assertEqual(r["basis"], "")

    def test_several_options_per_situation(self):
        both = R.chosen({"criticised": ["replay", "snap"]})
        self.assertEqual([o["id"] for _, o, _ in both], ["replay", "snap"])
        self.assertAlmostEqual(both[0][2], 1 / 2 ** 0.5)
        sim = R.build({"criticised": ["replay", "snap"]})
        self.assertEqual(sim.profile["answers"],
                         {"criticised": ["replay", "snap"]})

    def test_ruminator_reads_as_threat_up_regulation_down(self):
        lv = levels(R.build(RUMINATOR))
        self.assertEqual(lv["amygdala"], "more")
        self.assertEqual(lv["acc"], "more")
        self.assertEqual(lv["vlPFC"], "less")
        self.assertEqual(lv["vmPFC"], "less")

    def test_regulated_reads_the_other_way(self):
        lv = levels(R.build(REGULATED))
        self.assertEqual(lv["amygdala"], "less")
        self.assertEqual(lv["vlPFC"], "more")

    def test_regulation_onto_amygdala_counts_against_it(self):
        # Without DAMPENS a stronger brake would read as a busier amygdala.
        ref = E.reference(0)
        w = dict(ref)
        w["vlPFC->BLA"] += 0.3
        ns = E.node_scores(w, 0)
        self.assertLess(ns["BLA"], 0)
        self.assertGreater(ns["vlPFC"], 0)

    def test_every_part_is_explained(self):
        sim = R.build(RUMINATOR)
        srcs = R.sources(RUMINATOR)
        rows = E.engagement(sim.current.snapshot(), sim.warmup_days,
                            sim.warmup_days, srcs,
                            R.profile_rates(RUMINATOR))
        for r in rows:
            if r["level"] != "typical":
                self.assertTrue(r["why"]["meaning"], r["part"])
                self.assertTrue(r["why"]["reasons"], r["part"])

    def test_no_profile_means_no_readout(self):
        # Without answers there is nothing about the person to compare.
        from neuroforge.server import routes
        sim = Simulation(seed=0)
        sim.log_event("rumination")
        self.assertEqual(routes._state(sim)["engagement"], [])


if __name__ == "__main__":
    unittest.main()
