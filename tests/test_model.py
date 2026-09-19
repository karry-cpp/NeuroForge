"""Behavioural tests for the simulation model.

Run:  python -m tests.test_model      (or: python -m unittest discover)
"""

import unittest

from neuroforge.atlas import PATHWAYS
from neuroforge.engine import Simulation
from neuroforge.model import Connectome, alignment


class TestPlasticity(unittest.TestCase):

    def test_practice_strengthens_regulation(self):
        s = Simulation()
        before = s.current.pathway_strength("regulation")
        for _ in range(5):
            s.log_event("reappraisal")
            s.advance_day()
        self.assertGreater(s.current.pathway_strength("regulation"), before)

    def test_competition_weakens_rumination(self):
        """Strengthening regulation onto BLA should passively cost the
        rumination input onto BLA, without it ever being targeted directly."""
        s = Simulation()
        before = s.current.pathway_strength("rumination")
        for _ in range(20):
            s.log_event("pause")      # pause has NO rumination term
            s.advance_day()
        self.assertLess(s.current.pathway_strength("rumination"), before)

    def test_disuse_decays_but_consolidation_protects(self):
        fresh, trained = Simulation(seed=1), Simulation(seed=1)
        for _ in range(40):
            trained.log_event("reappraisal")
            trained.log_event("sleep_good")
            trained.advance_day()
        peak = trained.current.pathway_strength("regulation")
        trained.advance_day(60)     # two months off
        self.assertLess(trained.current.pathway_strength("regulation"), peak)
        self.assertGreater(trained.current.pathway_strength("regulation"),
                           fresh.current.pathway_strength("regulation"))

    def test_spacing_beats_massing(self):
        # Same seed on both: practice gain now varies, and comparing two
        # different sequences of luck would not test spacing at all.
        massed, spaced = Simulation(seed=1), Simulation(seed=1)
        for _ in range(20):
            massed.log_event("name_emotion")
        massed.advance_day(20)
        for _ in range(20):
            spaced.log_event("name_emotion")
            spaced.advance_day()
        self.assertGreater(spaced.current.pathway_strength("regulation"),
                           massed.current.pathway_strength("regulation"))

    def test_sleep_matters(self):
        good, bad = Simulation(seed=1), Simulation(seed=1)
        for sim, ev in ((good, "sleep_good"), (bad, "sleep_poor")):
            for _ in range(30):
                sim.log_event(ev)
                sim.log_event("reappraisal")
                sim.advance_day()
        self.assertGreater(good.current.pathway_strength("regulation"),
                           bad.current.pathway_strength("regulation"))

    def test_avoidance_moves_away_from_target(self):
        s = Simulation()
        a0 = s.alignment
        for _ in range(15):
            s.log_event("avoidance")
            s.advance_day()
        self.assertLess(s.alignment, a0)

    def test_threat_target_is_not_zero(self):
        self.assertGreater(PATHWAYS["threat"].target, 0.3,
                           "a silent threat system is not the goal")

    def test_weights_stay_bounded(self):
        s = Simulation()
        s.simulate_scripted(weeks=40, adherence=0.95)
        for c in s.current.conns:
            self.assertTrue(0.0 <= c.w <= 1.0)
            self.assertTrue(0.0 <= c.c <= 1.0)

    def test_target_brain_is_perfectly_aligned(self):
        self.assertAlmostEqual(alignment(Connectome.target_brain()), 1.0,
                               places=6)

    def test_replay_history_is_monotonic_in_days(self):
        s = Simulation()
        s.simulate_scripted(weeks=4)
        days = [h.day for h in s.history]
        self.assertEqual(days, sorted(days))
        self.assertEqual(s.snapshot_at(10).day, 10)

    def test_roundtrip_save_load(self):
        import os, tempfile
        s = Simulation()
        s.simulate_scripted(weeks=3)
        p = os.path.join(tempfile.mkdtemp(), "s.json")
        s.save(p)
        s2 = Simulation.load(p)
        self.assertAlmostEqual(s.alignment, s2.alignment, places=6)
        self.assertEqual(len(s.log), len(s2.log))


if __name__ == "__main__":
    unittest.main(verbosity=2)
