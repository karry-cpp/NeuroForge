"""Ask retrieval: which parts and knowledge a question pulls in."""

import os
import unittest

os.environ.setdefault("NEUROFORGE_LLM", "off")

from neuroforge import llm
from neuroforge.anatomy.structures import CORTICAL_REGIONS, STRUCTURES

SCENE = {
    "cortical_regions": [{"id": r.id, "name": r.name, "short": r.short}
                         for r in CORTICAL_REGIONS],
    "structures": [{"id": s.id, "name": s.name, "short": s.short}
                   for s in STRUCTURES],
}


def ask(q):
    parts, knowledge = llm.retrieve(q, SCENE)
    return [p["id"] for p in parts], [t for t, _ in knowledge]


# question, parts that must appear, parts that must not, knowledge title words
CASES = [
    ("why would bad sleep make my reappraisal stick less?",
     {"vmPFC"}, set(), {"sleep", "consolidation"}),
    ("so my prefrontal cortex grew?",
     {"dlPFC", "vlPFC", "vmPFC", "mPFC"}, set(), set()),
    ("i get rly angry and cant stop replaying it",
     {"amygdala"}, set(), {"Rumination"}),
    ("calm myself down", {"vmPFC"}, {"PCC", "mPFC"}, set()),
    ("my account manager", set(), {"acc", "dACC"}, set()),
    ("what does the cingulate do?", {"dACC", "acc", "sgACC", "PCC"}, set(),
     set()),
    ("is the striatum about habits?", {"caudate", "putamen", "accumbens"},
     set(), {"routine"}),
    ("tell me about the dlPFC", {"dlPFC"}, set(), set()),
]


class RetrievalTest(unittest.TestCase):
    def test_cases(self):
        for q, want, banned, know in CASES:
            with self.subTest(q=q):
                parts, titles = ask(q)
                self.assertTrue(want <= set(parts), (parts, want))
                self.assertFalse(banned & set(parts), parts)
                for word in know:
                    self.assertTrue(any(word in t for t in titles),
                                    (word, titles))

    def test_named_part_outranks_family(self):
        parts, _ = ask("what does the dlPFC add to the prefrontal cortex?")
        self.assertEqual(parts[0], "dlPFC")

    def test_no_match(self):
        self.assertEqual(ask("what does the claustrum do?"), ([], []))
        ctx, _ = llm.build_context("what does the claustrum do?", SCENE)
        self.assertIn("No specific region matched", ctx)

    def test_knowledge_only_still_grounds(self):
        ctx, _ = llm.build_context("how is progress measured?", SCENE)
        self.assertNotIn("No specific region matched", ctx)
        self.assertIn("concept: progress", ctx)


if __name__ == "__main__":
    unittest.main()
