"""Ask retrieval: which parts and knowledge a question pulls in."""

import os
import unittest
from unittest import mock

os.environ.setdefault("NEUROFORGE_LLM", "off")

from neuroforge import llm
from neuroforge.anatomy.structures import CORTICAL_REGIONS, GROUPS, STRUCTURES


def scene_part(part):
    return {
        "id": part.id, "name": part.name, "short": part.short,
        "knowledge": {
            "what_it_is": part.what_it_is,
            "where_it_is": part.where_it_is,
            "contributes_to": part.contributes_to,
            "in_the_model": part.in_the_model,
            "caveat": part.caveat,
        },
    }

SCENE = {
    "cortical_regions": [scene_part(r) for r in CORTICAL_REGIONS],
    "structures": [scene_part(s) for s in STRUCTURES],
    "groups": GROUPS,
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

    def test_prefrontal_family_includes_all_available_members(self):
        parts, _ = ask("what does the prefrontal cortex do?")
        group = next(group for group in GROUPS if group["id"] == "PFC")
        self.assertTrue(set(group["regions"]) <= set(parts))

    def test_explicit_parts_and_groups_outrank_topic_hints(self):
        question = ("prefrontal cortex and amygdala reward motivation "
                    "dopamine pleasure")
        parts, _ = ask(question)
        group = next(group for group in GROUPS if group["id"] == "PFC")
        self.assertEqual(parts[0], "amygdala")
        self.assertTrue(set(group["regions"]) <= set(parts))

    def test_no_match(self):
        self.assertEqual(ask("what does the claustrum do?"), ([], []))
        ctx, _ = llm.build_context("what does the claustrum do?", SCENE)
        self.assertIn("No specific region matched", ctx)

    def test_knowledge_only_still_grounds(self):
        ctx, _ = llm.build_context("how is progress measured?", SCENE)
        self.assertNotIn("No specific region matched", ctx)
        self.assertIn("concept: progress", ctx)

    def test_context_uses_retrieved_parts_and_reference_text(self):
        question = "why would bad sleep make my reappraisal stick less?"
        ctx, valid_ids = llm.build_context(question, SCENE)
        self.assertIn("### vmPFC", ctx)
        self.assertIn("what it is:", ctx)
        self.assertIn("### event: sleep and consolidation", ctx)
        self.assertIn("Science: Sleep supports consolidation", ctx)
        self.assertIn("Good sleep rule: sets sleep quality = 0.92", ctx)
        self.assertIn("Poor sleep rule: sets sleep quality = 0.30", ctx)
        self.assertIn("amygdala", valid_ids)

    def test_progress_context_includes_the_simulation_value(self):
        state = {"day": 7, "alignment": 0.42, "progress": 0.314159,
                 "stress": 0.3, "sleep": 0.7, "pathways": {}}
        ctx, _ = llm.build_context("how is progress measured?", SCENE, state)
        self.assertIn("simulated progress 0.314159", ctx)
        self.assertIn("current distance / initial distance", ctx)
        self.assertIn("not a measurement", ctx)

    def test_specific_group_does_not_expand_to_the_whole_family(self):
        parts, _ = ask("what does the dorsal striatum do?")
        self.assertEqual(set(parts), {"caudate", "putamen"})
        parts, titles = ask("what does the ventral striatum do?")
        self.assertEqual(set(parts), {"accumbens"})
        self.assertFalse(any("routine" in title for title in titles))
        parts, _ = ask("what does the lateral prefrontal cortex do?")
        self.assertEqual(set(parts), {"dlPFC", "vlPFC"})
        parts, _ = ask("what does the anterior cingulate do?")
        self.assertEqual(set(parts), {"dACC", "sgACC", "acc"})
        parts, _ = ask("what does the posterior cingulate do?")
        self.assertEqual(set(parts), {"PCC"})

    def test_groups_work_when_scene_omits_group_metadata(self):
        scene = {key: SCENE[key] for key in ("cortical_regions", "structures")}
        parts, _ = llm.retrieve("what is the prefrontal cortex?", scene)
        self.assertTrue({"dlPFC", "vlPFC", "vmPFC", "mPFC"}
                        <= {part["id"] for part in parts})

    def test_retrieval_and_highlighting_use_the_model_question_limit(self):
        question = ("x" * llm.MAX_QUESTION) + " amygdala and progress"
        parts, knowledge = llm.retrieve(question, SCENE)
        self.assertEqual(parts, [])
        self.assertEqual(knowledge, [])
        self.assertEqual(llm.asked_ids(question, SCENE), [])

    def test_context_budget_and_valid_ids(self):
        question = "sleep, reappraisal, rumination, habits and progress"
        state = {"day": 7, "alignment": 0.42, "progress": 0.314159,
                 "stress": 0.3, "sleep": 0.7, "pathways": {}}
        for budget in (0, 256, 2000, 7000):
            with self.subTest(budget=budget):
                ctx, valid_ids = llm.build_context(question, SCENE, state,
                                                   char_budget=budget)
                self.assertLessEqual(len(ctx), budget)
                self.assertIn("dlPFC", valid_ids)
        ctx, _ = llm.build_context(question, SCENE, state)
        self.assertIn("simulated progress 0.314159", ctx)
        self.assertIn("### concept: progress", ctx)
        with self.assertRaises(ValueError):
            llm.build_context(question, SCENE, char_budget=-1)

    def test_chat_route_passes_grounded_context_to_the_model(self):
        with mock.patch.dict(os.environ, {"NEUROFORGE_AUTOSAVE": "0"}):
            from neuroforge.server import routes

        state = {"day": 7, "alignment": 0.42, "progress": 0.314159,
                 "stress": 0.3, "sleep": 0.7, "pathways": {}}
        config = {"enabled": True, "model": "test", "local": True}
        handler = mock.Mock()
        with mock.patch.object(routes, "SCENE", SCENE), \
             mock.patch.object(routes, "_state", return_value=state), \
             mock.patch.object(routes, "_model", return_value={}), \
             mock.patch.object(routes, "public_chat_config", return_value=config), \
             mock.patch.object(routes, "stream_chat",
                               return_value=iter(["Simulated progress."])) as stream:
            routes._chat({"text": "how is progress measured?"}, handler)

        context = stream.call_args.args[1]
        self.assertIn("### concept: progress", context)
        self.assertIn("simulated progress 0.314159", context)
        handler.sse.assert_any_call(
            "done", {"text": "Simulated progress.", "ids": [], "asked": []})


if __name__ == "__main__":
    unittest.main()
