import os, time, json
os.environ["NEUROFORGE_LLM_BASE"] = "http://127.0.0.1:1234/v1"
CASES = [
 ("I skipped the meeting because I was dreading it.", "avoidance"),
 ("My colleague took credit for my work. I paused and named what I was feeling instead of snapping.", "name_emotion/pause"),
 ("I kept replaying the argument in my head all evening.", "rumination"),
 ("I didn't get angry when they cut me off in traffic.", "reappraisal/pause"),
 ("Went to bed at 2am again and slept badly.", "poor_sleep"),
 ("I went for a long run this morning.", "exercise"),
]
MODELS = ["qwen/qwen3-4b"]
lines = []
for m in MODELS:
    os.environ["NEUROFORGE_LLM_MODEL"] = m
    import importlib
    from neuroforge import interpret as I
    importlib.reload(I)
    for text, expect in CASES:
        t0 = time.time()
        try:
            r = I.classify_llm(text, timeout=120)
            got = ", ".join("%s(%s,%.2f)" % (p.event, p.category, p.confidence) for p in r.proposals[:2])
            lines.append("%-26s %5.1fs  expect=%-18s got=%s" % (m, time.time()-t0, expect, got))
        except Exception as e:
            lines.append("%-26s %5.1fs  expect=%-18s FAILED %s: %s" % (m, time.time()-t0, expect, type(e).__name__, str(e)[:70]))
        print(lines[-1], flush=True)
open("_llmbench.txt", "w", encoding="utf-8").write("\n".join(lines))
