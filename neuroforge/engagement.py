"""
Which parts of the brain are more or less engaged than typical.

MODEL, not measurement. A part's engagement here is the mean change, against
a reference brain, of the connections it takes part in:

  * a connection it DRIVES counts for it (the part is doing that work),
  * a connection that EXCITES it counts for it,
  * a connection that CALMS or IMPAIRS it counts against it - regulation
    onto the amygdala, and high arousal onto dlPFC.

The reference is what a typical person looks like after the same number of
days: someone who answers every situation with an even mix of its options
(reactions.typical_rates), then logs nothing. A question the user skips
counts as that same mix, so skipping moves nothing. Comparing against an
idle brain instead made nearly every part read "more engaged", because in
this model any use triggers homeostasis that tops connections back up
while idle ones decay. It is an invented typical, not a population norm,
and more engaged is not better or worse: that is the Target tab's question.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from .anatomy.anchors import NODE_TO_CORTEX, NODE_TO_STRUCTURE
from .atlas import EDGES
from .events import EVENTS_BY_ID

# Edges whose strength works AGAINST their destination's activity.
DAMPENS = {
    ("vlPFC", "BLA"), ("vmPFC", "BLA"),
    # atlas.py: moderate noradrenergic tone helps dlPFC, high tone impairs it.
    ("LC", "dlPFC"),
}

# Mean weight change beyond which a part reads as more / less engaged.
# Calibrated on contrasting profiles, where parts range to about +-0.5.
THRESHOLD = 0.04

# A listed reason must pull at least this share of the strongest one.
REASON_SHARE = 0.35

PARTS: Dict[str, List[str]] = {}
for _node, _part in {**NODE_TO_STRUCTURE, **NODE_TO_CORTEX}.items():
    PARTS.setdefault(_part, []).append(_node)

# What more / less engagement of each part means IN THIS MODEL, worded from
# the connections it actually has in atlas.py - not general claims about the
# region. (more, less)
MEANING: Dict[str, tuple] = {
    "amygdala": ("flags things as threatening sooner and holds on longer",
                 "raises the alarm less readily"),
    "acc": ("keeps signalling that something is wrong, feeding the loop of "
            "going over it",
            "sends fewer 'something is wrong' signals"),
    "hippocampus": ("brings past experience into how situations are read "
                    "more strongly",
                    "supplies less context, so 'this is safe now' is harder "
                    "to register"),
    "brainstem": ("drives more bodily arousal - tension, a racing heart",
                  "drives less bodily arousal"),
    "vlPFC": ("puts the brakes on and names feelings more readily",
              "brakes and names feelings less in the moment"),
    "vmPFC": ("signals 'this is actually safe' more strongly",
              "signals safety less, so worry is harder to switch off"),
    "dlPFC": ("holds goals in mind and steers deliberately more",
              "steers deliberately less, so reactions run on autopilot"),
    "caudate": ("chooses actions deliberately, checking they are worth it",
                "less deliberate, goal-checked action"),
    "putamen": ("runs more routines on autopilot, stress-driven or practised",
                "runs fewer automatic routines, old or new"),
    "accumbens": ("reinforces a new routine more strongly",
                  "reinforces new routines less, so they take longer to "
                  "stick"),
    "vta": ("sends a stronger 'that went better than expected' signal",
            "sends a weaker teaching signal, so new habits stick less"),
    "thalamus": ("lets more chosen or automatic actions through",
                 "lets fewer actions through"),
}


@lru_cache(maxsize=512)
def _reference(days: int, warmup: int) -> Dict[str, float]:
    from .engine import Simulation
    from .reactions import replay, typical_rates
    sim = Simulation(seed=0)
    if warmup > 0:
        replay(sim, typical_rates(), warmup)
    if days > warmup:
        sim.advance_day(days - warmup)
    return sim.current.snapshot()


def reference(days: int, warmup: int = 0) -> Dict[str, float]:
    return dict(_reference(max(0, int(days)), max(0, int(warmup))))


def _signs(node: str):
    """(edge key, +1/-1) for every edge touching `node`."""
    for e in EDGES:
        k = f"{e.src}->{e.dst}"
        if e.src == node:
            yield k, e.pathway, 1.0
        if e.dst == node:
            yield k, e.pathway, -1.0 if (e.src, e.dst) in DAMPENS else 1.0


def node_scores(weights: Dict[str, float], days: int,
                warmup: int = 0) -> Dict[str, float]:
    ref = _reference(max(0, int(days)), max(0, int(warmup)))
    nodes = {n for e in EDGES for n in (e.src, e.dst)}
    out = {}
    for n in nodes:
        vals = [s * (weights.get(k, ref[k]) - ref[k]) for k, _, s in _signs(n)]
        out[n] = sum(vals) / len(vals) if vals else 0.0
    return out


def _pull(part: str, events: Dict[str, float]) -> float:
    """Which way, and how hard, a bundle of events pushes this part."""
    total = 0.0
    for node in PARTS[part]:
        for _, pid, s in _signs(node):
            for eid, n in events.items():
                ev = EVENTS_BY_ID.get(eid)
                if ev:
                    total += n * s * ev.activations.get(pid, 0.0)
    return total


@lru_cache(maxsize=None)
def _levers(part: str) -> Dict[str, List[str]]:
    """The events that push this part up or down hardest, per occurrence."""
    pulls = [(_pull(part, {eid: 1.0}), eid) for eid in EVENTS_BY_ID]
    out = {}
    for key, sign in (("up", 1.0), ("down", -1.0)):
        ranked = sorted(((sign * p, eid) for p, eid in pulls if sign * p > 0),
                        reverse=True)
        top = ranked[0][0] if ranked else 0.0
        out[key] = [eid for p, eid in ranked if p >= REASON_SHARE * top][:3]
    return out


# A moment's push on a part must be at least this share of the strongest to
# be shown. Low because the amygdala, with the most connections, dominates:
# at 0.2 a rumination moment showed nothing but it and the ACC.
MOMENT_SHARE = 0.06

# Responses a moment can be compared against.
RESPONSES = ("name_emotion", "pause", "disengage", "reappraisal",
             "mindful_practice", "regulated_success")


def moves_for(events: Dict[str, float], limit: int = 6) -> List[Dict[str, Any]]:
    """Which way, and how hard, this bundle of events pushes each part -
    the same signed rules as the Current read-out, for one moment."""
    pulls = [(p, part) for part in PARTS if (p := _pull(part, events))]
    top = max((abs(p) for p, _ in pulls), default=0.0)
    keep = sorted((x for x in pulls if abs(x[0]) >= MOMENT_SHARE * top),
                  key=lambda x: -abs(x[0]))[:limit]
    out = []
    for p, part in keep:
        more = p > 0
        pair = MEANING.get(part, ("", ""))
        out.append({"part": part, "dir": "more" if more else "less",
                    "strength": round(abs(p) / top, 3),
                    "meaning": pair[0 if more else 1]})
    return out


def alternatives(event_ids: List[str], k: int = 3) -> List[Dict[str, Any]]:
    """Responses whose rules push hardest against what this moment pushes."""
    now = {e: 1.0 for e in event_ids if e in EVENTS_BY_ID}
    if not now:
        return []
    base = {part: _pull(part, now) for part in PARTS}
    scored = []
    for alt in RESPONSES:
        if alt in now:
            continue
        counter = sum(-b * _pull(part, {alt: 1.0}) for part, b in base.items())
        if counter > 0:
            scored.append((counter, alt))
    scored.sort(reverse=True)
    return [{"event": a, "label": EVENTS_BY_ID[a].label,
             "moves": moves_for({a: 1.0})} for _, a in scored[:k]]


def effects(events: Dict[str, float], limit: int = 4) -> List[Dict[str, Any]]:
    """Which parts a bundle of events pushes, and which way, strongest
    first. Same rules as the reasons, so a note and the rows agree."""
    pulls = [(p, part) for part in PARTS if (p := _pull(part, events))]
    top = max((abs(p) for p, _ in pulls), default=0.0)
    keep = sorted((x for x in pulls if abs(x[0]) >= REASON_SHARE * top),
                  key=lambda x: -abs(x[0]))[:limit]
    return [{"part": part, "dir": "more" if p > 0 else "less"}
            for p, part in keep]


def engagement(weights: Dict[str, float], days: int, warmup: int = 0,
               sources: Optional[List[Dict[str, Any]]] = None,
               rates: Optional[Dict[str, float]] = None
               ) -> List[Dict[str, Any]]:
    """Per-part engagement, strongest first, with what drove it.

    `days` counts from the start of any warm-up. `sources` are labelled
    event bundles - profile answers and logged events - and `rates` the
    profile's events per week; both are used only to explain a result,
    never to compute it.
    """
    ns = node_scores(weights, days, warmup)
    rows = []
    for part, nodes in PARTS.items():
        score = sum(ns[n] for n in nodes) / len(nodes)
        level = ("more" if score > THRESHOLD
                 else "less" if score < -THRESHOLD else "typical")
        why: Dict[str, Any] = {}
        if level != "typical":
            want = 1.0 if level == "more" else -1.0
            scored = sorted(((want * _pull(part, s["events"]), i)
                             for i, s in enumerate(sources or [])),
                            reverse=True)
            # A reason has to carry real weight, or a side effect such as
            # exercise's small goal-directed nudge gets listed under the ACC
            # beside the rumination that is actually driving it.
            top = scored[0][0] if scored else 0.0
            picked = [sources[i] for p, i in scored
                      if p > 0 and p >= REASON_SHARE * top][:2]
            reasons = [s["label"] for s in picked]
            refs = [s.get("ref", "") for s in picked]
            evidence = {"answer": 0, "note": 0, "log": 0}
            for p, i in scored:
                kind = sources[i].get("ref", "").split(":")[0]
                if p > 0 and kind in evidence:
                    evidence[kind] += sources[i].get("count", 1)
            # A part can also sit low (or high) because of what someone does
            # LESS of than typical - no single answer says so.
            if rates is not None and warmup > 0:
                from .reactions import RARELY, typical_rates
                typ = typical_rates()
                gaps = []
                for eid in EVENTS_BY_ID:
                    gap = typ.get(eid, 0.0) - rates.get(eid, 0.0)
                    c = -want * _pull(part, {eid: 1.0}) * gap
                    if gap > 0 and c > 0 and eid in RARELY:
                        gaps.append((c, RARELY[eid], f"rarely:{eid}"))
                gaps.sort(reverse=True)
                gtop = gaps[0][0] if gaps else 0.0
                extra = [g for g in gaps if g[0] >= REASON_SHARE * gtop
                         ][:3 - len(reasons)]
                reasons += [g[1] for g in extra]
                refs += [g[2] for g in extra]
            pair = MEANING.get(part)
            why = {"meaning": pair[0 if want > 0 else 1] if pair else "",
                   "reasons": reasons, "refs": refs, "evidence": evidence}
        # A typical score means two different things: what was said pulls
        # this part both ways with real weight, so the pulls cancel; or it
        # barely reaches it, and the model simply has too little to go on.
        # Per event, not per source, so one note that does both still counts.
        basis = ""
        if level == "typical":
            pulls = [_pull(part, {eid: n}) for s in sources or []
                     for eid, n in s["events"].items()]
            up = sum(p for p in pulls if p > 0)
            down = -sum(p for p in pulls if p < 0)
            basis = ("balanced" if up > 0 and down > 0
                     and min(up, down) >= REASON_SHARE * max(up, down)
                     else "unknown")
        rows.append({"part": part, "nodes": nodes, "score": round(score, 4),
                     "level": level, "basis": basis, "why": why,
                     "moves": _levers(part)})
    rows.sort(key=lambda r: -abs(r["score"]))
    return rows


def trend(history: List[Any], warmup: int, upto: int,
          span: int = 28) -> Dict[str, Any]:
    """Each part's score on each recorded day up to `upto`, last `span`
    days. Same reference as the read-out, so it is today's row, over time."""
    by_day: Dict[int, Dict[str, float]] = {}
    for h in history:
        if h.day <= upto:
            by_day[h.day] = h.weights
    days = sorted(by_day)[-span:]
    parts: Dict[str, List[float]] = {p: [] for p in PARTS}
    for d in days:
        ns = node_scores(by_day[d], warmup + d, warmup)
        for part, nodes in PARTS.items():
            parts[part].append(round(sum(ns[n] for n in nodes) / len(nodes), 4))
    return {"days": days, "parts": parts}


def logged_sources(log: List[Any]) -> List[Dict[str, Any]]:
    """Logged events grouped by type, as explanation sources."""
    counts: Dict[str, int] = {}
    for e in log:
        counts[e.event_id] = counts.get(e.event_id, 0) + 1
    return [{"label": f"logged {EVENTS_BY_ID[eid].label.lower()} "
                      f"{n} time{'s' if n != 1 else ''}",
             "ref": f"log:{eid}", "count": n,
             "events": {eid: float(n)}}
            for eid, n in counts.items() if eid in EVENTS_BY_ID]
