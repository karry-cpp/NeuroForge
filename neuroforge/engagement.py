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


def _touches(part: str, events: Dict[str, float]) -> bool:
    """Whether any of these events acts on a connection of this part at all,
    in either direction - so opposite pushes cannot cancel to 'untouched'."""
    for node in PARTS[part]:
        for _, pid, _ in _signs(node):
            for eid in events:
                ev = EVENTS_BY_ID.get(eid)
                if ev and ev.activations.get(pid, 0.0):
                    return True
    return False


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
            scored = sorted(((want * _pull(part, s["events"]), s["label"])
                             for s in sources or []), reverse=True)
            # A reason has to carry real weight, or a side effect such as
            # exercise's small goal-directed nudge gets listed under the ACC
            # beside the rumination that is actually driving it.
            top = scored[0][0] if scored else 0.0
            reasons = [lbl for p, lbl in scored
                       if p > 0 and p >= REASON_SHARE * top][:2]
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
                        gaps.append((c, RARELY[eid]))
                gaps.sort(reverse=True)
                gtop = gaps[0][0] if gaps else 0.0
                reasons += [p for c, p in gaps if c >= REASON_SHARE * gtop
                            ][:3 - len(reasons)]
            pair = MEANING.get(part)
            why = {"meaning": pair[0 if want > 0 else 1] if pair else "",
                   "reasons": reasons}
        # A typical score means two different things: what was said pulls
        # this part both ways (or about as much as typical), or nothing said
        # reaches it at all and the model simply has no information.
        basis = ""
        if level == "typical":
            basis = ("balanced" if any(_touches(part, s["events"])
                                       for s in sources or [])
                     else "unknown")
        rows.append({"part": part, "nodes": nodes, "score": round(score, 4),
                     "level": level, "basis": basis, "why": why})
    rows.sort(key=lambda r: -abs(r["score"]))
    return rows


def logged_sources(log: List[Any]) -> List[Dict[str, Any]]:
    """Logged events grouped by type, as explanation sources."""
    counts: Dict[str, int] = {}
    for e in log:
        counts[e.event_id] = counts.get(e.event_id, 0) + 1
    return [{"label": f"logged {EVENTS_BY_ID[eid].label.lower()} "
                      f"{n} time{'s' if n != 1 else ''}",
             "events": {eid: float(n)}}
            for eid, n in counts.items() if eid in EVENTS_BY_ID]
