"""
How you tend to react - the starting point for the Current tab.

Without this, every user starts from the same default brain, and "your
current brain" is really everyone's until weeks of logging have moved it.

THE BOUNDARY IS THE SAME ONE interpret.py DEFENDS
-------------------------------------------------
An answer never sets a weight. Each option names existing event types and
how often a week they happen; those events are replayed for a few weeks
through the ordinary engine, under the ordinary fixed rules, and the result
becomes day 0. The questions choose which rules run - nothing else.

These are everyday situations written for this app, not a validated
instrument, and the UI says so. Nothing here is an assessment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .engine import Simulation
from .events import EVENTS_BY_ID

WEEKS = 6

# Per-week frequencies are how often the replayed history runs that event.
# Invented, like every constant in the model; chosen so a strong tendency
# (3/week) clearly moves its pathways and a mild one (1/week) only nudges.
QUESTIONS: List[Dict[str, Any]] = [
    {"id": "criticised",
     "prompt": "Someone criticises your work in front of others.",
     "options": [
         {"id": "replay", "label": "I replay it for hours, sometimes days",
          "events": {"rumination": 3.0}},
         {"id": "snap", "label": "I snap back or say something I regret",
          "events": {"reactive_outburst": 2.0}},
         {"id": "avoid", "label": "I go quiet and avoid them afterwards",
          "events": {"avoidance": 2.0}},
         {"id": "useful",
          "label": "I notice it stings, then look for what is useful in it",
          "events": {"name_emotion": 1.5, "reappraisal": 1.5}},
     ]},
    {"id": "plans",
     "prompt": "Plans fall through at the last minute.",
     "options": [
         {"id": "irritated",
          "label": "I get irritated and it colours the rest of my day",
          "events": {"reactive_outburst": 1.0, "stress_high": 1.5}},
         {"id": "chew",
          "label": "I say it is fine but keep chewing on it",
          "events": {"rumination": 2.0}},
         {"id": "redirect",
          "label": "I pause, then find something else worth doing",
          "events": {"pause": 1.5, "behavioural_activation": 1.5}},
         {"id": "shrug", "label": "I shrug it off",
          "events": {"regulated_success": 2.0}},
     ]},
    {"id": "dreaded",
     "prompt": "There is a task you have been dreading.",
     "options": [
         {"id": "putoff", "label": "I put it off until the last moment",
          "events": {"avoidance": 3.0}},
         {"id": "brood", "label": "I think about it constantly but do not start",
          "events": {"rumination": 2.0, "avoidance": 1.5}},
         {"id": "anyway", "label": "I start anyway, even while anxious",
          "events": {"exposure": 1.5, "behavioural_activation": 1.5}},
         {"id": "steps", "label": "I break it into small steps and do one",
          "events": {"behavioural_activation": 2.5}},
     ]},
    {"id": "night",
     "prompt": "Lying in bed after a hard day.",
     "options": [
         {"id": "replay", "label": "My mind replays everything that went wrong",
          "events": {"rumination": 2.5, "sleep_poor": 2.0}},
         {"id": "scroll", "label": "I scroll my phone until very late",
          "events": {"sleep_poor": 3.0}},
         {"id": "calm", "label": "I do something calming - breathing, reading",
          "events": {"mindful_practice": 2.0, "sleep_good": 2.0}},
         {"id": "sleep", "label": "I fall asleep fairly easily",
          "events": {"sleep_good": 4.0}},
     ]},
    {"id": "cutoff",
     "prompt": "Someone cuts in front of you, in traffic or in a queue.",
     "options": [
         {"id": "shout", "label": "I shout or gesture before I have thought",
          "events": {"reactive_outburst": 2.0}},
         {"id": "stew", "label": "I stay angry and bring it up later",
          "events": {"rumination": 2.0, "reactive_outburst": 1.0}},
         {"id": "letgo", "label": "I feel the flash of anger, then let it go",
          "events": {"name_emotion": 1.0, "disengage": 2.0}},
         {"id": "barely", "label": "I barely notice",
          "events": {"regulated_success": 1.5}},
     ]},
    {"id": "feeling",
     "prompt": "When you notice a strong feeling building.",
     "options": [
         {"id": "takes", "label": "It takes over before I know what it is",
          "events": {"reactive_outburst": 1.5, "stress_high": 1.0}},
         # Suppression has no event of its own; avoidance plus raised stress
         # is the nearest thing the rules can express.
         {"id": "push", "label": "I push it down and carry on",
          "events": {"avoidance": 1.0, "stress_high": 1.0}},
         {"id": "name", "label": "I can usually name it - 'this is frustration'",
          "events": {"name_emotion": 3.0}},
         {"id": "breath", "label": "I take a breath before doing anything",
          "events": {"pause": 3.0}},
     ]},
    {"id": "week",
     "prompt": "A typical week for you.",
     "options": [
         {"id": "pressure", "label": "Constant pressure and deadlines",
          "events": {"stress_high": 3.0}},
         {"id": "lowenergy", "label": "Low energy, hard to get going",
          "events": {"avoidance": 1.0, "sleep_poor": 1.0}},
         {"id": "busy", "label": "Busy, with some breathing room",
          "events": {"stress_high": 1.0, "recovery": 1.0}},
         {"id": "calm", "label": "Fairly calm, time with people I like",
          "events": {"recovery": 3.0}},
     ]},
    {"id": "move",
     "prompt": "How often you move your body - a walk, a run, the gym.",
     "options": [
         {"id": "rarely", "label": "Rarely", "events": {}},
         {"id": "sometimes", "label": "Once or twice a week",
          "events": {"exercise": 1.5}},
         {"id": "often", "label": "Most days", "events": {"exercise": 4.0}},
     ]},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}

# How each answer reads after "because you ..." in an explanation. Quoting
# the option label instead gave lines like "I think about it constantly but
# do not start" under vlPFC, which made no sense out of context.
BECAUSE: Dict[str, str] = {
    "criticised/replay": "replay criticism for hours",
    "criticised/snap": "snap back when criticised",
    "criticised/avoid": "withdraw after criticism",
    "criticised/useful": "look for what is useful in criticism",
    "plans/irritated": "let irritation colour the rest of the day",
    "plans/chew": "keep chewing on setbacks",
    "plans/redirect": "pause and redirect when plans fall through",
    "plans/shrug": "shrug off setbacks",
    "dreaded/putoff": "put off tasks you dread",
    "dreaded/brood": "worry about tasks without starting them",
    "dreaded/anyway": "start dreaded tasks anyway",
    "dreaded/steps": "break hard tasks into small steps",
    "night/replay": "go over the day's mistakes at night",
    "night/scroll": "stay up late scrolling",
    "night/calm": "wind down with something calming",
    "night/sleep": "fall asleep easily",
    "cutoff/shout": "react before thinking when provoked",
    "cutoff/stew": "stay angry and bring it up later",
    "cutoff/letgo": "feel the anger, then let it go",
    "cutoff/barely": "barely notice provocations",
    "feeling/takes": "get swept up before you know what you feel",
    "feeling/push": "push feelings down",
    "feeling/name": "put names to your feelings",
    "feeling/breath": "take a breath before reacting",
    "week/pressure": "live with constant pressure",
    "week/lowenergy": "often feel low on energy",
    "week/busy": "have busy but manageable weeks",
    "week/calm": "have calm weeks with people you like",
    "move/rarely": "rarely exercise",
    "move/sometimes": "exercise once or twice a week",
    "move/often": "exercise most days",
}

# How too LITTLE of an event reads, for parts pulled by what someone does not
# do. Keyed by event id.
RARELY: Dict[str, str] = {
    "name_emotion": "rarely put names to feelings",
    "reappraisal": "rarely reframe things",
    "pause": "rarely pause before reacting",
    "disengage": "rarely let a thought go",
    "exposure": "rarely face what makes you anxious",
    "regulated_success": "rarely stay steady in the moment",
    "mindful_practice": "rarely do calming practice",
    "behavioural_activation": "rarely start things you do not feel like doing",
    "rumination": "rarely go over things again and again",
    "avoidance": "rarely avoid things",
    "reactive_outburst": "rarely lash out",
    "sleep_good": "rarely sleep well",
    "sleep_poor": "rarely sleep badly",
    "exercise": "rarely exercise",
    "stress_high": "rarely have high-stress days",
    "recovery": "rarely get downtime",
}

# Events per week for an event read from a described pattern, at intensity
# 1.0. Someone volunteering "I do X" is describing a habit, so it weighs like
# a strong answer; measured, 2-3/week on top of the typical mix moved no part
# past the engagement threshold, 4/week did.
TEXT_RATE = 4.0
MAX_NOTES = 12


def note_from(text: str, proposals: List[Any]) -> Optional[Dict[str, Any]]:
    """A described pattern plus what the classifier read it as, or None.

    Only known event ids survive; the rate scales with the intensity the
    classifier reported, so "I sometimes ..." weighs less than "I always".
    """
    events: Dict[str, float] = {}
    for p in proposals[:3]:
        if p.event in EVENTS_BY_ID:
            events[p.event] = round(TEXT_RATE * float(p.intensity), 2)
    if not events:
        return None
    return {"text": text.strip()[:400], "events": events}


def notes_of(profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The profile's described patterns, including the pre-list format."""
    if not profile:
        return []
    notes = list(profile.get("notes") or [])
    old = [e for e in profile.get("text_events") or [] if e in EVENTS_BY_ID]
    if old and not notes:
        notes.append({"text": profile.get("text", ""),
                      "events": {e: TEXT_RATE for e in old}})
    return notes


def _picked(v: Any) -> List[str]:
    # One id (the original single-choice format) or a list of them.
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def chosen(answers: Dict[str, Any]) -> List[Tuple[Dict, Dict, float]]:
    """(question, option, weight) for every answer that names a real option.

    Several options can be picked per situation. Each then counts for
    1/sqrt(n): picking two means both happen, but neither as often as if it
    were the only thing you do.

    Unknown ids are dropped, not coerced, for the same reason unknown event
    ids from a model are.
    """
    out = []
    for qid, val in (answers or {}).items():
        q = _BY_ID.get(str(qid))
        if not q:
            continue
        opts = [o for o in q["options"] if o["id"] in _picked(val)]
        for o in opts:
            out.append((q, o, 1.0 / math.sqrt(len(opts))))
    return out


def _question_mix(q: Dict[str, Any]) -> Dict[str, float]:
    """An even mix of every option to one question - how a typical person
    is taken to answer it."""
    mix: Dict[str, float] = {}
    for o in q["options"]:
        for eid, f in o["events"].items():
            mix[eid] = mix.get(eid, 0.0) + f / len(q["options"])
    return mix


def typical_rates() -> Dict[str, float]:
    """Events per week for the invented typical person: every question
    answered with an even mix of its options."""
    out: Dict[str, float] = {}
    for q in QUESTIONS:
        for eid, f in _question_mix(q).items():
            out[eid] = out.get(eid, 0.0) + f
    return out


def profile_rates(answers: Dict[str, Any],
                  notes: Optional[List[Dict[str, Any]]] = None
                  ) -> Dict[str, float]:
    """Events per week for this person.

    A question left unanswered counts as typical, so skipping it moves
    nothing; only what someone actually says pulls a part away from the
    reference.
    """
    answered = {q["id"] for q, _, _ in chosen(answers)}
    rate = rates(sources(answers, notes))
    for q in QUESTIONS:
        if q["id"] not in answered:
            for eid, f in _question_mix(q).items():
                rate[eid] = rate.get(eid, 0.0) + f
    return rate


def sources(answers: Dict[str, Any],
            notes: Optional[List[Dict[str, Any]]] = None
            ) -> List[Dict[str, Any]]:
    """Each answer and described pattern as a labelled bundle of per-week
    event rates."""
    out = [{"label": BECAUSE.get(f"{q['id']}/{opt['id']}", opt["label"]),
            "question": q["prompt"],
            "events": {e: f * w for e, f in opt["events"].items()}}
           for q, opt, w in chosen(answers) if opt["events"]]
    for n in notes or []:
        ev = {e: float(f) for e, f in (n.get("events") or {}).items()
              if e in EVENTS_BY_ID}
        if ev:
            out.append({"label": f"said \"{_snippet(n.get('text', ''))}\"",
                        "question": "", "events": ev})
    return out


def _snippet(text: str, n: int = 48) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def rates(srcs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Total events per week across all sources."""
    rate: Dict[str, float] = {}
    for s in srcs:
        for eid, f in s["events"].items():
            rate[eid] = rate.get(eid, 0.0) + f
    return rate


def replay(sim: Simulation, rate: Dict[str, float], days: int) -> None:
    """Run `rate` (events per week) through `sim` for `days`, evenly spread."""
    for d in range(days):
        for eid in sorted(rate):
            f = rate[eid]
            for _ in range(math.floor((d + 1) * f / 7) - math.floor(d * f / 7)):
                sim.log_event(eid)
        sim.advance_day()


def build(answers: Dict[str, str],
          notes: Optional[List[Dict[str, Any]]] = None,
          seed: int = 0, log: Optional[List[Any]] = None,
          day: int = 0) -> Simulation:
    """Replay the described habits for WEEKS, then call the result day 0.

    Deterministic: the schedule is spread evenly rather than drawn at
    random, and the engine is seeded, so the same answers always give the
    same brain.

    `log` and `day` carry over what was logged since: refining the
    description changes the starting point, and everything logged after it
    is replayed on top rather than thrown away.
    """
    notes = [n for n in (notes or []) if n.get("events")][:MAX_NOTES]
    rate = profile_rates(answers, notes)

    sim = Simulation(seed=seed)
    days = WEEKS * 7
    replay(sim, rate, days)

    picked: Dict[str, List[str]] = {}
    for q, opt, _ in chosen(answers):
        picked.setdefault(q["id"], []).append(opt["id"])
    sim.rebase(days, {
        "answers": picked,
        "notes": notes,
        "weeks": WEEKS,
    })
    entries = list(log or [])
    for d in range(day + 1):
        for e in entries:
            if e.day == d and e.event_id in EVENTS_BY_ID:
                sim.log_event(e.event_id, e.note, e.intensity)
        if d < day:
            sim.advance_day()
    return sim
