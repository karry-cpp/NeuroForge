"""
neuroforge.server.routes
========================

The JSON API. This is the ONLY contract between the Python simulation and
the WebGL renderer, which is what keeps the two independent.

  GET  /api/scene           anatomy: meshes, structures, knowledge
  GET  /api/model           static simulation description (pathways, events)
  GET  /api/sim/state       current simulation state (weights, metrics)
  POST /api/sim/log         record a behaviour
  POST /api/sim/interpret   free text -> suggested events (does NOT apply)
  POST /api/sim/analyse     free text + events -> what the rules engage
  POST /api/sim/advance     advance N days
  GET  /api/sim/projection  where current habits lead (run on a copy)
  POST /api/sim/replay      state at a past day
  POST /api/sim/reset       start over
  POST /api/sim/demo        fast-forward a scripted practice history
  POST /api/sim/save|load   persistence
  GET  /api/profile/questions   the "how you tend to react" situations
  POST /api/profile         answers (+ optional description) -> new day 0
  POST /api/profile/note    add a described pattern; refines day 0
  POST /api/profile/note/delete   remove one
  GET  /api/chat/config     is a model configured, and which one
  POST /api/chat            streamed grounded answer (server-sent events)
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Dict, List

from ..anatomy.anchors import (ANCHORS, NODE_TO_CORTEX, NODE_TO_STRUCTURE,
                               anchor_table)
from ..analyse import analyse
from ..atlas import EDGES, EXCLUSIONS, PATHWAYS, REGIONS
from ..engagement import effects, engagement, logged_sources, trend
from ..engine import Simulation
from ..events import CATEGORIES, EVENT_TYPES
from ..interpret import interpret, llm_config
from ..llm import (asked_ids, build_context, extract_markers, public_chat_config,
                   stream_chat, strip_markers)
from .. import reactions
from .app import API, ClientGone

SCENE: Dict[str, Any] = {}
SIM = Simulation()

# The server is threaded, so several requests can land on SIM at once. SIM is
# mutable shared state: log_event, advance_day and the outright reassignments
# in reset/demo/load are all read-modify-write. Without this, two requests
# arriving together can interleave and produce a day count or a weight that
# never actually occurred.
#
# Reentrant because a guarded handler may call another guarded helper.
#
# Note deliberately NOT guarded: /api/sim/interpret. It never touches SIM and
# it is the slowest endpoint we have (a language model can take tens of
# seconds). Holding this lock across that call would serialise the entire
# application behind it and undo the point of threading in the first place.
STATE_LOCK = threading.RLock()


def guarded(fn):
    """Run a route handler holding STATE_LOCK.

    Apply *below* the @API decorator so the registered callable is the
    locked wrapper, not the bare function.
    """
    @functools.wraps(fn)
    def wrapper(payload: Dict[str, Any]) -> Any:
        with STATE_LOCK:
            return fn(payload)
    return wrapper
SAVE_DIR = os.path.join(os.path.expanduser("~"), ".neuroforge")
AUTOSAVE = os.path.join(SAVE_DIR, "autosave.json")


def _autosave_on() -> bool:
    return os.environ.get("NEUROFORGE_AUTOSAVE", "1") != "0"


def _persist() -> None:
    """Write the session after every change, so a restart loses nothing.

    Written to a temporary file and swapped in, so a crash mid-write cannot
    leave a half-written save that fails to load next time.
    """
    if not _autosave_on():
        return
    tmp = AUTOSAVE + ".tmp"
    try:
        SIM.save(tmp)
        os.replace(tmp, AUTOSAVE)
    except OSError:
        pass


def _restore() -> Simulation:
    if _autosave_on() and os.path.exists(AUTOSAVE):
        try:
            return Simulation.load(AUTOSAVE)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return Simulation()


SIM = _restore()


# ==========================================================================
#  Static description
# ==========================================================================

@API.get("/api/scene")
def _scene(_: Dict[str, Any]) -> Dict[str, Any]:
    return SCENE


@API.get("/api/model")
def _model(_: Dict[str, Any]) -> Dict[str, Any]:
    """Everything about the simulation that never changes."""
    return {
        "pathways": [{
            "id": p.id, "label": p.label, "desirable": p.desirable,
            "target": p.target, "color": p.color,
            "science": p.science, "metaphor": p.metaphor,
        } for p in PATHWAYS.values()],

        "nodes": [{
            "id": r.id, "name": r.name, "group": r.group,
            "role": r.role, "science": r.science, "caveat": r.caveat,
            "anchor": list(ANCHORS[r.id]) if r.id in ANCHORS else None,
            "structure": NODE_TO_STRUCTURE.get(r.id),
            "cortex": NODE_TO_CORTEX.get(r.id),
        } for r in REGIONS.values()],

        "edges": [{
            "id": f"{e.src}->{e.dst}", "src": e.src, "dst": e.dst,
            "pathway": e.pathway, "label": e.label,
            "curve": e.curve, "science": e.science,
            "target": PATHWAYS[e.pathway].target,
        } for e in EDGES],

        "events": [{
            "id": e.id, "label": e.label, "category": e.category,
            "icon": e.icon, "rule": e.rule, "science": e.science,
            "caveat": e.caveat, "activations": e.activations,
            "lr_mult": e.lr_mult, "reward": e.reward,
            "stress_delta": e.stress_delta,
        } for e in EVENT_TYPES],

        "categories": CATEGORIES,
        "anchors": anchor_table(),
        "exclusions": [{"name": n, "why": w} for n, w in EXCLUSIONS],
    }


# ==========================================================================
#  Live state
# ==========================================================================

def _engagement(sim: Simulation, weights: Dict[str, float],
                day: int) -> List[Dict[str, Any]]:
    # Without answers there is nothing about this person to compare - only
    # the default brain plus whatever was logged - so say nothing.
    if not sim.profile:
        return []
    prof = sim.profile
    srcs = reactions.sources(prof.get("answers", {}),
                             reactions.notes_of(prof))
    weeks = prof.get("weeks", reactions.WEEKS)
    # Per-week rates become totals so they weigh sensibly against log counts.
    why = [dict(s, events={k: v * weeks for k, v in s["events"].items()})
           for s in srcs]
    why += logged_sources([e for e in sim.log if e.day <= day])
    start = reactions.profile_rates(prof.get("answers", {}),
                                    reactions.notes_of(prof))
    # "Rarely ..." reasons read today's habits, so logging a practice
    # retires the claim that you rarely do it.
    return engagement(weights, sim.warmup_days + day, sim.warmup_days, why,
                      reactions.habit_rates(start, sim.log, day))


def _target_engagement(sim: Simulation) -> List[Dict[str, Any]]:
    """How the user's target brain compares with the typical one.

    No 'rarely ...' reasons here: an aim nobody mentioned is not a wish to
    do less of it.
    """
    if not sim.target_weights:
        return []
    tp = sim.target_profile or {}
    weeks = tp.get("weeks", reactions.WEEKS)
    srcs = reactions.sources(tp.get("answers", {}), reactions.notes_of(tp))
    why = [dict(s, events={k: v * weeks for k, v in s["events"].items()})
           for s in srcs]
    return engagement(sim.target_weights, weeks * 7, weeks * 7, why)


def _state(sim: Simulation, weights: Dict[str, float] | None = None,
           day: int | None = None) -> Dict[str, Any]:
    """Serialise the simulation. `weights` overrides for replay."""
    w = weights if weights is not None else sim.current.snapshot()
    d = sim.day if day is None else day
    return {
        "day": d,
        "date": str(sim.date_for(d)),
        "live": day is None,
        "weights": w,
        "engagement": _engagement(sim, w, d),
        "engagement_trend": (trend(sim.history, sim.warmup_days, d)
                             if sim.profile else None),
        "habits": reactions.habits(sim.profile, sim.target_profile,
                                   sim.log, d),
        "profile": sim.profile,
        "note_effects": [effects(n.get("events") or {})
                         for n in reactions.notes_of(sim.profile)],
        "goal_effects": [effects(n.get("events") or {})
                         for n in reactions.notes_of(sim.target_profile)],
        "target_engagement": _target_engagement(sim),
        "target_profile": sim.target_profile,
        "overnight": sim.last_overnight if day is None else None,
        "consolidation": {f"{c.spec.src}->{c.spec.dst}": round(c.c, 4)
                          for c in sim.current.conns},
        "start_weights": sim.start_weights(),
        "practice": sim.practice(),
        "targets": ({f"{c.spec.src}->{c.spec.dst}": round(c.w, 4)
                     for c in sim.target.conns} if sim.target_weights else
                    {f"{c.spec.src}->{c.spec.dst}": c.target
                     for c in sim.current.conns}),
        "pathways": {k: round(v, 4)
                     for k, v in sim.current.pathway_profile().items()},
        "pathway_targets": ({k: round(v, 4) for k, v in
                             sim.target.pathway_profile().items()}
                            if sim.target_weights else
                            {p.id: p.target for p in PATHWAYS.values()}),
        "pathway_progress": {k: round(v, 4)
                             for k, v in sim.pathway_progress().items()},
        "alignment": round(sim.alignment, 4),
        "progress": round(sim.progress, 4),
        "stress": round(sim.stress, 3),
        "sleep": round(sim.sleep, 3),
        "streak": sim.streak(),
        "total_practices": sum(
            1 for e in sim.log
            if _event_category(e.event_id) == "Practice"),
        "log_count": len(sim.log),
        "log": [{"day": e.day, "event": e.event_id, "note": e.note,
                 "intensity": e.intensity} for e in sim.log[-120:]],
        "history": [{"day": h.day, "alignment": h.alignment,
                     "progress": h.progress, "stress": h.stress,
                     "pathways": h.pathways, "events": h.events,
                     # Per-edge weights, rounded to 3 dp. The engine has
                     # always recorded these; without them the client can
                     # only ever draw "now", which is the whole point of a
                     # tool meant to show how things change.
                     "weights": {k: round(v, 3) for k, v in h.weights.items()}}
                    for h in sim.history],
    }


def _event_category(eid: str) -> str:
    from ..events import EVENTS_BY_ID
    return EVENTS_BY_ID[eid].category


@API.get("/api/sim/state")
@guarded
def _get_state(_: Dict[str, Any]) -> Dict[str, Any]:
    return _state(SIM)


@API.post("/api/sim/log")
@guarded
def _log(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record a behaviour and report exactly what it changed.

    The whole read-modify-read sequence runs under STATE_LOCK: 'before' and
    'after' have to bracket this event and no other, or the reported deltas
    would silently include somebody else's change.
    """
    eid = payload["event"]
    before = SIM.current.pathway_profile()
    ev = SIM.log_event(eid, payload.get("note", ""),
                       float(payload.get("intensity", 1.0)))
    if payload.get("advance_day"):
        SIM.advance_day()
    after = SIM.current.pathway_profile()

    deltas = [{
        "pathway": pid,
        "label": PATHWAYS[pid].label,
        "color": PATHWAYS[pid].color,
        "before": round(before[pid], 4),
        "after": round(after[pid], 4),
        "delta": round(after[pid] - before[pid], 5),
        "percent": round((after[pid] - before[pid]) * 100, 2),
    } for pid in after if abs(after[pid] - before[pid]) > 1e-5]
    deltas.sort(key=lambda d: -abs(d["delta"]))
    _persist()

    return {
        "state": _state(SIM),
        "applied": {
            "id": ev.id, "label": ev.label, "icon": ev.icon,
            "rule": ev.rule, "science": ev.science, "caveat": ev.caveat,
            "category": ev.category,
        },
        "deltas": deltas,
    }


@API.get("/api/sim/interpret/config")
def _interpret_config(_: Dict[str, Any]) -> Dict[str, Any]:
    """Whether a model is reachable, so the panel can say so before it is used.

    Called when the log surface opens. That also warms discover's cache, so
    the first Interpret does not pay for the port probe.

    Not @guarded: it never touches SIM.
    """
    return {"llm": llm_config()}


@API.post("/api/sim/interpret")
def _interpret(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a free-text entry and SUGGEST which known events it describes.

    This deliberately does not touch the simulation. It returns a proposal
    the user confirms or rejects, and the existing /api/sim/log endpoint is
    still the only way anything changes. Keeping the classifier and the
    engine apart is what stops a language model from becoming an unaudited
    source of claims about a real person's brain.
    """
    text = str(payload.get("text", ""))
    allow = bool(payload.get("allow_llm", True))
    result = interpret(text, allow_llm=allow)
    return {"interpretation": result.to_dict(), "llm": llm_config()}


@API.post("/api/sim/analyse")
def _analyse(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Explain what the rules say an entry engages, and what to light up.

    Deliberately NOT @guarded. It never touches SIM, and it is the slowest
    endpoint in the app; holding STATE_LOCK across a model call would put
    every other request behind it - the same rule /api/sim/interpret and
    /api/chat follow.
    """
    text = str(payload.get("text", ""))
    events = [str(e) for e in (payload.get("events") or [])][:4]
    allow = bool(payload.get("allow_llm", True))
    return {"analysis": analyse(text, events, allow_llm=allow,
                                open_ended=bool(payload.get("open")))}


@API.post("/api/sim/advance")
@guarded
def _advance(payload: Dict[str, Any]) -> Dict[str, Any]:
    SIM.advance_day(int(payload.get("days", 1)))
    _persist()
    return _state(SIM)


@API.get("/api/sim/projection")
@guarded
def _projection(_: Dict[str, Any]) -> Dict[str, Any]:
    """Where the current habits lead, run on a copy. Changes nothing."""
    return reactions.projection(SIM)


@API.post("/api/sim/replay")
@guarded
def _replay(payload: Dict[str, Any]) -> Dict[str, Any]:
    day = int(payload.get("day", 0))
    snap = SIM.snapshot_at(day)
    return _state(SIM, weights=snap.weights, day=snap.day)


@API.post("/api/sim/reset")
@guarded
def _reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    global SIM
    # Starting over clears what was logged, not who the user described or
    # wants to become - unless they ask for all of it to go.
    clear = bool(payload.get("clear_profile"))
    prof = None if clear else SIM.profile
    tw, tp = (None, None) if clear else (SIM.target_weights,
                                         SIM.target_profile)
    SIM = (reactions.build(prof.get("answers", {}), reactions.notes_of(prof))
           if prof else Simulation())
    SIM.set_target(tw, tp)
    _persist()
    return _state(SIM)


@API.post("/api/sim/demo")
@guarded
def _demo(payload: Dict[str, Any]) -> Dict[str, Any]:
    global SIM
    SIM = Simulation()
    SIM.simulate_scripted(weeks=int(payload.get("weeks", 12)),
                          adherence=float(payload.get("adherence", 0.72)))
    _persist()
    return _state(SIM)


@API.post("/api/sim/save")
@guarded
def _save(payload: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(SAVE_DIR, exist_ok=True)
    name = (payload.get("name") or "session").replace("/", "_")
    path = os.path.join(SAVE_DIR, f"{name}.json")
    SIM.save(path)
    return {"ok": True, "path": path}


@API.post("/api/sim/load")
@guarded
def _load(payload: Dict[str, Any]) -> Dict[str, Any]:
    global SIM
    name = (payload.get("name") or "session").replace("/", "_")
    path = os.path.join(SAVE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {"error": "no such save", "path": path}
    SIM = Simulation.load(path)
    _persist()
    return _state(SIM)


# --------------------------------------------------------------------------
# How you tend to react
# --------------------------------------------------------------------------

@API.get("/api/profile/questions")
def _profile_questions(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"questions": reactions.QUESTIONS, "weeks": reactions.WEEKS}


def _read_pattern(text: str, allow_llm: bool) -> tuple:
    """Classify a described pattern. Returns (note or None, read, source).

    Same classifier as log entries - a language model when one is
    reachable, keywords otherwise - and the same boundary: it only picks
    event types; unknown ones are dropped.
    """
    r = interpret(text, allow_llm=allow_llm)
    read = [{"event": p.event, "label": p.label} for p in r.proposals[:3]]
    return reactions.note_from(text, r.proposals), read, r.source


def _rebuild(answers: Dict[str, str], notes: List[Dict[str, Any]]) -> None:
    """New starting point from the description, keeping everything logged
    and the target."""
    global SIM
    old = SIM
    SIM = reactions.build(answers, notes, log=list(old.log), day=old.day)
    SIM.set_target(old.target_weights, old.target_profile)
    _persist()


@API.post("/api/profile")
def _profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Set the answers (and optionally add a described pattern).

    Not @guarded as a whole: the description goes through the classifier,
    which may call a language model, and holding STATE_LOCK across that
    would stall every other request. Only the rebuild is locked.
    """
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        answers = {}
    text = str(payload.get("text", ""))[:reactions.MAX_TEXT]
    note, read, source = (None, [], "")
    if text.strip():
        note, read, source = _read_pattern(
            text, bool(payload.get("allow_llm", True)))
    with STATE_LOCK:
        notes = reactions.notes_of(SIM.profile)
        if note:
            notes.append(note)
        _rebuild(answers, notes)
        return {"state": _state(SIM), "read_as": read, "source": source}


@API.post("/api/profile/note")
def _add_note(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add one described pattern and refine the starting point with it."""
    text = str(payload.get("text", "")).strip()[:reactions.MAX_TEXT]
    if not text:
        return {"added": False, "reason": "Nothing to add."}
    note, read, source = _read_pattern(
        text, bool(payload.get("allow_llm", True)))
    with STATE_LOCK:
        if not note:
            reason = ("Could not match that to anything the model knows "
                      "about. Try describing what you do - e.g. 'I go over "
                      "conversations for hours'.")
            if source == "keyword":
                reason += (" No language model is running, so only simple "
                           "keywords were checked - start LM Studio for a "
                           "proper reading.")
            return {"added": False, "source": source, "reason": reason}
        notes = reactions.notes_of(SIM.profile)
        if len(notes) >= reactions.MAX_NOTES:
            return {"added": False,
                    "reason": f"Up to {reactions.MAX_NOTES} patterns; "
                              "remove one first."}
        notes.append(note)
        _rebuild((SIM.profile or {}).get("answers", {}), notes)
        return {"added": True, "state": _state(SIM), "read_as": read,
                "source": source}


@API.post("/api/profile/note/delete")
@guarded
def _delete_note(payload: Dict[str, Any]) -> Dict[str, Any]:
    notes = reactions.notes_of(SIM.profile)
    i = int(payload.get("index", -1))
    if SIM.profile and 0 <= i < len(notes):
        notes.pop(i)
        _rebuild(SIM.profile.get("answers", {}), notes)
    return {"state": _state(SIM)}


@API.post("/api/profile/note/edit")
@guarded
def _edit_note(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Correct how a described pattern was read. The user picks the events;
    the rules still decide what those do."""
    notes = reactions.notes_of(SIM.profile)
    i = int(payload.get("index", -1))
    if not (SIM.profile and 0 <= i < len(notes)):
        return {"ok": False, "reason": "No such pattern."}
    note = reactions.reread(notes[i], [str(e) for e in
                                       payload.get("events") or []])
    if not note:
        return {"ok": False, "reason": "Pick at least one, or remove it."}
    notes[i] = note
    _rebuild(SIM.profile.get("answers", {}), notes)
    return {"ok": True, "state": _state(SIM)}


# --------------------------------------------------------------------------
# Who you want to become
# --------------------------------------------------------------------------

def _no_match(source: str) -> str:
    reason = ("Could not match that to anything the model knows about. Try "
              "describing what you do or want to do - e.g. 'I want to pause "
              "before I answer'.")
    if source == "keyword":
        reason += (" No language model is running, so only simple keywords "
                   "were checked - start LM Studio for a proper reading.")
    return reason


def _retarget(answers: Dict[str, Any], notes: List[Dict[str, Any]]) -> None:
    weights, profile = reactions.build_target(answers, notes)
    SIM.set_target(weights, profile)
    _persist()


@API.post("/api/target")
def _target(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Set how the user wants to react (and optionally add an aim)."""
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        answers = {}
    text = str(payload.get("text", ""))[:reactions.MAX_TEXT]
    note, read, source = (None, [], "")
    if text.strip():
        note, read, source = _read_pattern(
            text, bool(payload.get("allow_llm", True)))
    with STATE_LOCK:
        notes = reactions.notes_of(SIM.target_profile)
        if note:
            notes.append(note)
        _retarget(answers, notes)
        return {"state": _state(SIM), "read_as": read, "source": source}


@API.post("/api/target/goal")
def _add_goal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add one aim - 'I want to ...' - and rebuild the target with it."""
    text = str(payload.get("text", "")).strip()[:reactions.MAX_TEXT]
    if not text:
        return {"added": False, "reason": "Nothing to add."}
    note, read, source = _read_pattern(
        text, bool(payload.get("allow_llm", True)))
    with STATE_LOCK:
        if not note:
            return {"added": False, "source": source,
                    "reason": _no_match(source)}
        notes = reactions.notes_of(SIM.target_profile)
        if len(notes) >= reactions.MAX_NOTES:
            return {"added": False,
                    "reason": f"Up to {reactions.MAX_NOTES} aims; remove "
                              "one first."}
        notes.append(note)
        _retarget((SIM.target_profile or {}).get("answers", {}), notes)
        return {"added": True, "state": _state(SIM), "read_as": read,
                "source": source}


@API.post("/api/target/goal/delete")
@guarded
def _delete_goal(payload: Dict[str, Any]) -> Dict[str, Any]:
    notes = reactions.notes_of(SIM.target_profile)
    i = int(payload.get("index", -1))
    if SIM.target_profile and 0 <= i < len(notes):
        notes.pop(i)
        _retarget(SIM.target_profile.get("answers", {}), notes)
    return {"state": _state(SIM)}


@API.post("/api/target/goal/edit")
@guarded
def _edit_goal(payload: Dict[str, Any]) -> Dict[str, Any]:
    notes = reactions.notes_of(SIM.target_profile)
    i = int(payload.get("index", -1))
    if not (SIM.target_profile and 0 <= i < len(notes)):
        return {"ok": False, "reason": "No such aim."}
    note = reactions.reread(notes[i], [str(e) for e in
                                       payload.get("events") or []])
    if not note:
        return {"ok": False, "reason": "Pick at least one, or remove it."}
    notes[i] = note
    _retarget(SIM.target_profile.get("answers", {}), notes)
    return {"ok": True, "state": _state(SIM)}


@API.post("/api/target/clear")
@guarded
def _clear_target(_: Dict[str, Any]) -> Dict[str, Any]:
    SIM.set_target(None, None)
    _persist()
    return {"state": _state(SIM)}


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

@API.get("/api/chat/config")
def _chat_config(_: Dict[str, Any]) -> Dict[str, Any]:
    return public_chat_config()


@API.stream("/api/chat")
def _chat(payload: Dict[str, Any], h: Any) -> None:
    """
    Stream a grounded answer as server-sent events.

    Events emitted:
      meta    {ids: [...]}      regions the reply will reference
      token   {text: "..."}     incremental reply text
      show    {ids: [...]}      highlight these in the 3-D view
      done    {text: "..."}     full reply, markers stripped
      error   {message: "..."}

    Note what is NOT held here: STATE_LOCK. A local model can take a minute,
    and holding the lock across generation would freeze logging, advancing
    and every other request for that whole time. We take the lock only to
    snapshot the state we need, then let go before talking to the model.
    """
    question = str(payload.get("text", "")).strip()
    history = payload.get("history") or []
    if not question:
        h.sse_begin()
        h.sse("error", {"message": "empty question"})
        h.sse("done", {"text": ""})
        return

    cfg = public_chat_config()
    if not cfg.get("enabled"):
        h.sse_begin()
        h.sse("error", {
            "message": "No model configured. Set NEUROFORGE_LLM_BASE to an "
                       "OpenAI-compatible server (llama.cpp, Ollama, LM "
                       "Studio, vLLM) and restart."})
        h.sse("done", {"text": ""})
        return

    # Snapshot under the lock, then release it for the slow part.
    with STATE_LOCK:
        state = _state(SIM)
    scene = SCENE
    model = _model({})

    context, valid_ids = build_context(question, scene, state, model)
    # When the question names parts, only those glow; the rest become chips.
    asked = asked_ids(question, scene)

    h.sse_begin()
    h.sse("meta", {"model": cfg.get("model", ""), "local": cfg.get("local")})
    if asked:
        h.sse("show", {"ids": asked})

    buf: List[str] = []
    sent_ids: List[str] = list(asked)
    try:
        for piece in stream_chat(question, context, history):
            buf.append(piece)
            # Markers can straddle chunk boundaries, so re-scan the whole
            # buffer and only emit ids we have not already sent.
            joined = "".join(buf)
            for rid in extract_markers(joined, valid_ids):
                if rid not in sent_ids:
                    sent_ids.append(rid)
                    if not asked:
                        h.sse("show", {"ids": [rid]})
            h.sse("token", {"text": piece})
    except ClientGone:
        raise
    except Exception as exc:                              # noqa: BLE001
        h.sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    h.sse("done", {"text": strip_markers("".join(buf)).strip(),
                   "ids": sent_ids, "asked": asked})
