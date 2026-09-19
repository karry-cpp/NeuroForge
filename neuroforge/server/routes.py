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
  POST /api/sim/replay      state at a past day
  POST /api/sim/reset       start over
  POST /api/sim/demo        fast-forward a scripted practice history
  POST /api/sim/save|load   persistence
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
from ..engine import Simulation
from ..events import CATEGORIES, EVENT_TYPES
from ..interpret import interpret, llm_config
from ..llm import (build_context, extract_markers, public_chat_config,
                   stream_chat, strip_markers)
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

def _state(sim: Simulation, weights: Dict[str, float] | None = None,
           day: int | None = None) -> Dict[str, Any]:
    """Serialise the simulation. `weights` overrides for replay."""
    w = weights if weights is not None else sim.current.snapshot()
    return {
        "day": sim.day if day is None else day,
        "date": str(sim.date_for(sim.day if day is None else day)),
        "live": day is None,
        "weights": w,
        "consolidation": {f"{c.spec.src}->{c.spec.dst}": round(c.c, 4)
                          for c in sim.current.conns},
        "targets": {f"{c.spec.src}->{c.spec.dst}": c.target
                    for c in sim.current.conns},
        "pathways": {k: round(v, 4)
                     for k, v in sim.current.pathway_profile().items()},
        "pathway_targets": {p.id: p.target for p in PATHWAYS.values()},
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
    return {"analysis": analyse(text, events, allow_llm=allow)}


@API.post("/api/sim/advance")
@guarded
def _advance(payload: Dict[str, Any]) -> Dict[str, Any]:
    SIM.advance_day(int(payload.get("days", 1)))
    return _state(SIM)


@API.post("/api/sim/replay")
@guarded
def _replay(payload: Dict[str, Any]) -> Dict[str, Any]:
    day = int(payload.get("day", 0))
    snap = SIM.snapshot_at(day)
    return _state(SIM, weights=snap.weights, day=snap.day)


@API.post("/api/sim/reset")
@guarded
def _reset(_: Dict[str, Any]) -> Dict[str, Any]:
    global SIM
    SIM = Simulation()
    return _state(SIM)


@API.post("/api/sim/demo")
@guarded
def _demo(payload: Dict[str, Any]) -> Dict[str, Any]:
    global SIM
    SIM = Simulation()
    SIM.simulate_scripted(weeks=int(payload.get("weeks", 12)),
                          adherence=float(payload.get("adherence", 0.72)))
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
    return _state(SIM)


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

    h.sse_begin()
    h.sse("meta", {"model": cfg.get("model", ""), "local": cfg.get("local")})

    buf: List[str] = []
    sent_ids: List[str] = []
    try:
        for piece in stream_chat(question, context, history):
            buf.append(piece)
            # Markers can straddle chunk boundaries, so re-scan the whole
            # buffer and only emit ids we have not already sent.
            joined = "".join(buf)
            for rid in extract_markers(joined, valid_ids):
                if rid not in sent_ids:
                    sent_ids.append(rid)
                    h.sse("show", {"ids": [rid]})
            h.sse("token", {"text": piece})
    except ClientGone:
        raise
    except Exception as exc:                              # noqa: BLE001
        h.sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    h.sse("done", {"text": strip_markers("".join(buf)).strip(),
                   "ids": sent_ids})
