"""
neuroforge.llm
==============

Streaming chat grounded in the app's own anatomy and simulation state.

Design notes worth reading before changing anything here:

**The model is a narrator, not an authority.** It is given retrieved text that
the project already wrote and vetted, plus the current simulation numbers, and
asked to explain them. It is never asked to invent a mechanism, and it cannot
change the simulation. Logging still goes through `interpret` -> user
confirmation -> `/api/sim/log`. There is deliberately no `log_event` tool.

**Retrieval, not stuffing.** The full knowledge corpus is ~7k tokens and the
model description another ~6k. Putting all of that in every turn wrecks a 9B
model's attention and makes local generation crawl. `build_context` selects
only what the question mentions and targets roughly 2k tokens.

**Inline markers, not tool calls.** To drive the 3-D view the model emits
`[[show:dlPFC]]`. Small local models are unreliable at JSON function calling
and every call costs another round-trip on an already-slow local generation.
A marker is one token sequence in the middle of a stream that the client can
act on immediately, and if the model emits none you still get a usable answer.
Markers are validated against the real id list before they reach the client,
so a hallucinated region id is dropped rather than throwing in the renderer.

**Local models are first-class.** Any OpenAI-compatible server (llama.cpp,
Ollama, LM Studio, vLLM) works by pointing NEUROFORGE_LLM_BASE at it. That is
the intended setup: nothing about a personal log should need to leave the
machine.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import discover

# Hard cap on what we will echo back into a prompt. Prevents a huge pasted
# journal entry from pushing the grounding text out of the context window.
MAX_QUESTION = 1200
MAX_HISTORY_TURNS = 8

SYSTEM = """You are a neuroscience explainer built into NeuroForge, an \
interactive 3-D brain that runs an educational SIMULATION of how repeated \
behaviour might shift emotional-regulation circuits.

ABSOLUTE RULES — these are not style preferences:
1. The numbers you are shown are SIMULATION variables. They are not \
measurements of the user's brain. Never say or imply otherwise.
2. Never say anything like "you pruned your neurons", "your PFC gained grey \
matter", or "this shows your brain changed". If the user believes that, \
correct them plainly and kindly.
3. Distinguish REAL (established neuroscience) from MODEL (how this app \
chose to represent it). When you state a model behaviour, say so.
4. You cannot log events or change the simulation. If the user describes \
something they did, tell them to put it in the log box; do not claim to have \
recorded it.
5. No diagnosis, no treatment advice, no clinical claims. If someone sounds \
like they are in crisis, say plainly that this app is not equipped for that \
and suggest talking to a real person.
6. If the grounding text below does not cover something, say you do not know \
rather than inventing a mechanism. Made-up neuroscience is the worst possible \
failure here.

TO POINT AT THE BRAIN: write [[show:ID]] using an ID from the grounding text, \
e.g. [[show:amygdala]]. Use it the first time you discuss a region. It \
renders as a highlight in the 3-D view, so put it inline where it makes \
sense. Use at most three per reply.

Be concise and concrete. Short paragraphs. Prefer the user's own words back \
to them over jargon. Do not pad."""


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _aliases(rid: str, name: str, short: str) -> List[str]:
    """Surface forms a user might plausibly type for a region."""
    out = {rid.lower(), name.lower(), short.lower()}
    # "Dorsolateral prefrontal cortex" -> also match "dorsolateral"
    first = name.split()[0].lower()
    if len(first) > 5:
        out.add(first)
    return [a for a in out if len(a) >= 3]


# Plain-language hooks. A user asks about "fear" or "habit", not "BLA".
TOPIC_HINTS: Dict[str, Tuple[str, ...]] = {
    "amygdala": ("fear", "afraid", "threat", "panic", "scared", "alarm",
                 "anxiety", "anxious"),
    "hippocampus": ("memory", "remember", "context", "recall", "forget"),
    "dlPFC": ("focus", "concentrate", "working memory", "distract", "goal"),
    "vlPFC": ("inhibit", "stop myself", "hold back", "restrain", "label",
              "name the emotion"),
    "vmPFC": ("calm", "safety", "safe", "soothe", "value", "reappraise"),
    "dACC": ("conflict", "effort", "error", "notice", "monitor"),
    "sgACC": ("sad", "sadness", "low mood", "depress", "rumination",
              "ruminate", "spiral"),
    "aINS": ("body", "gut", "interocept", "heart", "breath", "feel it",
             "sensation", "disgust", "urge", "craving"),
    "OFC": ("reward", "value", "worth", "regret", "decision"),
    "accumbens": ("reward", "motivation", "want", "crave", "dopamine",
                  "relief", "pleasure"),
    "putamen": ("habit", "automatic", "autopilot", "routine"),
    "caudate": ("habit", "goal", "action", "routine"),
    "vta": ("dopamine", "reward", "motivation", "learning signal"),
    "PCC": ("mind wander", "daydream", "self", "rumination", "default"),
    "mPFC": ("self", "thinking about myself", "introspect", "default"),
    "TPJ": ("other people", "perspective", "social", "empathy"),
}


def _score(query: str, rid: str, name: str, short: str) -> int:
    """How relevant is this region to the question? Cheap, deliberate."""
    q = query.lower()
    score = 0
    for a in _aliases(rid, name, short):
        if a in q:
            score += 10
    for hint in TOPIC_HINTS.get(rid, ()):
        if hint in q:
            score += 4
    return score


def _fmt_knowledge(k: Any, budget: int) -> str:
    """Knowledge blobs are dicts of prose. Flatten within a char budget."""
    if isinstance(k, str):
        return k[:budget]
    if not isinstance(k, dict):
        return ""
    parts = []
    used = 0
    # Ordered by usefulness when explaining, not by dict order.
    for key in ("what_it_is", "contributes_to", "where_it_is",
                "common_misconception", "caveat"):
        v = k.get(key)
        if not isinstance(v, str) or not v:
            continue
        chunk = f"{key.replace('_', ' ')}: {v}"
        if used + len(chunk) > budget:
            chunk = chunk[: max(0, budget - used)]
        if not chunk:
            break
        parts.append(chunk)
        used += len(chunk)
        if used >= budget:
            break
    return "\n".join(parts)


def build_context(question: str, scene: Dict[str, Any],
                  state: Optional[Dict[str, Any]] = None,
                  model: Optional[Dict[str, Any]] = None,
                  char_budget: int = 7000) -> Tuple[str, List[str]]:
    """
    Assemble grounding text for one question.

    Returns (context_text, valid_ids). `valid_ids` is every id the model is
    allowed to reference in a [[show:...]] marker; anything else is stripped
    downstream.

    The budget is in characters (~4 chars/token) because that is what we can
    actually measure without a tokeniser. 7000 chars is roughly 1.7k tokens,
    which leaves a 9B model plenty of room to answer.
    """
    regions = list(scene.get("cortical_regions") or [])
    structs = list(scene.get("structures") or [])
    every = regions + structs
    valid_ids = [x["id"] for x in every]

    scored = []
    for x in every:
        s = _score(question, x["id"], x.get("name", ""), x.get("short", ""))
        if s > 0:
            scored.append((s, x))
    scored.sort(key=lambda t: -t[0])

    # Nothing matched: give a short menu instead of a random sample, so the
    # model can ask a sensible follow-up rather than guessing.
    if not scored:
        listing = ", ".join(f"{x['id']} ({x.get('name', '')})"
                            for x in every[:40])
        ctx = ("No specific region matched the question.\n"
               f"Regions available to discuss: {listing}\n")
        if state:
            ctx += "\n" + _fmt_state(state)
        return ctx[:char_budget], valid_ids

    picked = scored[:5]
    per = max(400, (char_budget - 1200) // max(1, len(picked)))

    lines = ["GROUNDING TEXT (written and reviewed by this project — prefer "
             "it over your own recall):\n"]
    for _, x in picked:
        lines.append(f"### {x['id']} — {x.get('name', '')}")
        body = _fmt_knowledge(x.get("knowledge"), per)
        if body:
            lines.append(body)
        lines.append("")

    if state:
        lines.append(_fmt_state(state))
    if model:
        lines.append(_fmt_edges(model, state, [x["id"] for _, x in picked]))

    return "\n".join(lines)[:char_budget], valid_ids


def _fmt_state(state: Dict[str, Any]) -> str:
    """Current simulation numbers, labelled unambiguously as simulation."""
    pw = state.get("pathways") or {}
    rows = ", ".join(f"{k} {v:.2f}" for k, v in list(pw.items())[:8])
    return (
        "CURRENT SIMULATION STATE (these are model variables, NOT "
        "measurements of the user):\n"
        f"day {state.get('day')}, simulated alignment {state.get('alignment')}, "
        f"stress {state.get('stress')}, sleep {state.get('sleep')}\n"
        f"simulated pathway strengths: {rows}\n"
    )


def _fmt_edges(model: Dict[str, Any], state: Optional[Dict[str, Any]],
               ids: List[str]) -> str:
    """Edges touching the regions we retrieved, with their current weights."""
    weights = (state or {}).get("weights") or {}
    out = []
    for e in (model.get("edges") or []):
        eid = e.get("id") or f"{e.get('src')}->{e.get('dst')}"
        if not any(i.lower() in eid.lower() for i in ids):
            continue
        w = weights.get(eid)
        out.append(f"  {eid}"
                   + (f" — simulated strength {w:.2f}" if isinstance(w, (int, float))
                      else ""))
        if len(out) >= 8:
            break
    if not out:
        return ""
    return ("RELEVANT SIMULATED CONNECTIONS (model constructs, not tractography):\n"
            + "\n".join(out) + "\n")


# --------------------------------------------------------------------------
# Marker handling
# --------------------------------------------------------------------------

MARKER = re.compile(r"\[\[show:([A-Za-z0-9_,\s-]+)\]\]")


def extract_markers(text: str, valid_ids: List[str]) -> List[str]:
    """Pull valid region ids out of [[show:...]] markers. Drop invented ones."""
    lower = {v.lower(): v for v in valid_ids}
    found: List[str] = []
    for m in MARKER.finditer(text):
        for part in m.group(1).split(","):
            key = part.strip().lower()
            if key in lower and lower[key] not in found:
                found.append(lower[key])
    return found[:3]


def strip_markers(text: str) -> str:
    return MARKER.sub("", text)


# --------------------------------------------------------------------------
# Streaming transport
# --------------------------------------------------------------------------

def chat_config() -> Dict[str, Any]:
    """
    Chat provider settings.

    Falls back to the NEUROFORGE_LLM_* variables that `interpret` already
    uses, so a single local server configuration powers both features. A key
    is NOT required: local servers generally ignore it, and demanding one
    would make the common local setup fail for no reason.
    """
    provider = (os.environ.get("NEUROFORGE_CHAT")
                or os.environ.get("NEUROFORGE_LLM") or "").strip().lower()
    base = (os.environ.get("NEUROFORGE_CHAT_BASE")
            or os.environ.get("NEUROFORGE_LLM_BASE") or "").strip()
    model = (os.environ.get("NEUROFORGE_CHAT_MODEL")
             or os.environ.get("NEUROFORGE_LLM_MODEL") or "").strip()
    key = (os.environ.get("NEUROFORGE_CHAT_KEY")
           or os.environ.get("NEUROFORGE_LLM_KEY") or "")
    # Nothing configured: look for a local runner on its usual port, so that
    # starting LM Studio or Ollama is the entire setup.
    runner = ""
    if not base:
        found = discover.find()
        if found:
            base = str(found["base"])
            model = model or str(found["model"])
            runner = str(found["runner"])
    local = bool(base) and ("127.0.0.1" in base or "localhost" in base
                            or "0.0.0.0" in base)
    return {
        "provider": provider or ("openai" if base else ""),
        "enabled": bool(base) or (provider == "openai" and bool(key)),
        "base": base or "https://api.openai.com/v1",
        "model": model or ("local-model" if local else "gpt-4o-mini"),
        "runner": runner,
        "small": discover.small_model(model or ""),
        "key": key,
        "local": local,
        "has_key": bool(key),
    }


def public_chat_config() -> Dict[str, Any]:
    """chat_config without the secret, safe to send to the browser."""
    c = chat_config()
    return {k: v for k, v in c.items() if k != "key"}


def stream_chat(question: str, context: str,
                history: Optional[List[Dict[str, str]]] = None,
                timeout: float = 120.0) -> Iterator[str]:
    """
    Yield reply text chunks from an OpenAI-compatible /chat/completions
    endpoint with stream=true.

    Timeout defaults to 120s: a 9B model on a laptop GPU is not fast, and a
    short timeout would kill legitimate replies. The server is threaded, so a
    long generation no longer blocks anything else.
    """
    cfg = chat_config()
    if not cfg["enabled"]:
        raise RuntimeError("chat not configured")

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = str(turn.get("content", ""))[:MAX_QUESTION]
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user",
                 "content": f"{context}\n\n---\nQuestion: "
                            f"{question.strip()[:MAX_QUESTION]}"})

    payload = {
        "model": cfg["model"],
        "messages": msgs,
        "temperature": 0.3,
        "max_tokens": 700,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = f"Bearer {cfg['key']}"

    req = urllib.request.Request(
        cfg["base"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece
