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
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import discover
from .anatomy.structures import GROUPS
from .atlas import PATHWAYS, Pathway
from .events import EVENTS_BY_ID
from .model import PROGRESS_REFERENCE

# Hard cap on what we will echo back into a prompt. Prevents a huge pasted
# journal entry from pushing the grounding text out of the context window.
MAX_QUESTION = 1200
MAX_HISTORY_TURNS = 8

SYSTEM = """You are a knowledgeable, friendly neuroscience expert. You answer \
questions about the brain, the nervous system, the mind and behaviour - \
anything the user is curious about, not only what NeuroForge models. \
NeuroForge is the interactive 3-D brain app you are built into; it runs an \
educational SIMULATION of emotional-regulation circuits.

HOW TO ANSWER:
- Answer from established neuroscience. Be accurate, clear and concrete; \
explain jargon in plain words. Say so when the science is unsettled or \
debated, and never invent a mechanism, study or statistic.
- Small talk ("hi", "thanks") gets a short, natural reply; do not lecture.
- Questions outside neuroscience: answer briefly if you can, without \
pretending to expertise you lack.
- Some messages include reference text and simulation numbers from the app. \
Use them when they are relevant; ignore them when they are not.

SAFETY - information only, never clinical:
1. No diagnosis. Never tell the user they have, or probably have, a \
condition, and never interpret their symptoms as a specific disorder. You \
may explain what a condition is and how it is generally understood.
2. No treatment, medication or dosing advice for the user's own situation. \
For personal health concerns, suggest speaking to a qualified professional.
3. If someone sounds like they are in crisis or at risk of harm, say plainly \
that you cannot help with that here and urge them to contact local emergency \
services or a crisis line, or someone they trust, now.
4. App numbers are SIMULATION variables, never measurements of the user's \
brain. Never say or imply that the app shows their brain changed.
5. You cannot log events or change the simulation. If the user describes \
something they did, tell them to use the Log mode.

HIGHLIGHTING THE 3-D BRAIN: every time you discuss a brain region whose ID \
is in the ID list, write [[show:ID]] right after its name, e.g. \
"the amygdala [[show:amygdala]]", so each part you mention lights up. \
Always write the name itself too; the marker is removed from the displayed \
text. Use up to eight per reply, and only IDs from the list.

Be concise. Short paragraphs. Do not pad."""


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

_GENERIC_NAME_WORDS = {
    "anterior", "posterior", "dorsal", "ventral", "medial", "lateral",
    "superior", "inferior", "primary", "secondary", "nucleus",
}


def _aliases(rid: str, name: str, short: str) -> List[str]:
    """Surface forms a user might plausibly type for a region."""
    out = {rid.lower(), name.lower(), short.lower()}
    # "Dorsolateral prefrontal cortex" -> also match "dorsolateral"
    words = name.split()
    if (words and len(words[0]) > 5
            and words[0].lower() not in _GENERIC_NAME_WORDS):
        out.add(words[0].lower())
    return [a for a in out if len(a) >= 3]


def _contains(query: str, term: str, prefix: bool = False) -> bool:
    if not term:
        return False
    end = "" if prefix else r"(?!\w)"
    return re.search(r"(?<!\w)" + re.escape(term) + end, query) is not None


def _has_topic(query: str, hints: Tuple[str, ...]) -> bool:
    return any(_contains(query, hint, prefix=True) for hint in hints)


# Plain-language hooks. A user asks about "fear" or "habit", not "BLA".
TOPIC_HINTS: Dict[str, Tuple[str, ...]] = {
    "amygdala": ("fear", "afraid", "threat", "panic", "scared", "alarm",
                 "anxiety", "anxious", "angry", "anger", "furious"),
    "hippocampus": ("memory", "remember", "context", "recall", "forget"),
    "dlPFC": ("focus", "concentrate", "working memory", "distract", "goal"),
    "vlPFC": ("inhibit", "stop myself", "hold back", "restrain", "label",
              "name the emotion"),
    "vmPFC": ("calm", "safety", "safe", "soothe", "value", "reapprais"),
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
        if _contains(q, a):
            score += 10
    for hint in TOPIC_HINTS.get(rid, ()):
        if _contains(q, hint, prefix=True):
            score += 4
    return score


def _group_scores(query: str, scene: Dict[str, Any],
                  parts: List[Dict[str, Any]]) -> Dict[str, int]:
    groups = {g["id"]: g for g in GROUPS}
    groups.update({g["id"]: g for g in (scene.get("groups") or [])})
    matches: List[Tuple[int, int, List[str]]] = []

    def add(phrase: str, members: List[str]) -> None:
        if not phrase or not members:
            return
        pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
        for match in re.finditer(pattern, query):
            matches.append((match.start(), match.end(), members))

    for group in groups.values():
        members = (list(group.get("regions") or [])
                   + list(group.get("structures") or []))
        for alias in {group["id"], group.get("name", ""), group.get("short", "")}:
            add(alias, members)

    striatum: List[str] = []
    for gid in ("dorsal_striatum", "ventral_striatum"):
        if gid in groups:
            striatum.extend(groups[gid].get("structures") or [])
    add("striatum", striatum)

    for phrase in ("anterior cingulate", "posterior cingulate", "cingulate"):
        members = [p["id"] for p in parts
                   if _contains(p.get("name", "").lower(), phrase)]
        add(phrase, members)

    scores: Dict[str, int] = {}
    for start, end, members in matches:
        if any(other_start <= start and end <= other_end
               and (other_start < start or end < other_end)
               for other_start, other_end, _ in matches):
            continue
        for rid in members:
            scores[rid] = 6
    return scores


def _event_reference(event_id: str) -> str:
    event = EVENTS_BY_ID[event_id]
    lines = [f"Science: {event.science}", f"Simulation: {event.rule}"]
    if event.caveat:
        lines.append(f"Caveat: {event.caveat}")
    return "\n".join(lines)


def _pathway_reference(pathway: Pathway) -> str:
    return f"Science: {pathway.science}\nSimulation: {pathway.metaphor}"


def _references(query: str) -> List[Tuple[str, str]]:
    references: List[Tuple[str, str]] = []
    if _has_topic(query, ("sleep", "slept", "insomnia")):
        good_sleep = EVENTS_BY_ID["sleep_good"]
        poor_sleep = EVENTS_BY_ID["sleep_poor"]
        text = (f"Science: {good_sleep.science}\n"
                f"Good sleep rule: {good_sleep.rule}\n"
                f"Poor sleep rule: {poor_sleep.rule}\n"
                f"Caveat: {good_sleep.caveat}")
        references.append(("event: sleep and consolidation", text))
    if _has_topic(query, ("reapprais", "refram")):
        references.append(("event: reappraisal",
                           _event_reference("reappraisal")))
    if _has_topic(query, ("ruminat", "replaying", "overthink", "spiral")):
        pathway = PATHWAYS["rumination"]
        references.append((f"pathway: {pathway.label}",
                           _pathway_reference(pathway)))
    if _has_topic(query, ("habit", "routine")):
        pathway = PATHWAYS["new_habit"]
        references.append((f"pathway: {pathway.label}",
                           _pathway_reference(pathway)))
    if any(_contains(query, word)
           for word in ("progress", "progressing", "progression")):
        references.append(("concept: progress", PROGRESS_REFERENCE))
    return references


def retrieve(question: str, scene: Dict[str, Any]
             ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Select parts and references from the question sent to the model."""
    parts = (list(scene.get("cortical_regions") or [])
             + list(scene.get("structures") or []))
    q = question.strip()[:MAX_QUESTION].lower()
    group_scores = _group_scores(q, scene, parts)
    scored: List[Tuple[bool, bool, int, Dict[str, Any]]] = []
    for part in parts:
        rid, name, short = part["id"], part.get("name", ""), part.get("short", "")
        named = any(_contains(q, alias) for alias in _aliases(rid, name, short))
        grouped = rid in group_scores
        score = _score(q, rid, name, short) + group_scores.get(rid, 0)
        if score > 0:
            scored.append((named, grouped, score, part))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    group_size = sum(grouped for _, grouped, _, _ in scored)
    named_outside_group = sum(named and not grouped for named, grouped, _, _ in scored)
    limit = max(5, min(8, group_size + named_outside_group))
    return [part for _, _, _, part in scored[:limit]], _references(q)


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
    if char_budget < 0:
        raise ValueError("char_budget must be non-negative")

    every = list(scene.get("cortical_regions") or []) + list(scene.get("structures") or [])
    valid_ids = [x["id"] for x in every]
    picked, knowledge = retrieve(question, scene)

    listing = ", ".join(f"{x['id']} ({x.get('name', '')})" for x in every)
    ids_line = f"Region IDs available for [[show:ID]]: {listing}\n"

    if not picked and not knowledge:
        ctx = ("No specific region matched the question; answer from general "
               "knowledge.\n" + ids_line)
        if state:
            ctx += "\n" + _fmt_state(state)
        return ctx[:char_budget], valid_ids

    lines = [ids_line]
    if state:
        lines.append(_fmt_state(state))
    lines.append("REFERENCE TEXT from this app (written and reviewed by the "
                 "project; use it where relevant):\n")
    sections: List[Tuple[str, Any]] = [
        (f"### {title}", text) for title, text in knowledge
    ] + [
        (f"### {part['id']} — {part.get('name', '')}", part.get("knowledge"))
        for part in picked
    ]
    edges = _fmt_edges(model, state, [part["id"] for part in picked]) if model else ""
    overhead = (len("\n".join(lines)) + sum(len(title) + 3 for title, _ in sections)
                + len(edges) + 2)
    per = max(0, (char_budget - overhead) // len(sections))
    for title, body in sections:
        lines.extend((title, _fmt_knowledge(body, per)[:per], ""))
    if edges:
        lines.append(edges)

    return "\n".join(lines)[:char_budget], valid_ids


def _fmt_state(state: Dict[str, Any]) -> str:
    """Current simulation numbers, labelled unambiguously as simulation."""
    pw = state.get("pathways") or {}
    rows = ", ".join(f"{k} {v:.2f}" for k, v in list(pw.items())[:8])
    progress = state.get("progress")
    progress_text = f", simulated progress {progress}" if progress is not None else ""
    return (
        "CURRENT SIMULATION STATE (these are model variables, NOT "
        "measurements of the user):\n"
        f"day {state.get('day')}, simulated alignment {state.get('alignment')}"
        f"{progress_text}, "
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
    return found[:8]


def strip_markers(text: str) -> str:
    return MARKER.sub("", text)


def asked_ids(question: str, scene: Dict[str, Any]) -> List[str]:
    """Parts the question itself names, in the order they appear."""
    q = question.strip()[:MAX_QUESTION].lower()
    hits: List[Tuple[int, str]] = []
    seen = set()
    for x in (scene.get("cortical_regions") or []) + (scene.get("structures") or []):
        label = x.get("label") or ""
        plain, _, inner = label.partition("(")
        aliases = {x["id"], x.get("name", ""), label, inner.rstrip(")")}
        # Single plain words ("Mood", "Touch") are too common to mean the part.
        if len(plain.split()) > 1:
            aliases.add(plain)
        pos = min((m.start() for a in aliases if a.strip()
                   for m in re.finditer(r"\b" + re.escape(a.strip().lower())
                                        + r"\b", q)), default=-1)
        # sgACC exists as both a structure and a cortex region; light it once.
        if pos >= 0 and x["id"].lower() not in seen:
            seen.add(x["id"].lower())
            hits.append((pos, x["id"]))
    return [rid for _, rid in sorted(hits)]


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


def _open(req: urllib.request.Request, timeout: float) -> Any:
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))
            detail = (detail.get("error") or {}).get("message") or ""
        except (ValueError, AttributeError, OSError):
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from None


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
    # Cloudflare-fronted providers (e.g. Groq) reject Python's default User-Agent.
    headers = {"Content-Type": "application/json",
               "User-Agent": "NeuroForge"}
    if cfg["key"]:
        headers["Authorization"] = f"Bearer {cfg['key']}"

    req = urllib.request.Request(
        cfg["base"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")

    with _open(req, timeout) as resp:
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
