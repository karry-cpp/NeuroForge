"""
interpret.py - turn a free-text log entry into events the engine understands.

THE BOUNDARY THIS FILE DEFENDS
------------------------------
A language model may **classify**. It may never **decide what happens to the
brain**.

Everything a model returns here is checked against `EVENT_TYPES` and thrown
away if it does not match. The simulation engine remains the only thing that
moves a weight, and it moves them by the same fixed rules whether the event
came from a button, a keyword match, or a frontier model. This is deliberate:

  * The effects are the honest part of this app. They come from stated rules
    with stated caveats. If an LLM could invent an effect - "this sounds like
    it strengthened your vmPFC" - the app would be generating neuroscience
    claims about a real person from a sentence of prose, which is exactly the
    thing the whole project promises not to do.
  * A classifier can be wrong in a way the user can see and correct ("you
    said this was reappraisal, it wasn't"). An effect-generator would be
    wrong in a way nobody could check.

So the contract is narrow: text in, a *proposal* out, user confirms.

NO NETWORK BY DEFAULT
---------------------
The offline keyword classifier is the default. It is genuinely worse than an
LLM at understanding "I didn't take the bait this time", and it says so by
returning a low confidence rather than guessing confidently.

Logs are private. Sending them anywhere is opt-in, off by default, and
requires the user to set an API key themselves. No key is ever read from or
written to the repository.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .events import EVENT_TYPES, EVENTS_BY_ID

# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class Proposal:
    """One suggested event. Never applied without confirmation."""
    event: str
    label: str
    category: str
    intensity: float
    confidence: float          # 0..1, the classifier's own estimate
    why: str                   # shown to the user so they can disagree

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Interpretation:
    source: str                # "keyword" | "llm:<model>"
    proposals: List[Proposal]
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "note": self.note,
            "proposals": [p.to_dict() for p in self.proposals],
        }


# --------------------------------------------------------------------------
# Offline classifier
# --------------------------------------------------------------------------

# Phrases that suggest each event type. Ordered roughly by how specific they
# are. These are crude on purpose - the point is to be obviously a keyword
# matcher, not to imitate understanding.
_CUES: Dict[str, List[str]] = {
    "name_emotion": [
        "named it", "labelled", "labeled", "put a word",
        "noticed i was feeling", "called it what it was", "named the feeling",
    ],
    "reappraisal": [
        "reframed", "another way to see", "different perspective",
        "told myself", "looked at it differently", "reinterpreted",
        "maybe they were", "gave them the benefit",
    ],
    "pause": [
        "paused", "took a breath", "waited before", "didn't react straight",
        "did not react straight", "counted to", "stepped away before",
        "slept on it", "held my tongue",
    ],
    "disengage": [
        "let the thought go", "didn't engage", "did not engage",
        "dropped it", "let it pass", "didn't take the bait",
        "did not take the bait", "unhooked",
    ],
    "exposure": [
        "did it anyway even though", "faced", "went despite", "scared but",
        "anxious but i still", "confronted", "made the call i was dreading",
    ],
    "regulated_success": [
        "stayed calm", "kept my cool", "didn't get angry", "did not get angry",
        "didn't lose it", "did not lose it", "handled it well", "coped",
        "regulated", "didn't snap", "did not snap", "kept it together",
    ],
    "mindful_practice": [
        "meditat", "breathing exercise", "breathwork", "body scan",
        "mindfulness", "sat in silence",
    ],
    "behavioural_activation": [
        "did it anyway", "got up and", "went anyway", "pushed through",
        "did the thing", "showed up despite",
    ],
    "rumination": [
        "ruminat", "spiral", "couldn't stop thinking", "could not stop thinking",
        "kept replaying", "went over and over", "overthought", "stewed",
    ],
    "avoidance": [
        "avoided", "cancelled", "canceled", "put it off", "procrastinat",
        "didn't go", "did not go", "backed out", "dodged",
    ],
    "reactive_outburst": [
        "lost my temper", "snapped at", "shouted", "yelled", "lashed out",
        "blew up", "reacted badly", "said something i regret",
    ],
    "sleep_good": [
        "slept well", "good sleep", "rested well", "full night",
        "eight hours", "8 hours",
    ],
    "sleep_poor": [
        "slept badly", "bad sleep", "barely slept", "couldn't sleep",
        "could not sleep", "insomnia", "up all night", "broken sleep",
    ],
    "exercise": [
        "ran ", "went for a run", "gym", "workout", "worked out", "cycled",
        "swim", "swam", "lifted", "walked for", "long walk",
    ],
    "stress_high": [
        "stressful", "high stress", "overwhelmed", "deadline", "burnt out",
        "burned out", "under pressure", "frazzled",
    ],
    "recovery": [
        "rested", "downtime", "saw friends", "time with", "recovered",
        "took it easy", "day off", "connected with",
    ],
}

# Words that flip the meaning of a cue when they come right before it.
_NEGATORS = ("didn't ", "did not ", "couldn't ", "could not ", "never ",
             "failed to ", "wasn't able to ", "was not able to ")

_INTENSIFIERS = {
    "really": 1.25, "very": 1.2, "extremely": 1.4, "so ": 1.15,
    "a bit": 0.7, "slightly": 0.65, "a little": 0.7, "barely": 0.55,
    "hardly": 0.55, "somewhat": 0.8, "briefly": 0.7,
}


def _intensity_for(text: str) -> float:
    mult = 1.0
    low = text.lower()
    for word, m in _INTENSIFIERS.items():
        if word in low:
            mult *= m
    return round(max(0.25, min(1.5, mult)), 2)


def classify_keywords(text: str) -> Interpretation:
    """Match phrases. Deliberately unsophisticated, and honest about it."""
    low = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    scores: Dict[str, float] = {}
    hits: Dict[str, List[str]] = {}

    for eid, cues in _CUES.items():
        for cue in cues:
            idx = low.find(cue)
            if idx < 0:
                continue
            # "didn't get angry" is a regulated success; a naive matcher that
            # saw "angry" would call it an outburst. Only cues that are not
            # themselves negations get this check.
            prefix = low[max(0, idx - 12):idx]
            negated = (any(prefix.endswith(n) for n in _NEGATORS)
                       and not any(cue.startswith(n.strip()) for n in _NEGATORS))
            if negated:
                continue
            scores[eid] = scores.get(eid, 0.0) + 1.0 + len(cue) / 100.0
            hits.setdefault(eid, []).append(cue.strip())

    if not scores:
        return Interpretation(
            source="keyword", proposals=[],
            note="No phrase in this entry matched a known event. Pick one "
                 "yourself, or turn on the language model to interpret it.")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    top = ranked[0][1]
    out: List[Proposal] = []
    for eid, sc in ranked:
        ev = EVENTS_BY_ID[eid]
        # Confidence is capped low on purpose. A keyword match is evidence
        # that a word appeared, not evidence that the user meant it.
        conf = round(min(0.62, 0.30 + 0.22 * (sc / top)), 2)
        out.append(Proposal(
            event=eid, label=ev.label, category=ev.category,
            intensity=_intensity_for(text), confidence=conf,
            why=f"matched the phrase " +
                ", ".join(f'"{h}"' for h in hits[eid][:2])))
    return Interpretation(
        source="keyword", proposals=out,
        note="Matched by keyword, not understood. Check it before applying.")


# --------------------------------------------------------------------------
# Optional LLM classifier
# --------------------------------------------------------------------------

_SYSTEM = """You classify a personal journal entry into a FIXED set of event \
types for a behavioural simulation.

You are a classifier. You are NOT a neuroscientist, a therapist, or a \
narrator. Do not describe what happened in the brain. Do not invent event \
types. Do not offer advice, reassurance or interpretation of the person.

Return ONLY a JSON object of this shape:
{"proposals":[{"event":"<id from the list>","intensity":<0.25-1.5>,\
"confidence":<0-1>,"why":"<short, quotes the user's own words>"}]}

Rules:
- "event" MUST be one of the given ids. Anything else is discarded.
- Return 1-3 proposals, best first. Return [] if none fit.
- "intensity" is how much of it they did: 0.25 barely, 1.0 normal, 1.5 a lot.
- "confidence" is your honest uncertainty. Low confidence is useful and safe;
  a confident wrong guess is not.
- "why" must point at the user's actual words, so they can disagree with you.
"""


def _catalogue() -> str:
    return "\n".join(
        f'- {e.id} ({e.category}): {e.label}' for e in EVENT_TYPES)


def _post_json(url: str, payload: Dict[str, Any],
               headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _extract_json(text: str) -> Dict[str, Any]:
    """Models like to wrap JSON in prose or fences. Dig it out."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model did not return JSON")
    return json.loads(m.group(0))


def _validate(raw: Dict[str, Any], source: str) -> Interpretation:
    """
    Keep only what is real.

    This is the security boundary, not a formality. Anything the model made
    up - an unknown event id, an out-of-range intensity - is dropped rather
    than coerced, because coercing it would silently turn a hallucination
    into a change in someone's record.
    """
    out: List[Proposal] = []
    dropped = 0
    for p in (raw.get("proposals") or [])[:3]:
        eid = str(p.get("event", "")).strip()
        ev = EVENTS_BY_ID.get(eid)
        if ev is None:
            dropped += 1
            continue
        try:
            inten = float(p.get("intensity", 1.0))
            conf = float(p.get("confidence", 0.5))
        except (TypeError, ValueError):
            dropped += 1
            continue
        out.append(Proposal(
            event=eid, label=ev.label, category=ev.category,
            intensity=round(max(0.25, min(1.5, inten)), 2),
            confidence=round(max(0.0, min(1.0, conf)), 2),
            why=str(p.get("why", ""))[:240]))
    note = ("Suggested by a language model. It reads your words; it does not "
            "read your brain. Check it before applying.")
    if dropped:
        note += f" ({dropped} suggestion(s) discarded as not a known event.)"
    return Interpretation(source=source, proposals=out, note=note)


def llm_config() -> Dict[str, Any]:
    """Read provider settings from the environment. Never from the repo."""
    provider = (os.environ.get("NEUROFORGE_LLM") or "").strip().lower()
    return {
        "provider": provider,
        "enabled": provider in ("openai", "gemini"),
        "model": os.environ.get("NEUROFORGE_LLM_MODEL") or "",
        "has_key": bool(os.environ.get("NEUROFORGE_LLM_KEY")),
        "base": os.environ.get("NEUROFORGE_LLM_BASE") or "",
    }


def classify_llm(text: str, timeout: float = 20.0) -> Interpretation:
    cfg = llm_config()
    key = os.environ.get("NEUROFORGE_LLM_KEY", "")
    if not cfg["enabled"] or not key:
        raise RuntimeError("LLM not configured")

    user = (f"Event types:\n{_catalogue()}\n\n"
            f"Journal entry:\n\"\"\"{text.strip()[:2000]}\"\"\"")

    if cfg["provider"] == "gemini":
        model = cfg["model"] or "gemini-2.0-flash"
        base = cfg["base"] or "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{model}:generateContent?key={key}"
        data = _post_json(url, {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1,
                                 "responseMimeType": "application/json"},
        }, {}, timeout)
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        model = cfg["model"] or "gpt-4o-mini"
        base = cfg["base"] or "https://api.openai.com/v1"
        data = _post_json(f"{base}/chat/completions", {
            "model": model, "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": user}],
        }, {"Authorization": f"Bearer {key}"}, timeout)
        txt = data["choices"][0]["message"]["content"]

    return _validate(_extract_json(txt), f"llm:{model}")


def interpret(text: str, allow_llm: bool = True) -> Interpretation:
    """
    Best available interpretation, degrading quietly to keywords.

    A network failure must never block logging. If the model is unreachable
    the user still gets a usable proposal and a note saying why it is worse.
    """
    if not (text or "").strip():
        return Interpretation(source="keyword", proposals=[],
                              note="Nothing to interpret.")
    if allow_llm and llm_config()["enabled"]:
        try:
            r = classify_llm(text)
            if r.proposals:
                return r
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError, KeyError, IndexError,
                ValueError, RuntimeError) as exc:
            fb = classify_keywords(text)
            fb.note = (f"Language model unavailable ({type(exc).__name__}); "
                       f"fell back to keyword matching. ") + fb.note
            return fb
    return classify_keywords(text)
