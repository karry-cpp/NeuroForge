"""
Explain, in plain language, what the app's own rules say an entry engages.

    The anatomy is DERIVED. The prose is GENERATED.

Every region, structure and pathway an analysis names is computed here, by
walking the app's data:

    event id -> EVENT_TYPES[id].activations   (pathway -> signed engagement)
             -> EDGES filtered by pathway     (which connections carry it)
             -> edge src/dst nodes            (which parts of the atlas)
             -> NODE_TO_STRUCTURE / NODE_TO_CORTEX   (what to light up)

That derivation is load-bearing, not decoration: the ids it produces are fed
straight to `viewer.selectMany()`, so they have to be exact strings that
exist in the scene. A model asked to name regions would spell them its own
way and highlight nothing. It is handed the finished list plus the science
text from `atlas.py` and `events.py`, and writes the English.

`_from_rules` needs no model at all and is what ships when nothing is
connected - the default state of the app.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .atlas import EDGES, PATHWAYS
from .anatomy.anchors import NODE_TO_CORTEX, NODE_TO_STRUCTURE
from .events import EVENTS_BY_ID

# Engagements smaller than this are numerical noise in the rule table and
# would pad the highlight with parts that barely participate.
MIN_ENGAGEMENT = 0.05
MAX_TEXT = 1200


class _NoModel(Exception):
    """No model configured - distinct from one that is configured and failing."""


# ---------------------------------------------------------------------------
# Derivation - the honest half
# ---------------------------------------------------------------------------

def _nodes_for(pathway: str) -> List[str]:
    seen: List[str] = []
    for e in EDGES:
        if e.pathway != pathway:
            continue
        for n in (e.src, e.dst):
            if n not in seen:
                seen.append(n)
    return seen


def targets_for(event_ids: List[str]) -> Dict[str, Any]:
    """Regions, structures and pathways an event set engages, per the rules.

    Signed: an event that *weakens* a pathway still engages it, and the user
    needs to see which direction the rule pushes. `rumination -0.20` is not
    the same claim as `regulation +0.70`, and collapsing them to "involved"
    would throw away the only part that carries meaning.
    """
    engaged: Dict[str, float] = {}
    events: List[Any] = []
    for eid in event_ids:
        ev = EVENTS_BY_ID.get(eid)
        if ev is None:                      # unknown id -> dropped, not coerced
            continue
        events.append(ev)
        for pid, amount in ev.activations.items():
            if pid not in PATHWAYS or abs(amount) < MIN_ENGAGEMENT:
                continue
            # Strongest engagement wins when two events touch one pathway.
            if abs(amount) > abs(engaged.get(pid, 0.0)):
                engaged[pid] = amount

    order = sorted(engaged.items(), key=lambda kv: -abs(kv[1]))
    pathways: List[Dict[str, Any]] = []
    regions: List[str] = []
    structures: List[str] = []
    edge_keys: List[str] = []

    for pid, amount in order:
        p = PATHWAYS[pid]
        for n in _nodes_for(pid):
            rid = NODE_TO_CORTEX.get(n)
            sid = NODE_TO_STRUCTURE.get(n)
            if rid and rid not in regions:
                regions.append(rid)
            if sid and sid not in structures:
                structures.append(sid)
        for e in EDGES:
            if e.pathway == pid:
                key = f"{e.src}->{e.dst}"
                if key not in edge_keys:
                    edge_keys.append(key)
        pathways.append({
            "id": pid,
            "label": p.label,
            "amount": round(amount, 3),
            "direction": "strengthen" if amount > 0 else "weaken",
            "desirable": p.desirable,
            # Sign alone is the wrong thing to colour by: strengthening the
            # stress habit is a +0.70 that moves you away from the target.
            "beneficial": (amount > 0) == p.desirable,
            "science": p.science,
        })

    return {
        "events": [{"id": e.id, "label": e.label, "category": e.category,
                    "rule": e.rule, "science": e.science, "caveat": e.caveat}
                   for e in events],
        "pathways": pathways,
        "modulators": _modulators(events),
        "regions": regions,
        "structures": structures,
        "edges": edge_keys,
    }


def _modulators(events: List[Any]) -> List[Dict[str, Any]]:
    """Effects that change the conditions rather than any one connection.

    Four of the five State events carry no activations at all: sleep and
    stress act on how well anything else consolidates. Reading only
    `activations` made those entries come back as "nothing matched", which
    was false - they matched, they just do not move a single edge.
    """
    out: List[Dict[str, Any]] = []
    stress = 0.0
    sleep: Optional[float] = None
    rate = 1.0
    for e in events:
        if e.sets_sleep is not None:
            sleep = float(e.sets_sleep)     # absolute: the last report wins
        stress += float(e.stress_delta or 0.0)
        rate *= float(e.lr_mult or 1.0)
    if sleep is not None:
        out.append({"kind": "sleep", "label": "Sleep quality",
                    "amount": round(sleep, 3), "absolute": True})
    if abs(stress) >= 0.005:
        out.append({"kind": "stress", "label": "Stress load",
                    "amount": round(stress, 3), "absolute": False})
    if abs(rate - 1.0) >= 0.005:
        out.append({"kind": "rate", "label": "How fast this sticks",
                    "amount": round(rate, 3), "absolute": True})
    return out


def _readable(pathways: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    up = [p["label"].lower() for p in pathways if p["amount"] > 0]
    down = [p["label"].lower() for p in pathways if p["amount"] < 0]
    return up, down


def _join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _from_rules(targets: Dict[str, Any]) -> str:
    """The reference explanation, composed from the rule table alone."""
    evs = targets["events"]
    pws = targets["pathways"]
    mods = targets["modulators"]
    if not evs:
        return ("Nothing in this entry matched a known event, so the app has "
                "no rule to apply and nothing to show. You can still pick an "
                "event by hand.")
    names = _join([e["label"].lower() for e in evs])
    bits = [f"Recorded as {names}."]
    if pws:
        up, down = _readable(pws)
        if up:
            bits.append(f"That strengthens {_join(up)} in the model.")
        if down:
            bits.append(f"It weakens {_join(down)}.")
        bits.append(pws[0]["science"].split(". ")[0].rstrip(".") + ".")
    if mods:
        bits.append(_mod_sentence(mods))
    if not pws and mods:
        bits.append("That does not move any one connection on its own - it "
                    "changes how well everything else you log will stick.")
    bits.append("These are fixed rules being applied, not a measurement of "
                "you.")
    return " ".join(bits)


def _mod_sentence(mods: List[Dict[str, Any]]) -> str:
    parts = []
    for m in mods:
        if m["kind"] == "sleep":
            parts.append(f"sets sleep quality to {m['amount']:.2f}")
        elif m["kind"] == "stress":
            verb = "raises" if m["amount"] > 0 else "lowers"
            parts.append(f"{verb} stress load by {abs(m['amount']):.2f}")
        else:
            parts.append(f"scales the learning rate to {m['amount']:.2f}")
    return "It " + _join(parts) + "."


# ---------------------------------------------------------------------------
# Prompting - the generated half
# ---------------------------------------------------------------------------

_SYSTEM = """You explain what a journal entry engages in a brain SIMULATION.

You are given the entry, the events it was classified as, and the pathways \
the app's rules say those events engage. Write the explanation.

RULES
- Use ONLY the pathways and brain parts listed in the brief. Naming anything \
else breaks the highlight the reader sees beside your text.
- Second person, calm and plain. No hype, no diagnosis, no advice.
- 3 to 5 sentences. No headings, no bullet points, no markdown.
- Begin by connecting what they wrote to what was recognised."""


def _brief(text: str, targets: Dict[str, Any]) -> str:
    lines = [f'ENTRY: "{text.strip()[:400]}"', "", "RECOGNISED AS:"]
    for e in targets["events"]:
        lines.append(f"- {e['label']} ({e['category']}) :: {e['rule']}")
        if e["science"]:
            lines.append(f"  known: {e['science'][:260]}")
    lines.append("")
    lines.append("PATHWAYS ENGAGED (the only ones you may mention):")
    if targets["pathways"]:
        for p in targets["pathways"]:
            lines.append(
                f"- {p['label']} :: {p['direction']} by {abs(p['amount']):.2f}"
                f" :: {p['science'][:260]}")
    else:
        lines.append("- none. This event changes the CONDITIONS for learning, "
                     "not any single connection.")
    if targets["modulators"]:
        lines.append("")
        lines.append("CONDITIONS CHANGED:")
        for m in targets["modulators"]:
            # Absolute settings and deltas read identically if both are
            # printed with a sign, and a model told "sleep +0.30" reports it
            # as an improvement when 0.30 means a bad night.
            if m["absolute"]:
                lines.append(
                    f"- {m['label']} is SET TO {m['amount']:.2f} "
                    f"on a 0-1 scale where 0 is worst and 1 is best "
                    f"(this is a level, NOT an increase)")
            else:
                word = "increases" if m["amount"] > 0 else "decreases"
                lines.append(
                    f"- {m['label']} {word} by {abs(m['amount']):.2f}")
    parts = targets["regions"] + targets["structures"]
    lines.append("")
    lines.append("BRAIN PARTS INVOLVED: " + (", ".join(parts) or "none"))
    return "\n".join(lines)


def _clean(reply: str) -> str:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    reply = re.sub(r"^.*?</think>", "", reply, flags=re.S)
    reply = re.sub(r"\[\[.*?\]\]", "", reply)          # chat highlight markers
    reply = re.sub(r"[*_#`]+", "", reply)              # stray markdown
    # Models emit typographic quotes; they survive JSON but look like mojibake
    # anywhere the page is not served as UTF-8.
    reply = (reply.replace("\u2019", "'").replace("\u2018", "'")
                  .replace("\u201c", '"').replace("\u201d", '"')
                  .replace("\u2014", " - ").replace("\u2013", "-"))
    reply = re.sub(r"\s+", " ", reply).strip()
    return reply[:MAX_TEXT]


def _acceptable(reply: str) -> bool:
    return len(reply) >= 40


def analyse(text: str, event_ids: List[str],
            allow_llm: bool = True,
            timeout: float = 0.0) -> Dict[str, Any]:
    """Explain an entry. Anatomy from the rules, prose from a model if one is
    connected and its answer passes the honesty checks."""
    targets = targets_for(event_ids)
    fallback = _from_rules(targets)
    out: Dict[str, Any] = dict(targets)
    out["text"] = fallback
    out["source"] = "rules"
    out["model"] = ""

    if not allow_llm or not targets["events"]:
        return out

    try:
        reply, model = _generate(text, targets, timeout)
    except _NoModel:
        # Nothing configured is the documented default, not a failure. Saying
        # "unreachable" here would make a working app look broken.
        return out
    except Exception:
        out["note"] = "no model reachable - rule-based explanation shown"
        return out

    reply = _clean(reply)
    if not _acceptable(reply):
        out["note"] = "model returned no usable text"
        return out

    out["text"] = reply
    out["source"] = "model"
    out["model"] = model
    return out


def _generate(text: str, targets: Dict[str, Any],
              timeout: float) -> Tuple[str, str]:
    import os

    from .interpret import _post_json, llm_config

    cfg = llm_config()
    key = os.environ.get("NEUROFORGE_LLM_KEY", "")
    if not cfg["enabled"]:
        raise _NoModel()
    if not cfg["base"] and not key:
        raise _NoModel()

    base = (cfg["base"] or "https://api.openai.com/v1").rstrip("/")
    model = cfg["model"] or "gpt-4o-mini"
    local = any(h in base for h in
                ("127.0.0.1", "localhost", "0.0.0.0", "::1"))

    prompt = _brief(text, targets)
    # Reasoning models otherwise spend the whole budget thinking and return
    # empty content. Qwen3 reads this marker in the prompt itself, which is
    # the only lever that works when a server ignores enable_thinking.
    if "qwen" in model.lower():
        prompt += "\n/no_think"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
        "stream": False,
    }
    if local:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _post_json(f"{base}/chat/completions", payload, headers,
                      timeout or (300.0 if local else 20.0))
    choice = (data.get("choices") or [{}])[0]
    return str((choice.get("message") or {}).get("content") or ""), model
