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
from .anatomy.structures import CORTICAL_BY_ID, STRUCTURES_BY_ID
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

_SYSTEM = """You explain, in plain language, what a brain SIMULATION says \
is going on when someone feels or does what they describe.

The brief gives what they wrote and a list of brain parts, each as \
TOKEN :: what it does. It may also give the events their words were \
recognised as and the pathways the app's rules engage.

RULES
- Name brain parts ONLY by their exact TOKEN from the brief: "vlPFC", not \
"the ventrolateral prefrontal area". The reader clicks those tokens to find \
each part on the brain, so any other spelling is a dead link.
- NEVER mention a brain part that is not listed in the brief - no \
"hypothalamus", no "frontal lobe" - even if you believe it is involved. It \
does not exist in this simulation and highlights nothing.
- Never attach a token to the wrong part, e.g. "the insula (ACC)".
- For each part you name, say what it is doing and how that connects to \
what they described. Plain language: "keeps flagging it as a threat", not \
"upregulates salience".
- If pathways are listed, mention only those pathways.
- Do not quote the numbers in the brief; they are rule settings, not findings.
- These are simulation variables, not measurements of the reader. Never write \
that their neurons, synapses or grey matter changed.
- Second person, calm and plain. No hype, no diagnosis, no advice.
- Ordinary sentence capitalisation.
- 4 to 6 sentences. No headings, no bullet points, no markdown.
- Start directly with the explanation. Do not repeat the brief."""

# Parts an open-ended (Visualise) reading may add beyond the rules: what that
# view draws, minus cerebellum, which is not modelled. The insula is here
# because body sensations are its job and no event rule reaches it.
_OPEN_PARTS = ["aINS", "pINS", "acc", "sgacc", "PCC", "mPFC", "vmPFC",
               "dlPFC", "vlPFC", "OFC", "somatosensory", "amygdala",
               "hippocampus", "brainstem", "caudate", "putamen"]

# Regexes for words a model uses for a part that are not its id, short or name.
_ALIASES = {"aINS": (r"(?<!posterior )insula",)}


def _part(pid: str) -> Any:
    return CORTICAL_BY_ID.get(pid) or STRUCTURES_BY_ID.get(pid)


def _token(pid: str) -> str:
    """The spelling the model is asked to write: the on-screen short name
    when it is a single clean word ("ACC", "NAcc"), otherwise the id."""
    p = _part(pid)
    short = getattr(p, "short", "") or ""
    return short if re.fullmatch(r"[A-Za-z]+", short) else pid


def _role(pid: str) -> str:
    p = _part(pid)
    text = (getattr(p, "contributes_to", "") or "").strip()
    first = text.split(". ")[0].rstrip(".")
    return first[:160] if first else ""


def _named_parts(reply: str, candidates: List[str]) -> List[str]:
    """Candidates the reply names, in the order it names them.

    Whole-word and case-insensitive: "Amygdala" opening a sentence counts,
    "hypothalamus" does not count as the thalamus.
    """
    found: List[Tuple[int, str]] = []
    for pid in candidates:
        p = _part(pid)
        names = {pid, _token(pid), getattr(p, "name", "") or ""}
        pats = [re.escape(n) for n in names if n] + list(_ALIASES.get(pid, ()))
        best = -1
        for pat in pats:
            m = re.search(rf"(?<![A-Za-z]){pat}(?![A-Za-z])", reply, re.I)
            if m and (best < 0 or m.start() < best):
                best = m.start()
        if best >= 0:
            found.append((best, pid))
    return [pid for _, pid in sorted(found)]


def _brief(text: str, targets: Dict[str, Any],
           extra: Optional[List[str]] = None) -> str:
    lines = [f'ENTRY: "{text.strip()[:400]}"', ""]
    if targets["events"]:
        lines.append("RECOGNISED AS:")
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
        if parts:
            roles = _part_roles(targets)
            lines.append("BRAIN PARTS INVOLVED (name the main ones):")
            for p in parts:
                detail = "; ".join(roles.get(p, [])) or "carries this activity"
                lines.append(f"- {_token(p)} :: {detail}")
        else:
            lines.append("BRAIN PARTS INVOLVED: none")
        if extra:
            lines.append("")
            lines.append("OTHER PARTS YOU MAY NAME (only for something in "
                         "the entry the parts above do not cover, such as a "
                         "body sensation):")
            lines += [f"- {_token(p)} :: {_role(p)}" for p in extra]
    elif extra:
        lines.append("No event rule matched this entry.")
        lines.append("")
        lines.append("BRAIN PARTS TO CHOOSE FROM (name the 2 to 4 that best "
                     "fit what they describe):")
        lines += [f"- {_token(p)} :: {_role(p)}" for p in extra]
    else:
        lines.append("BRAIN PARTS INVOLVED: none")
    return "\n".join(lines)


def _part_roles(targets: Dict[str, Any]) -> Dict[str, List[str]]:
    """What each part is actually doing, taken from the edges in play."""
    roles: Dict[str, List[str]] = {}
    live = {p["id"] for p in targets["pathways"]}
    for e in EDGES:
        if e.pathway not in live:
            continue
        for node, other, arrow in ((e.src, e.dst, "to"), (e.dst, e.src, "from")):
            part = NODE_TO_STRUCTURE.get(node) or NODE_TO_CORTEX.get(node)
            if not part:
                continue
            line = f"{arrow} {other}: {e.label}"
            got = roles.setdefault(part, [])
            if line not in got and len(got) < 3:
                got.append(line)
    return roles


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
    # Length only. A phrase blocklist was tried and removed: it cannot
    # distinguish a claim from a disclaimer, and the honest sentence
    # "nothing here prunes your neurons" trips every version of it.
    return len(reply) >= 40


def _sentence_case(text: str, protect: List[str]) -> str:
    """qwen3-4b writes in lower case whatever the prompt says, so fix it here.

    Part names are left alone: capitalising a sentence that opens with
    "vlPFC" would produce "VlPFC", which no longer matches the token the
    panel turns into a link.
    """
    keep = {p.lower() for p in protect}

    def up(m: "re.Match[str]") -> str:
        lead, word = m.group(1), m.group(2)
        if word.lower() in keep:
            return lead + word
        return lead + word[:1].upper() + word[1:]

    return re.sub(r"(^|[.!?]\s+)([A-Za-z]+)", up, text)


def analyse(text: str, event_ids: List[str],
            allow_llm: bool = True,
            timeout: float = 0.0,
            open_ended: bool = False) -> Dict[str, Any]:
    """Explain an entry. Anatomy from the rules, prose from a model if one is
    connected and its answer is usable.

    `open_ended` is the Visualise tab: with no events given, the text is
    classified here, and the model may add parts from `_OPEN_PARTS` for what
    the rules do not reach. Rule-derived parts are always kept.
    """
    if open_ended and not event_ids:
        from .interpret import interpret
        event_ids = [p.event for p in
                     interpret(text, allow_llm=allow_llm).proposals[:2]]
    targets = targets_for(event_ids)
    fallback = _from_rules(targets)
    out: Dict[str, Any] = dict(targets)
    out["text"] = fallback
    out["source"] = "rules"
    out["model"] = ""

    if not allow_llm:
        return out

    ruled = targets["regions"] + targets["structures"]
    extra = [p for p in _OPEN_PARTS if p not in ruled] if open_ended else []
    if not targets["events"] and not extra:
        return out

    try:
        reply, model = _generate(text, targets, timeout, extra)
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

    added = _named_parts(reply, extra)
    out["regions"] = targets["regions"] + [p for p in added
                                          if p in CORTICAL_BY_ID]
    out["structures"] = targets["structures"] + [p for p in added
                                                if p not in CORTICAL_BY_ID]
    named = ruled + added
    out["text"] = _sentence_case(reply, named + [_token(p) for p in named])
    out["source"] = "model"
    out["model"] = model
    return out


def _generate(text: str, targets: Dict[str, Any], timeout: float,
              extra: Optional[List[str]] = None) -> Tuple[str, str]:
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

    prompt = _brief(text, targets, extra)
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
        "max_tokens": 2000,
        "stream": False,
    }
    if local:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _post_json(f"{base}/chat/completions", payload, headers,
                      timeout or (300.0 if local else 20.0))
    choice = (data.get("choices") or [{}])[0]
    return str((choice.get("message") or {}).get("content") or ""), model
