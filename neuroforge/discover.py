"""Discover local OpenAI-compatible model runners without network access."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


_RUNNERS = (
    ("ollama", "http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/tags"),
    ("lmstudio", "http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1/models"),
)

# Probing two ports on every call would put a socket timeout in front of each
# request. Cached, with a short life so that starting a runner after the app
# is still noticed without a restart.
_CACHE: Optional[Dict[str, str]] = None
_CACHED_AT = 0.0
_TTL = 20.0


def _off() -> bool:
    """NEUROFORGE_LLM=off disables discovery entirely.

    Without this the test suite calls whatever model the developer happens to
    have loaded: it hit the network, took as long as generation took, and
    asserted against the output.
    """
    return (os.environ.get("NEUROFORGE_LLM") or "").strip().lower() in (
        "off", "none", "0", "false", "disabled")


def _get_json(url: str, timeout: float = 1.5) -> Optional[Dict[str, Any]]:
    # 1.5s, not 150ms: a model server that is busy loading weights answers
    # slowly, and treating that as "not installed" silently downgrades the
    # whole feature to keyword matching.
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, dict) else None


def _model_name(payload: Optional[Dict[str, Any]], runner: str) -> str:
    if not payload:
        return "local-model"
    if runner == "ollama":
        models = payload.get("models")
        if isinstance(models, list) and models and isinstance(models[0], dict):
            return str(models[0].get("name") or "local-model")
    # LM Studio's /v1/models lists everything *downloaded*, not what is
    # loaded, and in no useful order. Taking data[0] blindly can hand an
    # embedding model to a chat endpoint - which fails every request - or a
    # 12GB model that does not fit in VRAM. Its native endpoint reports both
    # `type` and `state`, so prefer that when it answers.
    if runner == "lmstudio":
        name = _lmstudio_pick()
        if name:
            return name
    models = payload.get("data")
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and not _is_embedding(str(entry.get("id") or "")):
                return str(entry.get("id"))
    return "local-model"


def _is_embedding(model_id: str) -> bool:
    return "embed" in model_id.lower()


def _lmstudio_pick() -> str:
    """A loaded, non-embedding model from LM Studio's native model list."""
    payload = _get_json("http://127.0.0.1:1234/api/v0/models")
    entries = (payload or {}).get("data")
    if not isinstance(entries, list):
        return ""
    usable = [e for e in entries
              if isinstance(e, dict)
              and e.get("type") in ("llm", "vlm")
              and not _is_embedding(str(e.get("id") or ""))]
    for entry in usable:
        if entry.get("state") == "loaded":
            return str(entry.get("id"))
    return str(usable[0].get("id")) if usable else ""


def find(force: bool = False) -> Optional[Dict[str, str]]:
    """The first reachable supported local runner, if any.

    `force` clears the cache and re-probes, which is also how a test resets
    state after changing the environment.
    """
    global _CACHE, _CACHED_AT
    if force:
        _CACHE, _CACHED_AT = None, 0.0
    if _off():
        return None
    if _CACHE is not None and (time.monotonic() - _CACHED_AT) < _TTL:
        return _CACHE
    for runner, base, probe in _RUNNERS:
        payload = _get_json(probe)
        if payload is not None:
            _CACHE = {"runner": runner, "base": base,
                      "model": _model_name(payload, runner)}
            _CACHED_AT = time.monotonic()
            return _CACHE
    _CACHE, _CACHED_AT = None, time.monotonic()
    return None


def resolve() -> Dict[str, Any]:
    """Resolve explicit environment settings, then optional local discovery."""
    if _off():
        return {"base": "", "model": "local-model", "runner": "", "auto": False}
    base = (os.environ.get("NEUROFORGE_LLM_BASE") or "").strip()
    model = (os.environ.get("NEUROFORGE_LLM_MODEL") or "").strip()
    runner = ""
    auto = False
    if base:
        local = find()
        runner = str(local["runner"]) if local and local["base"] == base else ""
    else:
        local = find()
        if local:
            base = str(local["base"])
            model = model or str(local["model"])
            runner = str(local["runner"])
            auto = True
    return {"base": base, "model": model or "local-model",
            "runner": runner, "auto": auto}


def small_model(model: str) -> bool:
    """Identify common small local model names for UI copy and prompting."""
    name = (model or "").lower()
    return any(token in name for token in ("0.5b", "1b", "1.5b", "3b", "tiny", "mini"))
