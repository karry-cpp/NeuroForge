"""
python -m neuroforge   ->   launch the 3-D application.

  --subdiv N   cortical mesh detail (5 fast · 6 default · 7 very detailed)
  --port N     HTTP port
  --rebuild    force regeneration of the anatomical mesh cache
  --cli        run the original terminal prototype instead
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_env() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    names = {"NEUROFORGE_CHAT_BASE", "NEUROFORGE_CHAT_MODEL",
             "NEUROFORGE_CHAT_KEY", "NEUROFORGE_LLM_BASE",
             "NEUROFORGE_LLM_MODEL", "NEUROFORGE_LLM_KEY"}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in names and value.strip() and not line.lstrip().startswith("#"):
            os.environ.setdefault(name, value.strip())
    # Log interpretation and analysis read NEUROFORGE_LLM_*; default them to the chat provider.
    for part in ("BASE", "MODEL", "KEY"):
        chat = os.environ.get(f"NEUROFORGE_CHAT_{part}")
        if chat:
            os.environ.setdefault(f"NEUROFORGE_LLM_{part}", chat)


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(
        prog="neuroforge",
        description="NeuroForge - an interactive 3-D neuroscience simulation.")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--subdiv", type=int, default=6,
                    help="cortical mesh detail: 5=fast, 6=default, 7=heavy")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate the anatomy cache")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cli", action="store_true",
                    help="run the original terminal prototype")
    a = ap.parse_args()

    if a.cli:
        from .prototype_cli import main as cli_main
        cli_main()
        return

    if a.rebuild:
        from .anatomy.build import get_scene
        get_scene(a.subdiv, rebuild=True)

    from .server.app import serve
    serve(port=a.port, open_browser=not a.no_browser, subdivisions=a.subdiv)


if __name__ == "__main__":
    main()
