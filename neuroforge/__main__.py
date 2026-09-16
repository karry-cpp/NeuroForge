"""
python -m neuroforge   ->   launch the 3-D application.

  --subdiv N   cortical mesh detail (5 fast · 6 default · 7 very detailed)
  --port N     HTTP port
  --rebuild    force regeneration of the anatomical mesh cache
  --cli        run the original terminal prototype instead
"""

from __future__ import annotations

import argparse


def main() -> None:
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
