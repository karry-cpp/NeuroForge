"""
neuroforge.prototype_cli
========================

STAGE 1 - the minimal working prototype.

No GUI, no dependencies. It proves the model works: log behaviours, advance
days, watch the connection strengths of the CURRENT brain move toward the
TARGET brain, and see an ASCII version of both "brains" side by side.

Run:  python -m neuroforge.prototype_cli
      python -m neuroforge.prototype_cli --demo 12
"""

from __future__ import annotations

import argparse

from .atlas import PATHWAYS
from .engine import Simulation
from .events import CATEGORIES, EVENT_TYPES, EVENTS_BY_ID, events_in

BAR = "█"


def bar(v: float, width: int = 22) -> str:
    n = max(0, min(width, round(v * width)))
    return BAR * n + "·" * (width - n)


def render(sim: Simulation) -> None:
    print("\n" + "=" * 78)
    print(f" Day {sim.day:>3} ({sim.date_for(sim.day)})   "
          f"stress {sim.stress:.2f}   sleep {sim.sleep:.2f}   "
          f"streak {sim.streak()}d")
    print("=" * 78)
    print(f"{'PATHWAY':<26}{'CURRENT':<26}{'TARGET':<26}")
    print("-" * 78)
    for pid, pw in PATHWAYS.items():
        cur = sim.current.pathway_strength(pid)
        tgt = pw.target
        flag = "↑" if pw.desirable else "↓"
        name = pw.label if len(pw.label) <= 24 else pw.label[:23] + "…"
        print(f"{flag} {name:<24}{bar(cur)} {cur:.2f}   "
              f"{bar(tgt)} {tgt:.2f}")
    print("-" * 78)
    print(f"Current→Target alignment : {bar(sim.alignment, 30)} "
          f"{sim.alignment*100:5.1f}%")
    p = sim.progress
    print(f"Gap closed since day 0   : {bar(max(0,p), 30)} {p*100:5.1f}%")
    print("\n[metaphor] Bars are simulated pattern strengths, not "
          "measurements of your brain.")


def menu() -> None:
    sim = Simulation()
    render(sim)
    while True:
        print("\nLog what happened:")
        n = 1
        idx = {}
        for cat in CATEGORIES:
            items = events_in(cat)
            print(f"  -- {cat} --")
            for e in items:
                idx[n] = e.id
                print(f"   {n:>2}. {e.icon} {e.label}")
                n += 1
        print("   n. next day     w. skip a week     i <num>. info     q. quit")
        raw = input("> ").strip().lower()
        if raw == "q":
            break
        if raw == "n":
            sim.advance_day(); render(sim); continue
        if raw == "w":
            sim.advance_day(7); render(sim); continue
        if raw.startswith("i"):
            try:
                e = EVENTS_BY_ID[idx[int(raw.split()[1])]]
            except Exception:
                print("?"); continue
            print(f"\n{e.icon} {e.label}\n  RULE     : {e.rule}"
                  f"\n  SCIENCE  : {e.science}"
                  f"\n  NOT CLAIMING: {e.caveat or '-'}")
            continue
        try:
            sim.log_event(idx[int(raw)])
        except Exception:
            print("?"); continue
        render(sim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, metavar="WEEKS",
                    help="fast-forward a scripted practice history")
    ap.add_argument("--adherence", type=float, default=0.7)
    a = ap.parse_args()
    if a.demo:
        sim = Simulation()
        print("Day 0 baseline:")
        render(sim)
        sim.simulate_scripted(weeks=a.demo, adherence=a.adherence)
        print(f"\n\nAfter {a.demo} weeks at {a.adherence:.0%} adherence:")
        render(sim)
    else:
        menu()


if __name__ == "__main__":
    main()
