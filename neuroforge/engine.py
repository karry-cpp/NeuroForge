"""
neuroforge.engine
=================

Orchestration: holds the current brain, the target brain, the practice log,
the day counter and the history needed for the replay/timeline feature.

The engine is completely UI-independent - you can drive it from the CLI
prototype, the Tkinter app, a future Qt app, or a unit test.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

from .atlas import PATHWAYS
from .events import EVENTS_BY_ID, EventType, arousal_gain, sleep_gain
from .model import (Connectome, alignment, per_pathway_progress, progress)


@dataclass
class LogEntry:
    day: int
    event_id: str
    note: str = ""
    intensity: float = 1.0     # 0.25 .. 1.5 - "how much of it did you do?"


@dataclass
class DaySnapshot:
    day: int
    weights: Dict[str, float]
    pathways: Dict[str, float]
    alignment: float
    progress: float
    stress: float
    sleep: float
    events: List[str] = field(default_factory=list)


class Simulation:
    """The whole simulated organism."""

    def __init__(self, start_date: Optional[date] = None):
        self.current = Connectome()
        self.target = Connectome.target_brain()
        self.day: int = 0
        self.start_date: date = start_date or date.today()
        self.stress: float = 0.35
        self.sleep: float = 0.70
        self.log: List[LogEntry] = []
        self.history: List[DaySnapshot] = []
        self._start_vector = list(self.current.vector())
        self._start_pathways = self.current.pathway_profile()
        self._reps_today: Dict[str, int] = {}
        self._today_lr_bonus: float = 1.0
        self._today_events: List[str] = []
        self.listeners: List[Callable[[], None]] = []
        self._record()

    # ----------------------------------------------------------- events
    def subscribe(self, fn: Callable[[], None]) -> None:
        self.listeners.append(fn)

    def _emit(self) -> None:
        for fn in self.listeners:
            fn()

    # ------------------------------------------------------------ core
    def log_event(self, event_id: str, note: str = "",
                  intensity: float = 1.0) -> EventType:
        """Record one behaviour and update the simulation."""
        ev = EVENTS_BY_ID[event_id]

        # state effects
        self.stress = _clip01(self.stress + ev.stress_delta * intensity)
        if ev.sets_sleep is not None:
            self.sleep = ev.sets_sleep
        if ev.id == "exercise":
            self._today_lr_bonus = max(self._today_lr_bonus, 1.15)

        # plasticity
        if ev.activations:
            reps = self._reps_today.get(ev.id, 0)
            lr = (ev.lr_mult * self._today_lr_bonus
                  * arousal_gain(self.stress) * sleep_gain(self.sleep))
            acts = {k: v * intensity for k, v in ev.activations.items()}
            self.current.apply_practice(acts, lr_mult=lr,
                                        repetitions_today=reps,
                                        reward=ev.reward)
            self._reps_today[ev.id] = reps + 1

        self.log.append(LogEntry(self.day, event_id, note, intensity))
        self._today_events.append(event_id)
        self._record(replace=True)
        self._emit()
        return ev

    def advance_day(self, n: int = 1) -> None:
        for _ in range(n):
            practised = bool(self._today_events)
            self.current.tick_day(sleep_quality=self.sleep,
                                  stress=self.stress,
                                  practised_today=practised)
            # stress and sleep relax toward their own set-points
            self.stress += 0.12 * (0.30 - self.stress)
            self.sleep += 0.30 * (0.70 - self.sleep)
            self.day += 1
            self._reps_today.clear()
            self._today_lr_bonus = 1.0
            self._today_events = []
            self._record()
        self._emit()

    # ------------------------------------------------------- read-outs
    @property
    def alignment(self) -> float:
        return alignment(self.current)

    @property
    def progress(self) -> float:
        return progress(self.current, self._start_vector)

    def pathway_progress(self) -> Dict[str, float]:
        return per_pathway_progress(self.current, self._start_pathways)

    def date_for(self, day: int) -> date:
        return self.start_date + timedelta(days=day)

    def streak(self) -> int:
        """Consecutive days (ending today) with at least one Practice event."""
        practice_days = {e.day for e in self.log
                         if EVENTS_BY_ID[e.event_id].category == "Practice"}
        s, d = 0, self.day
        while d in practice_days:
            s += 1
            d -= 1
        return s

    # -------------------------------------------------------- history
    def _record(self, replace: bool = False) -> None:
        snap = DaySnapshot(
            day=self.day,
            weights=self.current.snapshot(),
            pathways={k: round(v, 4)
                      for k, v in self.current.pathway_profile().items()},
            alignment=round(self.alignment, 4),
            progress=round(self.progress, 4),
            stress=round(self.stress, 3),
            sleep=round(self.sleep, 3),
            events=list(self._today_events),
        )
        if replace and self.history and self.history[-1].day == self.day:
            self.history[-1] = snap
        else:
            self.history.append(snap)

    def snapshot_at(self, day: int) -> DaySnapshot:
        best = self.history[0]
        for s in self.history:
            if s.day <= day:
                best = s
            else:
                break
        return best

    # ---------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": 2,
            "day": self.day,
            "start_date": self.start_date.isoformat(),
            "stress": self.stress,
            "sleep": self.sleep,
            "weights": self.current.snapshot(),
            "log": [asdict(e) for e in self.log],
            "history": [asdict(h) for h in self.history],
            "start_vector": self._start_vector,
            "start_pathways": self._start_pathways,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Simulation":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        sim = cls(start_date=date.fromisoformat(d["start_date"]))
        sim.day = d["day"]
        sim.stress = d["stress"]
        sim.sleep = d["sleep"]
        sim.current.load(d["weights"])
        sim.log = [LogEntry(**e) for e in d["log"]]
        sim.history = [DaySnapshot(**h) for h in d["history"]]
        sim._start_vector = d.get("start_vector", sim._start_vector)
        sim._start_pathways = d.get("start_pathways", sim._start_pathways)
        return sim

    # ------------------------------------------------------------ demo
    def simulate_scripted(self, weeks: int = 8, adherence: float = 0.7,
                          seed: int = 7) -> None:
        """Fast-forward a plausible practice history (for demos/tests)."""
        import random
        rng = random.Random(seed)
        practice = ["name_emotion", "pause", "reappraisal", "disengage",
                    "mindful_practice", "regulated_success",
                    "behavioural_activation"]
        for _ in range(weeks * 7):
            self.log_event("sleep_good" if rng.random() < 0.7
                           else "sleep_poor")
            if rng.random() < 0.25:
                self.log_event("stress_high")
            if rng.random() < 0.30:
                self.log_event("exercise")
            if rng.random() < adherence:
                for _ in range(rng.randint(1, 3)):
                    self.log_event(rng.choice(practice))
                if rng.random() < 0.18:
                    self.log_event("exposure")
            if rng.random() < (1.0 - adherence) * 0.9:
                self.log_event(rng.choice(["rumination", "avoidance",
                                           "reactive_outburst"]))
            self.advance_day()


def _clip01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v
