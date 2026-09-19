"""
neuroforge.model
================

The plasticity mathematics.

------------------------------------------------------------------------------
SCIENCE / METAPHOR BOUNDARY
------------------------------------------------------------------------------
REAL principles that motivate the equations:
  * Use-dependent strengthening: co-active pathways that are repeatedly
    engaged become more efficient (Hebbian learning, LTP/LTD).
  * Soft bounds / saturation: strengthening is not unlimited.
  * Two time-scales: a labile, fast-changing component and a slower,
    consolidated component that resists forgetting.
  * Synaptic homeostasis: total drive onto a target is regulated, so
    strengthening one input effectively competes with others.
  * Sleep-dependent consolidation: replay during sleep stabilises new
    learning.
  * Spacing effect: distributed practice beats massed practice.
  * Inverted-U arousal: moderate arousal aids learning, extreme arousal
    impairs prefrontally-dependent learning.
  * Reconsolidation-like lability: reactivating a pattern briefly makes it
    updatable.
METAPHOR: the specific equations, constants and units are invented for this
app. They are a *cartoon* of the above principles. Nothing here estimates
synapse counts, gray matter, dendritic spines or real-time brain activity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .atlas import EDGES, PATHWAYS, EdgeSpec, target_weight

# --------------------------------------------------------------------------
# Tunable constants (all dimensionless, all invented)
# --------------------------------------------------------------------------


@dataclass
class Params:
    base_lr: float = 0.030          # learning rate per practice repetition
    unlearn_ratio: float = 0.85     # weakening is slightly slower than growth
    daily_decay: float = 0.012      # "use it or lose it" per day
    baseline_pull: float = 0.006    # drift back toward temperament per day
    consolidation_gain: float = 0.09
    consolidation_decay: float = 0.004
    eligibility_decay: float = 0.55  # per day
    homeostasis: float = 0.16       # strength of input competition
    input_budget: float = 1.35      # target total incoming strength per node
    w_min: float = 0.02
    w_max: float = 0.99
    spacing_halflife: float = 1.6   # nth rep today is worth ~1/(1+r/halflife)
    stress_drift: float = 0.010     # chronic stress pushes threat circuits up
    # Spread of the per-practice gain. Two people doing the same thing, or
    # the same person on two days, do not get the same result: attention,
    # motivation and context all vary and none of them are logged here.
    # Lognormal with mean 1.0, so this adds uncertainty without quietly
    # inflating or deflating the average outcome.
    session_variance: float = 0.30


PARAMS = Params()


# --------------------------------------------------------------------------
# A single simulated connection
# --------------------------------------------------------------------------


@dataclass
class Connection:
    spec: EdgeSpec
    w: float                  # simulated strength, [0,1]
    c: float = 0.05           # consolidation, [0,1]: resistance to decay
    e: float = 0.0            # eligibility trace: recent practice, [0,~3]
    activity: float = 0.0     # visual pulse energy, decays fast

    @property
    def key(self) -> Tuple[str, str]:
        return (self.spec.src, self.spec.dst)

    @property
    def pathway(self) -> str:
        return self.spec.pathway

    @property
    def target(self) -> float:
        return target_weight(self.spec)


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


class Connectome:
    """A set of connections plus the plasticity rules that update them."""

    def __init__(self, weights: Dict[str, float] | None = None,
                 params: Params | None = None):
        self.p = params or PARAMS
        self.conns: List[Connection] = []
        for spec in EDGES:
            k = f"{spec.src}->{spec.dst}"
            w = weights[k] if weights and k in weights else spec.w0
            self.conns.append(Connection(spec=spec, w=w))
        self.by_pathway: Dict[str, List[Connection]] = {}
        for c in self.conns:
            self.by_pathway.setdefault(c.pathway, []).append(c)
        self._incoming: Dict[str, List[Connection]] = {}
        for c in self.conns:
            self._incoming.setdefault(c.spec.dst, []).append(c)

    # -- construction helpers ------------------------------------------
    @classmethod
    def target_brain(cls) -> "Connectome":
        """The TARGET brain: every connection sitting at its pathway target.

        This is an *aspiration encoded as numbers*, not a claim about what a
        healthy brain looks like on a scan.
        """
        cm = cls()
        for c in cm.conns:
            c.w = c.target
            c.c = 0.85
        return cm

    def snapshot(self) -> Dict[str, float]:
        # 8dp, not 5: equilibrate() leaves the starting weights on arbitrary
        # floats rather than the 2dp literals in atlas.py, and at 5dp a
        # save/load round trip no longer reproduced alignment to 6 places.
        return {f"{c.spec.src}->{c.spec.dst}": round(c.w, 8)
                for c in self.conns}

    def load(self, snap: Dict[str, float]) -> None:
        for c in self.conns:
            k = f"{c.spec.src}->{c.spec.dst}"
            if k in snap:
                c.w = snap[k]

    # -- core learning step --------------------------------------------
    def apply_practice(self,
                       activations: Dict[str, float],
                       lr_mult: float = 1.0,
                       repetitions_today: int = 0,
                       reward: float = 0.0) -> None:
        """One logged behaviour.

        activations: pathway_id -> signed engagement in roughly [-1, +1].
                     +1 = "this circuit was strongly run"
                     -1 = "this circuit was actively NOT engaged"
        lr_mult:     global learning-rate multiplier (sleep, stress, exercise,
                     novelty, dopamine).
        repetitions_today: implements the spacing effect - the 4th repetition
                     of the same thing in one day teaches less than the 1st.
        reward:      reward-prediction-error-like term that boosts
                     consolidation of whatever was just practised.
        """
        p = self.p
        spacing = 1.0 / (1.0 + repetitions_today / p.spacing_halflife)
        eta = p.base_lr * lr_mult * spacing

        touched: set[str] = set()
        for pid, a in activations.items():
            if pid not in self.by_pathway or a == 0.0:
                continue
            for c in self.by_pathway[pid]:
                touched.add(c.spec.dst)
                drive = eta * a
                if drive > 0:
                    # soft upper bound: the closer to 1, the harder to gain
                    dw = drive * (1.0 - c.w)
                else:
                    dw = drive * p.unlearn_ratio * (c.w - p.w_min)
                c.w = _clip(c.w + dw, p.w_min, p.w_max)
                # eligibility: only *practice* (not disengagement) leaves a
                # trace to be consolidated overnight
                c.e += max(0.0, a) * (1.0 + 0.8 * max(0.0, reward)) * spacing
                c.activity = min(2.0, c.activity + abs(a))

        if touched:
            self._homeostasis(touched)

    def equilibrate(self, rounds: int = 16) -> None:
        """Settle the starting weights into homeostatic balance.

        The hand-set starting weights in atlas.py do not happen to satisfy
        the input budget, so without this the first practice to touch a node
        also paid off that imbalance in one step - one repetition moved the
        model several percent no matter how small the learning rate was. A
        brain is already balanced before anyone starts practising.
        """
        for _ in range(rounds):
            self._homeostasis()

    def _homeostasis(self, nodes: Iterable[str] | None = None) -> None:
        """Competition for a limited 'input budget' at each target node.

        REAL: synaptic scaling keeps total drive onto a neuron in a workable
        range; strengthening some inputs relatively weakens others.
        METAPHOR: this is why practising vlPFC->BLA regulation *passively*
        weakens the dACC->BLA rumination input - they compete for the same
        postsynaptic target. It is the honest version of "pruning": nothing
        is deleted, it just loses the competition.
        """
        p = self.p
        keys = self._incoming.keys() if nodes is None else nodes
        for dst in keys:
            group = self._incoming.get(dst, ())
            if len(group) < 2:
                continue
            total = sum(c.w for c in group)
            if total <= 1e-6:
                continue
            excess = (total - p.input_budget) / total
            if abs(excess) < 1e-9:
                continue
            for c in group:
                # consolidated connections resist being scaled down
                resist = 1.0 - 0.6 * c.c
                c.w = _clip(c.w - p.homeostasis * excess * c.w * resist,
                            p.w_min, p.w_max)

    # -- the passage of time -------------------------------------------
    def tick_day(self, sleep_quality: float = 0.7, stress: float = 0.3,
                 practised_today: bool = False) -> None:
        """Advance the simulation by one day.

        REAL: without rehearsal, newly acquired patterns are forgotten; sleep
        (especially slow-wave and REM) supports consolidation via replay;
        chronic stress biases threat circuitry.
        METAPHOR: the exact rates below are invented.
        """
        p = self.p
        for c in self.conns:
            # 1. consolidation: recent practice + sleep -> durability
            gain = p.consolidation_gain * math.tanh(c.e) * (0.35 + sleep_quality)
            c.c = _clip(c.c + gain - p.consolidation_decay, 0.0, 0.98)

            # 2. decay - only what is not consolidated is fragile
            fragile = 1.0 - c.c
            decay = p.daily_decay * fragile * (1.25 - 0.5 * sleep_quality)
            c.w = _clip(c.w - decay * (c.w - p.w_min), p.w_min, p.w_max)

            # 3. drift back toward temperamental baseline
            c.w += p.baseline_pull * (c.spec.baseline - c.w)

            # 4. chronic stress quietly reinforces threat / autopilot
            if c.pathway in ("threat", "stress_habit", "rumination"):
                c.w = _clip(c.w + p.stress_drift * max(0.0, stress - 0.45)
                            * (1.0 - c.w), p.w_min, p.w_max)

            # 5. traces fade
            c.e *= p.eligibility_decay
            c.activity *= 0.4
        # NOTE: no homeostatic rescaling here. Competition between inputs is
        # driven by *use*, not by the calendar - otherwise an unwanted
        # pathway would fade simply because time passed, which would be both
        # wrong and quietly dishonest.

    # -- read-outs ------------------------------------------------------
    def pathway_strength(self, pid: str) -> float:
        g = self.by_pathway.get(pid, [])
        return sum(c.w for c in g) / len(g) if g else 0.0

    def pathway_profile(self) -> Dict[str, float]:
        return {pid: self.pathway_strength(pid) for pid in PATHWAYS}

    def vector(self) -> List[float]:
        return [c.w for c in self.conns]

    def target_vector(self) -> List[float]:
        return [c.target for c in self.conns]


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------
# Current -> Target similarity
# --------------------------------------------------------------------------


def alignment(cm: Connectome) -> float:
    """0..1 'how close is the current pattern to the chosen target'.

    Plain mean absolute difference across connections, inverted.
    """
    v, t = cm.vector(), cm.target_vector()
    return 1.0 - sum(abs(a - b) for a, b in zip(v, t)) / len(v)


def progress(cm: Connectome, start_vector: List[float]) -> float:
    """0..1 'how much of the original gap has been closed'.

    Uses the distance at day 0 as the denominator, so it starts at 0 and
    reaches 1 when the target is met. Can go slightly negative if things
    move backwards, which is honest and worth showing.
    """
    t = cm.target_vector()
    d0 = sum(abs(a - b) for a, b in zip(start_vector, t))
    dn = sum(abs(a - b) for a, b in zip(cm.vector(), t))
    if d0 <= 1e-9:
        return 1.0
    return _clip(1.0 - dn / d0, -0.5, 1.0)


def per_pathway_progress(cm: Connectome,
                         start: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for pid, pw in PATHWAYS.items():
        cur = cm.pathway_strength(pid)
        s = start.get(pid, cur)
        d0 = abs(s - pw.target)
        dn = abs(cur - pw.target)
        out[pid] = 1.0 if d0 < 1e-9 else _clip(1.0 - dn / d0, -0.5, 1.0)
    return out
