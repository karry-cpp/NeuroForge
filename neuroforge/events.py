"""
neuroforge.events
=================

The rulebook: exactly what happens to the simulation when each behaviour is
logged. Every rule is explicit, inspectable and shown in the UI, so the user
can always see *why* a line got thicker.

------------------------------------------------------------------------------
SCIENCE / METAPHOR BOUNDARY
------------------------------------------------------------------------------
REAL: each event type below corresponds to a technique or state with genuine
      empirical support (affect labelling, cognitive reappraisal, response
      inhibition, exposure/extinction, behavioural activation, sleep and
      exercise effects on learning, stress effects on habit bias).
METAPHOR: the numbers attached to them are chosen so the simulation *feels*
      right over weeks of use. They are not effect sizes. Logging one
      reappraisal does not prune a neuron; it nudges a variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class EventType:
    id: str
    label: str
    category: str                      # Practice | Slip | State
    icon: str
    activations: Dict[str, float]      # pathway -> signed engagement
    lr_mult: float = 1.0               # local learning-rate multiplier
    reward: float = 0.0                # RPE-like consolidation boost
    stress_delta: float = 0.0          # effect on the stress scalar
    sets_sleep: float | None = None    # if this event reports sleep
    rule: str = ""                     # human-readable rule
    science: str = ""                  # what is actually known
    caveat: str = ""                   # what we are NOT claiming


def _e(*a, **k) -> EventType:
    return EventType(*a, **k)


EVENT_TYPES: List[EventType] = [

    # ---------------------------------------------------------------- practice
    _e("name_emotion", "Named the emotion", "Practice", "🏷",
       {"regulation": 0.70, "context": 0.20, "rumination": -0.20},
       lr_mult=1.0, stress_delta=-0.03,
       rule="regulation +0.70 · context +0.20 · rumination −0.20",
       science="Affect labelling ('putting feelings into words') is "
               "associated with increased right vlPFC activity and reduced "
               "amygdala response. It is a low-effort technique that "
               "reliably takes some heat out of an emotional state.",
       caveat="The app is not detecting that your amygdala calmed down. It "
              "is applying a fixed rule you can read above."),

    _e("reappraisal", "Reframed the situation", "Practice", "🔄",
       {"regulation": 0.95, "goal_directed": 0.35, "context": 0.25,
        "rumination": -0.30},
       lr_mult=1.05, reward=0.3, stress_delta=-0.06,
       rule="regulation +0.95 · goal-directed +0.35 · context +0.25 · "
            "rumination −0.30",
       science="Cognitive reappraisal is one of the best-supported emotion "
               "regulation strategies: changing the meaning of a situation "
               "changes the emotional response, and is accompanied by "
               "prefrontal recruitment and reduced amygdala response.",
       caveat="Reappraisal is effortful and works better for situations you "
              "cannot change. It is not universally superior."),

    _e("pause", "Paused before reacting", "Practice", "⏸",
       {"regulation": 0.75, "goal_directed": 0.30, "new_habit": 0.25,
        "stress_habit": -0.30, "threat": -0.10},
       stress_delta=-0.04,
       rule="regulation +0.75 · goal-directed +0.30 · new habit +0.25 · "
            "stress autopilot −0.30",
       science="Inserting a delay between stimulus and response engages "
               "inhibitory control (right IFG / vlPFC) and gives slower "
               "evaluative processes time to contribute.",
       caveat="'Inhibition' is modelled as one number; the real system has "
              "many partly separable inhibitory mechanisms."),

    _e("disengage", "Chose not to engage with a thought", "Practice", "🌊",
       {"rumination": -0.60, "regulation": 0.45, "new_habit": 0.15},
       rule="rumination −0.60 · regulation +0.45 · new habit +0.15",
       science="Not elaborating a thought means it is not rehearsed, and "
               "unrehearsed retrieval routes lose their competitive "
               "advantage. Attentional disengagement training shows modest "
               "but real effects.",
       caveat="This is the app's most 'pruning-flavoured' rule. Nothing is "
              "deleted: a competing route simply stops being reinforced."),

    _e("exposure", "Faced something anxiety-provoking", "Practice", "🚪",
       {"context": 1.00, "regulation": 0.55, "goal_directed": 0.30,
        "threat": -0.30, "stress_habit": -0.25},
       lr_mult=1.35, reward=0.6, stress_delta=0.10,
       rule="context +1.00 · regulation +0.55 · goal-directed +0.30 · "
            "threat −0.30   (learning rate ×1.35, high consolidation)",
       science="Exposure works by inhibitory learning: a new 'safe here' "
               "memory is formed that competes with the old threat memory. "
               "Learning is strongest when expectancy is violated and some "
               "arousal is present - which is why this event carries the "
               "largest learning-rate multiplier and also raises stress.",
       caveat="Extinction does NOT erase the original association. Fear can "
              "return with time, context change or a bad experience. The "
              "model reflects this: threat weights drift back up if the "
              "practice stops."),

    _e("regulated_success", "Regulated successfully in the moment",
       "Practice", "✅",
       {"regulation": 0.65, "new_habit": 0.75, "goal_directed": 0.35,
        "context": 0.30, "rumination": -0.35, "threat": -0.15},
       lr_mult=1.1, reward=1.0, stress_delta=-0.08,
       rule="new habit +0.75 · regulation +0.65 · … · rumination −0.35   "
            "(max reward → strong consolidation)",
       science="A better-than-expected outcome produces a positive reward "
               "prediction error, and dopaminergic signalling reinforces "
               "whatever preceded it. This is the step that turns a "
               "technique into a default.",
       caveat="Self-reported success is not an objective outcome measure."),

    _e("mindful_practice", "Meditation / breathing practice", "Practice", "🧘",
       {"regulation": 0.50, "context": 0.35, "new_habit": 0.20,
        "threat": -0.20, "rumination": -0.25},
       stress_delta=-0.10,
       rule="regulation +0.50 · context +0.35 · threat −0.20 · "
            "rumination −0.25",
       science="Mindfulness training shows small-to-moderate effects on "
               "rumination and emotional reactivity; long-term practitioners "
               "show altered prefrontal-amygdala coupling. Effect sizes in "
               "well-controlled trials are modest.",
       caveat="Claims about meditation 'rewiring the brain' are routinely "
              "overstated in popular media."),

    _e("behavioural_activation", "Did the thing anyway (activation)",
       "Practice", "🏃",
       {"goal_directed": 0.70, "new_habit": 0.45, "stress_habit": -0.25,
        "rumination": -0.20},
       reward=0.45,
       rule="goal-directed +0.70 · new habit +0.45 · stress autopilot −0.25",
       science="Behavioural activation - acting according to goals rather "
               "than mood - has strong support in depression treatment and "
               "restores contact with reinforcement.",
       caveat=""),

    # ------------------------------------------------------------------ slips
    _e("rumination", "Ruminated / spiralled", "Slip", "🌀",
       {"rumination": 1.00, "threat": 0.45, "regulation": -0.20,
        "context": -0.15},
       stress_delta=0.10,
       rule="rumination +1.00 · threat +0.45 · regulation −0.20 · "
            "context −0.15",
       science="Perseverative thinking rehearses a negative interpretation, "
               "which makes it more accessible later, and sustains "
               "physiological arousal well after the trigger is gone.",
       caveat="A rumination episode is not brain damage. In this model it is "
              "fully reversible - and it is normal for the line to get "
              "thicker sometimes."),

    _e("avoidance", "Avoided the situation", "Slip", "🚫",
       {"stress_habit": 0.75, "threat": 0.45, "context": -0.40,
        "regulation": -0.15},
       reward=0.5,   # relief IS reinforcing - that is the trap
       stress_delta=0.05,
       rule="stress autopilot +0.75 · threat +0.45 · context −0.40   "
            "(and it gets a *reward* bonus, because relief reinforces)",
       science="Avoidance is negatively reinforced: the immediate relief is "
               "rewarding, which strengthens avoidance and prevents the "
               "disconfirming evidence that would update the threat "
               "expectation. This is the central maintaining mechanism in "
               "anxiety disorders.",
       caveat="Avoidance is sometimes the correct choice. The model is not "
              "a moral judgement."),

    _e("reactive_outburst", "Reacted automatically / lashed out", "Slip", "💥",
       {"threat": 0.65, "stress_habit": 0.70, "regulation": -0.30,
        "goal_directed": -0.25},
       stress_delta=0.12,
       rule="stress autopilot +0.70 · threat +0.65 · regulation −0.30 · "
            "goal-directed −0.25",
       science="Under high arousal, prefrontally-guided control degrades and "
               "well-learned stimulus-response behaviour dominates.",
       caveat=""),

    # ------------------------------------------------------------------ state
    _e("sleep_good", "Slept well", "State", "🌙", {},
       sets_sleep=0.92, stress_delta=-0.08,
       rule="sets sleep quality = 0.92 → stronger overnight consolidation, "
            "slower decay",
       science="Sleep supports consolidation of both declarative and "
               "procedural learning; hippocampal replay during slow-wave "
               "sleep and REM-related processing are implicated. Sleep "
               "deprivation increases amygdala reactivity and weakens "
               "prefrontal-amygdala coupling.",
       caveat="Sleep is modelled as a single 0-1 number per day."),

    _e("sleep_poor", "Slept badly", "State", "😵", {},
       sets_sleep=0.30, stress_delta=0.10,
       rule="sets sleep quality = 0.30 → weak consolidation, faster decay",
       science="See above - the effect is bidirectional and quite large in "
               "experimental studies.",
       caveat=""),

    _e("exercise", "Exercised", "State", "💪", {"goal_directed": 0.15}, 
       lr_mult=1.0, stress_delta=-0.10,
       rule="stress −0.10 and today's learning rate ×1.15 (applied globally)",
       science="Aerobic exercise acutely improves executive function and is "
               "associated with elevated BDNF, a molecule involved in "
               "synaptic plasticity. Human evidence for the BDNF pathway "
               "specifically is indirect.",
       caveat="'Exercise grows your brain' is a large oversimplification."),

    _e("stress_high", "High stress day", "State", "⚡", {},
       stress_delta=0.28,
       rule="stress +0.28 → learning rate follows an inverted-U; threat and "
            "autopilot pathways drift upward each day stress stays high",
       science="Acute stress can enhance amygdala-dependent learning while "
               "impairing prefrontal function, and shifts behaviour toward "
               "habit. Chronic stress produces dendritic remodelling in "
               "animal models - reversible in many cases.",
       caveat="The app has no idea what your cortisol is doing."),

    _e("recovery", "Rest / recovery / connection", "State", "🫧", {},
       stress_delta=-0.20,
       rule="stress −0.20",
       science="Social support and recovery time buffer stress responses.",
       caveat=""),
]

EVENTS_BY_ID: Dict[str, EventType] = {e.id: e for e in EVENT_TYPES}
CATEGORIES = ["Practice", "Slip", "State"]


def events_in(category: str) -> List[EventType]:
    return [e for e in EVENT_TYPES if e.category == category]


# --------------------------------------------------------------------------
# Global modulators
# --------------------------------------------------------------------------


def arousal_gain(stress: float) -> float:
    """Inverted-U: learning is best at moderate arousal.

    REAL: the Yerkes-Dodson / catecholamine inverted-U for
    prefrontally-dependent performance is well documented.
    METAPHOR: this exact Gaussian is invented.
    """
    peak, width = 0.40, 0.30
    return 0.45 + 0.85 * pow(2.718281828, -((stress - peak) ** 2) /
                             (2 * width * width))


def sleep_gain(sleep_quality: float) -> float:
    """Poor sleep does not stop learning, it stops it *sticking*."""
    return 0.6 + 0.6 * sleep_quality
