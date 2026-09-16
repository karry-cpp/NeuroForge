"""
neuroforge.atlas
================

The *static* knowledge base of the simulation: which simplified brain regions
exist, how they are laid out on screen, and which hypothetical functional
connections we model.

------------------------------------------------------------------------------
SCIENCE / METAPHOR BOUNDARY  (read this before you read anything else)
------------------------------------------------------------------------------
REAL:      The regions and the broad direction of the connections below are
           drawn from well-replicated findings in affective neuroscience
           (e.g. prefrontal-amygdala regulation, hippocampal contextual
           modulation of threat, dorsomedial -> dorsolateral striatal shift
           during habit formation).
METAPHOR:  Every *number* in this file is invented. `w` ("connection
           strength") is a dimensionless variable in [0, 1] representing the
           strength of a learned behavioural pattern **inside this
           simulation**. It is NOT a synapse count, NOT gray-matter volume,
           NOT BOLD signal, NOT connectivity measured from your brain.
           The 2-D coordinates are a schematic, not an anatomical atlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Pathways: named functional circuits made of several connections.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pathway:
    id: str
    label: str
    desirable: bool          # do we want this to grow?
    target: float            # target connection strength in [0,1]
    color: str               # hex colour used for its edges
    science: str             # what is actually known
    metaphor: str            # what the app is doing with it


PATHWAYS: Dict[str, Pathway] = {
    "regulation": Pathway(
        "regulation", "Top-down regulation", True, 0.90, "#4cc9f0",
        science=(
            "Ventrolateral and ventromedial prefrontal areas are reliably "
            "recruited during affect labelling, reappraisal and extinction "
            "recall, and their activity is inversely correlated with "
            "amygdala reactivity. The influence is largely indirect "
            "(via intercalated cells of the amygdala and other relays), "
            "not a single monosynaptic 'off switch'."
        ),
        metaphor=(
            "One numeric strength value for 'how accessible is deliberate "
            "regulation right now'. Rises with practice, decays with disuse."
        ),
    ),
    "context": Pathway(
        "context", "Contextual safety learning", True, 0.85, "#80ffdb",
        science=(
            "The hippocampus supplies context: it gates whether a cue is "
            "treated as dangerous *here and now*. Extinction is new "
            "context-dependent inhibitory learning, not erasure of the "
            "original memory - which is why fear can return outside the "
            "learning context (renewal) or after time (spontaneous recovery)."
        ),
        metaphor=(
            "Strength of 'this situation is being read as present-tense "
            "rather than as a rerun of the past'."
        ),
    ),
    "goal_directed": Pathway(
        "goal_directed", "Goal-directed control", True, 0.80, "#a0c4ff",
        science=(
            "Dorsolateral PFC maintains goals in working memory; dorsomedial "
            "striatum supports action-outcome (goal-directed) control. "
            "Dorsal ACC signals conflict/need for control and helps recruit "
            "dlPFC."
        ),
        metaphor="Strength of effortful, deliberate action selection.",
    ),
    "new_habit": Pathway(
        "new_habit", "Automatised new routine", True, 0.75, "#b8f2a6",
        science=(
            "With repetition, control shifts from dorsomedial to "
            "dorsolateral striatum and behaviour becomes stimulus-response "
            "(habitual) and less outcome-sensitive. Dopaminergic input from "
            "VTA/SNc reinforces what just worked. In humans this takes weeks "
            "to months, with wide individual variability."
        ),
        metaphor=(
            "How automatic the *new*, regulated response has become - the "
            "'becomes easier to access' step of your flow diagram."
        ),
    ),
    "rumination": Pathway(
        "rumination", "Rumination / threat appraisal loop", False, 0.15,
        "#ff6b6b",
        science=(
            "Perseverative negative thought is associated with sustained "
            "amygdala responding, altered default-mode network dynamics and "
            "reduced flexible switching. Repeated retrieval of a negative "
            "interpretation makes that interpretation easier to retrieve "
            "again (retrieval-induced strengthening)."
        ),
        metaphor=(
            "A self-reinforcing loop whose strength grows each time it is "
            "run and shrinks when the response is not engaged."
        ),
    ),
    "threat": Pathway(
        "threat", "Rapid threat response", False, 0.50, "#ffb703",
        science=(
            "Basolateral amygdala -> central amygdala -> hypothalamic/"
            "brainstem outputs drive autonomic and behavioural threat "
            "responses. This circuit is ADAPTIVE. The goal is calibration, "
            "not elimination."
        ),
        metaphor=(
            "Deliberately given a mid-range target (0.50), not 0. A brain "
            "with no alarm system is not a healthy brain."
        ),
    ),
    "stress_habit": Pathway(
        "stress_habit", "Stress-driven autopilot", False, 0.20, "#c77dff",
        science=(
            "Acute and chronic stress bias behaviour away from goal-directed "
            "control toward habitual responding, partly via "
            "glucocorticoid/noradrenergic effects on prefrontal function."
        ),
        metaphor="How strongly stress hijacks the steering wheel.",
    ),
}


# --------------------------------------------------------------------------
# Regions (nodes)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    group: str
    x: float                 # schematic coords in [0,1]; x small = anterior
    y: float
    radius: float
    color: str
    role: str                # one-line plain-English role
    science: str             # what is actually established
    caveat: str              # what this app is NOT claiming


def _R(*a, **k) -> Region:
    return Region(*a, **k)


REGIONS: Dict[str, Region] = {r.id: r for r in [
    _R("dlPFC", "Dorsolateral prefrontal cortex", "PFC",
       0.28, 0.22, 26, "#4cc9f0",
       "Holds the goal in mind while you do something hard.",
       "Working-memory maintenance and top-down biasing of attention and "
       "action selection. Performance follows an inverted-U with arousal "
       "(too little or too much noradrenaline/dopamine both impair it).",
       "The node does not track your dlPFC. It tracks a simulated variable."),
    _R("vlPFC", "Ventrolateral prefrontal cortex", "PFC",
       0.20, 0.38, 24, "#4895ef",
       "Puts the brakes on. Also does the work of naming a feeling.",
       "Right vlPFC/IFG is consistently engaged during response inhibition "
       "(stop-signal, go/no-go) and during affect labelling, where its "
       "activity inversely tracks amygdala response.",
       "'Inhibition' here is a modelled number, not measured inhibition."),
    _R("vmPFC", "Ventromedial prefrontal cortex", "PFC",
       0.19, 0.56, 24, "#3f37c9",
       "Signals 'this is actually safe' and values outcomes.",
       "Central to extinction recall and safety signalling; works with the "
       "hippocampus to make safety context-dependent. Also encodes "
       "subjective value.",
       "Extinction is new learning layered over old learning, not deletion."),
    _R("dACC", "Dorsal anterior cingulate cortex", "PFC",
       0.40, 0.18, 22, "#7209b7",
       "Notices that something is off and calls for more control.",
       "Conflict / error / surprise monitoring and signalling the need for "
       "cognitive control; also contributes to affective salience.",
       "Included because control has to be *triggered* by something."),
    _R("BLA", "Basolateral amygdala", "Amygdala",
       0.52, 0.62, 24, "#ff6b6b",
       "Decides how emotionally important something is.",
       "Associative hub for emotional salience and fear/threat learning; "
       "receives cortical and hippocampal input, projects to CeA.",
       "Not 'the fear centre'. It handles salience broadly, including "
       "appetitive learning."),
    _R("CeA", "Central amygdala", "Amygdala",
       0.62, 0.70, 20, "#e63946",
       "Pulls the alarm: heart rate, freezing, startle.",
       "Output nucleus driving autonomic and behavioural threat responses "
       "via hypothalamic and brainstem targets.",
       "The app shows an arrow, not your actual autonomic state."),
    _R("HPC", "Hippocampus", "Hippocampus",
       0.72, 0.56, 24, "#80ffdb",
       "Supplies context: where, when, is this the same as last time?",
       "Episodic/contextual memory, pattern separation vs. completion, and "
       "context-dependent gating of extinction. Also central to memory "
       "consolidation during sleep (replay).",
       "Pattern separation is a real computational idea; the number here is "
       "not a measurement of it."),
    _R("DMS", "Dorsomedial striatum (caudate)", "BasalGanglia",
       0.44, 0.42, 22, "#a0c4ff",
       "Acts on purpose, checking whether the outcome is still worth it.",
       "Action-outcome / goal-directed control. Dominates early learning.",
       "Rodent DMS ~ human associative caudate; the mapping is approximate."),
    _R("DLS", "Dorsolateral striatum (putamen)", "BasalGanglia",
       0.56, 0.36, 22, "#b8f2a6",
       "Runs the routine automatically once it is well learned.",
       "Stimulus-response / habitual control; takes over with extended "
       "repetition. The DMS->DLS shift is the classic habit-formation "
       "signature.",
       "'Automatic' in this app is a slider, not a measured behaviour."),
    _R("NAcc", "Nucleus accumbens (ventral striatum)", "BasalGanglia",
       0.36, 0.52, 20, "#ffd6a5",
       "Registers 'that went better than expected' and reinforces it.",
       "Site of dopaminergic reward-prediction-error signalling that "
       "reinforces preceding actions; also negative reinforcement (relief "
       "from avoidance).",
       "Relief-based reinforcement is why avoidance is so sticky."),
    _R("VTA", "Ventral tegmental area", "Neuromodulator",
       0.74, 0.78, 17, "#fca311",
       "Dopamine: teaches from surprise.",
       "Phasic dopamine approximates a reward prediction error and gates "
       "plasticity in striatal and cortical targets.",
       "Modelled as a global learning-rate multiplier, not a transmitter."),
    _R("LC", "Locus coeruleus", "Neuromodulator",
       0.86, 0.72, 17, "#ffafcc",
       "Noradrenaline: arousal and alertness.",
       "Tonic/phasic noradrenergic signalling; the inverted-U relationship "
       "between arousal and prefrontal function is well documented.",
       "Drives an inverted-U learning-rate curve in the simulation."),
    _R("THAL", "Thalamus (motor loop relay)", "BasalGanglia",
       0.62, 0.24, 18, "#cdb4db",
       "The output gate: what actually gets executed.",
       "Cortico-basal ganglia-thalamo-cortical loops gate action selection.",
       "Simplified to a single 'action output' node."),
]}


# --------------------------------------------------------------------------
# Connections (edges)
# --------------------------------------------------------------------------


@dataclass
class EdgeSpec:
    src: str
    dst: str
    pathway: str
    label: str
    w0: float                # starting strength in the CURRENT brain
    baseline: float          # temperamental set-point it drifts back toward
    science: str
    curve: float = 0.0       # visual bow of the drawn line


EDGES: List[EdgeSpec] = [
    # --- top-down regulation -------------------------------------------
    EdgeSpec("vlPFC", "BLA", "regulation", "inhibit / label", 0.30, 0.30,
             "Affect labelling and inhibition covary with reduced amygdala "
             "response; the pathway is functional and largely indirect.",
             curve=0.18),
    EdgeSpec("vmPFC", "BLA", "regulation", "safety / extinction recall",
             0.28, 0.30,
             "vmPFC drives extinction *recall*; damage impairs retaining "
             "extinction across days.", curve=0.10),
    EdgeSpec("dlPFC", "vlPFC", "regulation", "recruit control", 0.40, 0.40,
             "Lateral prefrontal areas act as a cooperative control network.",
             curve=-0.10),
    # --- contextual safety ---------------------------------------------
    EdgeSpec("HPC", "vmPFC", "context", "context -> safety", 0.32, 0.32,
             "Hippocampal-vmPFC interaction makes extinction "
             "context-dependent.", curve=-0.22),
    EdgeSpec("HPC", "DMS", "context", "context -> action choice", 0.30, 0.30,
             "Contextual information constrains which goal-directed action "
             "is appropriate.", curve=0.18),
    # --- goal-directed control -----------------------------------------
    EdgeSpec("dACC", "dlPFC", "goal_directed", "conflict -> control",
             0.45, 0.45,
             "Conflict/error signals recruit lateral prefrontal control.",
             curve=0.12),
    EdgeSpec("dlPFC", "DMS", "goal_directed", "goal -> deliberate action",
             0.38, 0.38,
             "Prefrontal input to associative striatum supports "
             "action-outcome control.", curve=0.10),
    EdgeSpec("DMS", "THAL", "goal_directed", "deliberate output", 0.40, 0.40,
             "Cortico-striato-thalamic gating of chosen actions.",
             curve=0.10),
    # --- automatisation of the new routine -----------------------------
    EdgeSpec("DMS", "DLS", "new_habit", "deliberate -> automatic",
             0.18, 0.18,
             "The dorsomedial->dorsolateral shift with extended training is "
             "the canonical habit-formation finding.", curve=-0.12),
    EdgeSpec("NAcc", "DLS", "new_habit", "reinforce routine", 0.20, 0.20,
             "Ventral-to-dorsal striatal 'spiral' via dopamine supports "
             "progressive automatisation.", curve=0.16),
    EdgeSpec("DLS", "THAL", "new_habit", "automatic output", 0.22, 0.22,
             "Habitual responses are executed with little deliberation.",
             curve=-0.10),
    EdgeSpec("VTA", "NAcc", "new_habit", "reward prediction error",
             0.45, 0.45,
             "Phasic dopamine reinforces whatever preceded a "
             "better-than-expected outcome.", curve=0.14),
    # --- rumination loop (undesirable) ---------------------------------
    EdgeSpec("BLA", "dACC", "rumination", "salience -> 'something is wrong'",
             0.62, 0.55,
             "Affective salience biases monitoring toward threat "
             "interpretations.", curve=0.16),
    EdgeSpec("dACC", "BLA", "rumination", "re-appraise as threat", 0.60, 0.52,
             "Repeatedly running a negative interpretation makes it more "
             "accessible next time.", curve=0.16),
    EdgeSpec("HPC", "BLA", "rumination", "past bleeds into present",
             0.58, 0.50,
             "Pattern completion can make a new situation retrieve an old "
             "emotional memory; overgeneralisation is a hallmark of anxiety "
             "and PTSD.", curve=-0.16),
    # --- rapid threat output -------------------------------------------
    EdgeSpec("BLA", "CeA", "threat", "threat signal", 0.70, 0.62,
             "BLA->CeA is the core route from learned threat value to "
             "defensive output.", curve=0.10),
    EdgeSpec("CeA", "LC", "threat", "arousal / autonomic", 0.66, 0.60,
             "Central amygdala drives noradrenergic arousal.", curve=0.12),
    EdgeSpec("LC", "dlPFC", "threat", "arousal -> prefrontal (inverted U)",
             0.50, 0.50,
             "Moderate noradrenergic tone helps prefrontal function; high "
             "tone impairs it.", curve=-0.30),
    # --- stress-driven autopilot ---------------------------------------
    EdgeSpec("BLA", "DLS", "stress_habit", "stress -> autopilot", 0.55, 0.45,
             "Stress biases the system from goal-directed toward habitual "
             "control.", curve=0.14),
    EdgeSpec("CeA", "DLS", "stress_habit", "react without deciding",
             0.48, 0.40,
             "Defensive states narrow the behavioural repertoire.",
             curve=0.18),
]


def target_weight(edge: EdgeSpec) -> float:
    """The TARGET brain's strength for this connection.

    Not a prediction about a real brain: it is the direction of practice the
    user has chosen. Undesirable pathways are pushed toward a *non-zero*
    target because the threat system is supposed to exist.
    """
    return PATHWAYS[edge.pathway].target


def edge_key(e: EdgeSpec) -> Tuple[str, str]:
    return (e.src, e.dst)


# Groups -> nice display colours for the region legend
GROUP_COLORS = {
    "PFC": "#4cc9f0",
    "Amygdala": "#ff6b6b",
    "Hippocampus": "#80ffdb",
    "BasalGanglia": "#b8f2a6",
    "Neuromodulator": "#fca311",
}

GROUP_NOTES = {
    "PFC": "Cognitive control, planning, reappraisal, inhibition, goals.",
    "Amygdala": "Emotional salience and rapid threat responses.",
    "Hippocampus": "Context and memory; 'is this now, or is this then?'",
    "BasalGanglia": "Habit formation and automatic behaviour.",
    "Neuromodulator": "Learning-rate and arousal modulators (not regions "
                      "you 'train' directly).",
}

# Explicitly excluded, and why - shown in the app's Honesty panel.
EXCLUSIONS = [
    ("Insula", "Excluded by design at the user's request. In the literature "
               "it is central to interoception and to how bodily states are "
               "read as emotion."),
    ("Default mode network", "Highly relevant to rumination, but it is a "
                             "distributed network rather than a region; "
                             "folded into the rumination pathway instead."),
    ("Hypothalamus / HPA axis", "Cortisol dynamics are represented only as a "
                                "single abstract 'stress' scalar."),
    ("Serotonergic raphe, cerebellum, sensory cortices",
     "Omitted for clarity, not because they are unimportant."),
]
