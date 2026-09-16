"""
neuroforge.anatomy.structures
=============================

The anatomical inventory: which structures exist, where they sit, how their
geometry is built, and what the neuroscience actually says about each.

------------------------------------------------------------------------------
COORDINATES
------------------------------------------------------------------------------
Positions are given in millimetres in an approximately MNI-like stereotaxic
frame (+x right, +y anterior, +z superior), with the origin near the centre
of the cerebrum rather than at the anterior commissure. They are placed from
standard anatomical references and are accurate to roughly a centimetre -
good enough to teach *where things are relative to each other*, not good
enough to plan anything.

------------------------------------------------------------------------------
THE CENTRAL SCIENTIFIC POINT
------------------------------------------------------------------------------
None of these structures is a module that "does" an emotion or a decision.
They are nodes in overlapping, distributed networks. Every one of them
participates in many functions, and every function here recruits many of
them. "PFC = good, amygdala = bad" is wrong and the app says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .mesh import Mesh, build_blob, build_tube, mirror_x

Vec3 = Tuple[float, float, float]


@dataclass
class Structure:
    id: str
    name: str
    short: str                    # label shown on the 3-D pin
    system: str                   # grouping for the UI
    color: str                    # base colour
    bilateral: bool
    builder: Callable[[], Mesh]
    centroid: Vec3                # right-side / midline centroid, mm
    # ---- knowledge -------------------------------------------------------
    what_it_is: str
    where_it_is: str
    contributes_to: str
    in_the_model: str
    caveat: str
    opacity: float = 1.0
    order: int = 50
    # Key into subcortex_real.ASEG. When set, and when the segmentation has
    # been downloaded, the shape is extracted from real data and `builder`
    # is used only as an offline fallback. Such meshes are already bilateral
    # and must not be mirrored again.
    aseg: str = ""


# ==========================================================================
#  Geometry builders
# ==========================================================================
# Each returns the RIGHT-hemisphere (or midline) mesh; bilateral structures
# are mirrored automatically at load time.

def _none() -> Mesh:
    """
    No procedural fallback.

    Used by structures that exist only when the real segmentation has been
    downloaded. Returning an empty mesh is deliberate: inventing a plausible
    ventricle would be exactly the kind of confident fiction this app is
    trying not to produce. The builder skips empty meshes entirely.
    """
    return Mesh(positions=[], normals=[], indices=[])


def _amygdala() -> Mesh:
    # Almond-shaped, hence the name (Greek: amygdale). Sits in the
    # anterior medial temporal lobe, just rostral to the hippocampal head.
    return build_blob((23, -2, -19), (9.5, 8.0, 6.5), subdiv=3,
                      lumpiness=0.10, seed=11)


def _hippocampus() -> Mesh:
    # An elongated, curved structure arcing back and up through the medial
    # temporal lobe: head (anterior, thickest) -> body -> tail (thin).
    path = [(22, -6, -24), (26, -14, -23), (29, -24, -18),
            (28, -34, -10), (23, -40, -1), (16, -42, 4)]
    return build_tube(path, lambda t: 5.6 - 2.9 * t, segments=12, rings=30)


def _caudate() -> Mesh:
    # A long C-shaped nucleus that follows the lateral ventricle: a bulky
    # head anteriorly, tapering through the body to a thin tail that curves
    # down and forward into the temporal lobe, ending near the amygdala.
    path = [(13, 20, 2), (15, 14, 12), (17, 2, 20), (20, -14, 20),
            (26, -30, 10), (30, -32, -6), (27, -24, -16)]
    return build_tube(path, lambda t: 6.4 - 4.6 * t, segments=12, rings=32)


def _putamen() -> Mesh:
    # Lens-shaped, lateral to the globus pallidus. Together with the caudate
    # it forms the dorsal striatum - the input stage of the basal ganglia.
    return build_blob((27, 2, 0), (6.5, 17.0, 13.0), subdiv=3,
                      lumpiness=0.06, seed=23)


def _pallidum() -> Mesh:
    return build_blob((18, 0, -1), (4.5, 10.0, 8.5), subdiv=3, seed=31)


def _accumbens() -> Mesh:
    # Ventral striatum, where caudate and putamen meet inferiorly.
    return build_blob((10, 13, -9), (5.5, 6.0, 4.5), subdiv=3, seed=41)


def _thalamus() -> Mesh:
    # The great relay: almost every cortical area talks to it.
    return build_blob((11, -18, 7), (8.0, 13.0, 8.5), subdiv=3,
                      lumpiness=0.07, seed=53)


def _vta() -> Mesh:
    # Midbrain dopaminergic origin, near the midline.
    return build_blob((5, -20, -14), (4.0, 4.5, 3.5), subdiv=2, seed=61)


def _cingulate() -> Mesh:
    # The cingulate gyrus is a flattened cortical sheet arching over the
    # corpus callosum on the MEDIAL wall of each hemisphere. `aspect` makes
    # the cross-section a flattened ellipse so it reads as a sheet.
    path = [(6, 26, -14), (6, 38, -4), (6, 40, 10), (6, 30, 24),
            (6, 14, 32), (6, -6, 36)]
    return build_tube(path, lambda t: 8.5 - 1.5 * t, segments=14, rings=28,
                      aspect=0.34)


def _sgacc() -> Mesh:
    # Subgenual ACC (Brodmann 25), below the genu of the corpus callosum.
    return build_blob((5, 30, -14), (4.0, 7.0, 6.0), subdiv=3, seed=71)


def _corpus_callosum() -> Mesh:
    # ~200 million axons connecting the hemispheres. Included for spatial
    # context: it is the landmark the cingulate arches over.
    path = [(0, 34, 6), (0, 38, 14), (0, 26, 24), (0, 2, 27),
            (0, -22, 22), (0, -34, 8), (0, -30, 0)]
    return build_tube(path, lambda t: 5.0 + 2.0 * (t * (1 - t)) * 4,
                      segments=14, rings=30, aspect=2.4)


def _brainstem() -> Mesh:
    path = [(0, -18, -14), (0, -22, -30), (0, -24, -46), (0, -22, -62)]
    return build_tube(path, lambda t: 11.0 - 3.0 * t, segments=16, rings=18)


def _cerebellum() -> Mesh:
    return build_blob((0, -62, -42), (46.0, 30.0, 22.0), subdiv=4,
                      lumpiness=0.16, seed=83)


# ==========================================================================
#  The inventory
# ==========================================================================

STRUCTURES: List[Structure] = [

    # ------------------------------------------------------- amygdala
    Structure(
        "amygdala", "Amygdala", "Amygdala", "Limbic", "#ff5c6e", True,
        _amygdala, (23, -2, -19), order=10, aseg="amygdala",
        what_it_is=(
            "A cluster of about a dozen distinct nuclei, not a single "
            "organ. The two that matter most here are the basolateral "
            "complex (BLA), which learns what is emotionally significant, "
            "and the central nucleus (CeA), which drives the response."),
        where_it_is=(
            "Deep in the anterior medial temporal lobe, immediately in "
            "front of and slightly above the head of the hippocampus, "
            "about 2-3 cm in from the temple."),
        contributes_to=(
            "Assigning emotional and motivational significance to events, "
            "and rapidly mobilising a response. It is heavily involved in "
            "threat learning, but it also responds to rewards, to novelty, "
            "and to socially relevant cues such as faces. It biases "
            "attention and memory toward whatever it has flagged."),
        in_the_model=(
            "The origin of the 'salience' signal. When you log a trigger, "
            "this is the structure that lights first. Regulation pathways "
            "in the simulation target it; they do not switch it off."),
        caveat=(
            "The amygdala is NOT 'the fear centre' and it is not the "
            "villain. People with amygdala damage are not enlightened; "
            "they have impaired threat detection and social judgement. The "
            "target state in this app deliberately keeps threat "
            "responding at a moderate value rather than at zero."),
    ),

    # ---------------------------------------------------- hippocampus
    Structure(
        "hippocampus", "Hippocampus", "Hippocampus", "Limbic", "#4ade80",
        True, _hippocampus, (26, -24, -16), order=11, aseg="hippocampus",
        what_it_is=(
            "A curved, layered cortical structure - the name is Greek for "
            "seahorse. Contains subfields (dentate gyrus, CA3, CA1, "
            "subiculum) with quite different computational jobs."),
        where_it_is=(
            "Medial temporal lobe, curving backwards and upwards from just "
            "behind the amygdala, wrapping around the brainstem."),
        contributes_to=(
            "Binding events into episodes: what happened, where, when. It "
            "supports 'pattern separation' (keeping similar experiences "
            "distinct) and 'pattern completion' (retrieving a whole memory "
            "from a fragment). Crucially for this app, it supplies CONTEXT: "
            "it is a major reason the same cue can feel dangerous in one "
            "setting and harmless in another."),
        in_the_model=(
            "Drives the 'contextual safety' pathway - the simulated "
            "capacity to register 'this is happening now' rather than "
            "responding to a rerun of something older."),
        caveat=(
            "Overgeneralisation - the past bleeding into the present - is "
            "modelled here as a hippocampal contribution to the rumination "
            "loop. That is a simplification of an active research area, "
            "not a settled fact."),
    ),

    # ------------------------------------------------------ cingulate
    Structure(
        "acc", "Anterior cingulate cortex", "ACC", "Prefrontal", "#fbbf24",
        True, _cingulate, (6, 24, 12), order=5, opacity=0.95,
        what_it_is=(
            "A cortical sheet on the medial wall of each hemisphere, "
            "arching over the corpus callosum. It is functionally "
            "subdivided: a dorsal/mid part more tied to cognition and "
            "action, a rostral/ventral part more tied to affect."),
        where_it_is=(
            "On the inner (medial) surface of each hemisphere, wrapped "
            "directly around the corpus callosum. You cannot see it from "
            "the outside - hide a hemisphere to view it."),
        contributes_to=(
            "Monitoring. It responds to conflict, errors, surprise, pain "
            "and effort cost, and it is strongly implicated in signalling "
            "that more control is needed - which helps recruit lateral "
            "prefrontal cortex. It sits at the junction between "
            "'something is wrong' and 'do something about it'."),
        in_the_model=(
            "The bridge. It receives the salience signal and opens two "
            "competing routes: the old habitual response and the "
            "deliberate regulated one. Without it, control has nothing to "
            "trigger it."),
        caveat=(
            "There is genuine, ongoing scientific disagreement about what "
            "the ACC computes - conflict, expected value of control, "
            "surprise, and effort accounts all have serious support. The "
            "app picks one framing for teaching purposes."),
    ),

    Structure(
        "sgacc", "Subgenual ACC (BA25)", "sgACC", "Prefrontal", "#f59e0b",
        True, _sgacc, (5, 30, -14), order=6,
        what_it_is="A small ventral sector of cingulate cortex below the "
                   "genu (the front bend) of the corpus callosum.",
        where_it_is="Medial wall, low and anterior, tucked under the front "
                    "of the corpus callosum.",
        contributes_to="Autonomic and affective regulation; it is densely "
                       "connected with the amygdala, hypothalamus and "
                       "brainstem. It has attracted attention in mood "
                       "research and as a target in deep brain stimulation "
                       "studies for treatment-resistant depression.",
        in_the_model="Grouped with the ventromedial regulation pathway.",
        caveat="Findings here come largely from small clinical studies. "
               "Treat any strong claim about 'the depression area' with "
               "scepticism.",
    ),

    # ----------------------------------------------------- basal ganglia
    Structure(
        "caudate", "Caudate nucleus", "Caudate", "Basal ganglia", "#60a5fa",
        True, _caudate, (17, 2, 14), order=20, aseg="caudate",
        what_it_is="A long C-shaped nucleus following the curve of the "
                   "lateral ventricle. With the putamen it forms the "
                   "dorsal striatum.",
        where_it_is="Deep, arching from just behind the frontal lobe "
                    "backwards and then down into the temporal lobe.",
        contributes_to=(
            "Goal-directed action: selecting what to do based on what the "
            "outcome is currently worth. Its associative territory is "
            "densely interconnected with prefrontal cortex."),
        in_the_model=(
            "Carries the 'deliberate, effortful action' pathway - the one "
            "you use when a regulated response is still new and hard."),
        caveat="The rodent dorsomedial/dorsolateral striatal distinction "
               "maps only approximately onto human caudate/putamen.",
    ),

    Structure(
        "putamen", "Putamen", "Putamen", "Basal ganglia", "#38bdf8", True,
        _putamen, (27, 2, 0), order=21, aseg="putamen",
        what_it_is="A lens-shaped nucleus lateral to the globus pallidus; "
                   "the sensorimotor input zone of the striatum.",
        where_it_is="Deep and lateral, behind the insula, at roughly the "
                    "level of the ear.",
        contributes_to=(
            "Habitual, stimulus-driven action. As a behaviour is repeated "
            "over a long period, control shifts from associative "
            "(caudate-like) toward sensorimotor (putamen-like) territory, "
            "and the behaviour becomes more automatic and less sensitive "
            "to whether the outcome is still worth having."),
        in_the_model=(
            "The destination of practice. Repetition in the simulation "
            "gradually moves strength from the deliberate pathway into "
            "this automatic one. That transfer is what 'the new response "
            "becomes easier to access' means here."),
        caveat="This shift takes weeks to months of consistent repetition "
               "in humans, with enormous individual variation. The popular "
               "'21 days to a habit' figure has no good evidence behind it.",
    ),

    Structure(
        "pallidum", "Globus pallidus", "Pallidum", "Basal ganglia",
        "#818cf8", True, _pallidum, (18, 0, -1), order=22, aseg="pallidum",
        what_it_is="The main output stage of the basal ganglia.",
        where_it_is="Medial to the putamen, lateral to the internal capsule.",
        contributes_to="Gating: it tonically inhibits the thalamus, and "
                       "selecting an action means releasing that brake on "
                       "one option while holding it on others.",
        in_the_model="Represented as part of the action-output gate.",
        caveat="",
    ),

    Structure(
        "accumbens", "Nucleus accumbens", "NAcc", "Basal ganglia", "#f0abfc",
        True, _accumbens, (10, 13, -9), order=23, aseg="accumbens",
        what_it_is="The ventral striatum, where the caudate and putamen "
                   "merge below.",
        where_it_is="Deep, low and anterior, near the midline.",
        contributes_to=(
            "Reinforcement learning. Dopaminergic input signals a reward "
            "PREDICTION ERROR - the difference between what happened and "
            "what was expected - and this reinforces whatever preceded it. "
            "Relief counts: escaping something unpleasant is rewarding."),
        in_the_model=(
            "Why avoidance is sticky. In the simulation, avoidance carries "
            "a reward bonus, because the immediate relief genuinely "
            "reinforces the avoiding - which is the central maintaining "
            "mechanism in anxiety."),
        caveat="Reducing this to 'the pleasure centre' is badly wrong. It "
               "tracks prediction error and motivation, not enjoyment.",
    ),

    # ---------------------------------------------------------- others
    Structure(
        "thalamus", "Thalamus", "Thalamus", "Subcortical", "#c4b5fd", True,
        _thalamus, (11, -18, 7), order=30, aseg="thalamus",
        what_it_is="A pair of egg-shaped nuclear complexes at the centre "
                   "of the brain.",
        where_it_is="Dead centre, on either side of the third ventricle.",
        contributes_to="Relaying and gating. Almost all sensory input and "
                       "most cortico-cortical traffic passes through it, "
                       "and it closes the cortex-basal ganglia loop.",
        in_the_model="The output gate: the point where a selected action "
                     "is actually released.",
        caveat="",
    ),

    Structure(
        "vta", "Ventral tegmental area", "VTA", "Neuromodulatory",
        "#fb923c", True, _vta, (5, -20, -14), order=31,
        what_it_is="A small midbrain nucleus containing dopaminergic "
                   "neurons.",
        where_it_is="Midbrain, near the midline, below the thalamus.",
        contributes_to="Broadcasting reward prediction error to the "
                       "striatum and prefrontal cortex, gating plasticity "
                       "in its targets.",
        in_the_model="A global multiplier on the simulation's learning "
                     "rate, not a structure you practise directly.",
        caveat="Dopamine is not 'the pleasure chemical'. It is much closer "
               "to a teaching and motivation signal.",
    ),

    Structure(
        "callosum", "Corpus callosum", "CC", "White matter", "#e2e8f0",
        False, _corpus_callosum, (0, 2, 20), order=40, aseg="corpus_callosum", opacity=0.55,
        what_it_is="The largest white-matter tract in the brain: roughly "
                   "200 million axons connecting the hemispheres.",
        where_it_is="Midline, arching across the centre of the brain.",
        contributes_to="Interhemispheric communication.",
        in_the_model="Spatial context only - it is the landmark the "
                     "cingulate arches over.",
        caveat="Included so the cingulate's position makes sense.",
    ),

    Structure(
        "brainstem", "Brainstem", "Brainstem", "Subcortical", "#94a3b8",
        False, _brainstem, (0, -22, -38), order=41, aseg="brainstem", opacity=0.85,
        what_it_is="Midbrain, pons and medulla.",
        where_it_is="Descending from the centre of the brain to the spinal "
                    "cord.",
        contributes_to="Arousal, autonomic control, and the actual "
                       "execution of many defensive responses - the "
                       "changes in heart rate and breathing you feel.",
        in_the_model="The physiological endpoint of the threat pathway.",
        caveat="",
    ),

    Structure(
        "cerebellum", "Cerebellum", "Cerebellum", "Subcortical", "#7dd3fc",
        False, _cerebellum, (0, -62, -42), order=42, aseg="cerebellum", opacity=0.75,
        what_it_is="A densely folded structure containing more neurons "
                   "than the rest of the brain combined.",
        where_it_is="Beneath the occipital lobes, behind the brainstem.",
        contributes_to="Prediction and calibration - classically of "
                       "movement, but increasingly recognised in cognition "
                       "and emotion too.",
        in_the_model="Not modelled. Shown for anatomical completeness, and "
                     "as a reminder of how much this simulation leaves out.",
        caveat="Its role in emotion is real but outside this model's scope.",
    ),

    # ---- structures that only exist once real segmentation is available ----
    # These have no procedural fallback: rather than invent a shape for a
    # ventricle, the app simply does not show one until the atlas is present.
    Structure(
        "ventricles", "Ventricles", "Ventricles", "Subcortical", "#38bdf8",
        False, _none, (0, -20, 8), order=60, aseg="ventricles", opacity=0.5,
        what_it_is="Four connected cavities filled with cerebrospinal fluid: "
                   "two lateral ventricles, the third and the fourth. They "
                   "are spaces, not tissue.",
        where_it_is="Deep and central. The lateral ventricles arc through "
                    "each hemisphere; the third sits on the midline between "
                    "the thalami; the fourth lies behind the brainstem.",
        contributes_to="Cushioning, waste clearance and chemical transport. "
                       "Their shape is also clinically important - they "
                       "enlarge when surrounding tissue is lost.",
        in_the_model="Not modelled. Included because they are a large part "
                     "of what is actually inside a head, and because they "
                     "make the geometry of everything around them legible.",
        caveat="Ventricle size varies enormously between healthy people. "
               "This is one template brain, so read nothing into the size.",
    ),

    Structure(
        "ventraldc", "Ventral diencephalon", "Ventral DC", "Subcortical",
        "#fb923c", True, _none, (11, -16, -12), order=35, aseg="ventral_dc",
        what_it_is="Not one structure but a segmentation region. It contains "
                   "the hypothalamus, subthalamic nucleus, substantia nigra, "
                   "red nucleus and nearby white matter, which the automatic "
                   "segmentation cannot reliably separate at this resolution.",
        where_it_is="Below the thalamus, above the brainstem, wrapped around "
                    "the top of the midbrain.",
        contributes_to="An enormous amount for its size: autonomic control, "
                       "hormone release, arousal, appetite, dopamine "
                       "production and movement initiation all have "
                       "machinery in here.",
        in_the_model="Not modelled directly, though the hypothalamic stress "
                     "axis sits inside this territory and is referenced by "
                     "the threat pathway.",
        caveat="It is shown as one lump because that is genuinely how the "
               "atlas labels it. Do not read this as a claim that the "
               "hypothalamus and substantia nigra are one structure - they "
               "are distinct, and separating them needs higher-resolution "
               "imaging than this template provides.",
    ),
]

STRUCTURES_BY_ID: Dict[str, Structure] = {s.id: s for s in STRUCTURES}


# ==========================================================================
#  Cortical regions (surface parcels, not separate meshes)
# ==========================================================================
#
# APPROXIMATION: these are highlighted by per-vertex label on the cortical
# surface, assigned by geometric rule in mesh.classify_cortex(). They are
# NOT derived from a probabilistic cytoarchitectonic atlas.

@dataclass
class CorticalRegion:
    id: str
    label_index: int
    name: str
    short: str
    color: str
    what_it_is: str
    where_it_is: str
    contributes_to: str
    in_the_model: str
    caveat: str
    # True for cortex that no external view can show, because it is folded
    # inside a sulcus. The viewer switches to x-ray when one of these is
    # selected - otherwise the region highlights perfectly correctly and the
    # user sees nothing at all, which is indistinguishable from a broken app.
    buried: bool = False


CORTICAL_REGIONS: List[CorticalRegion] = [
    CorticalRegion(
        "dlPFC", 1, "Dorsolateral prefrontal cortex", "dlPFC", "#22d3ee",
        what_it_is="The upper outer surface of the frontal lobe, roughly "
                   "Brodmann areas 9 and 46.",
        where_it_is="Front of the brain, high and to the side - behind and "
                    "above the outer end of the eyebrow.",
        contributes_to=(
            "Holding a goal active in mind while something else competes "
            "for your attention, and biasing attention and action toward "
            "that goal. It is a top contributor to working memory and "
            "cognitive control. Its performance follows an inverted-U with "
            "arousal: too little or too much both degrade it, which is a "
            "large part of why regulation gets hard when you are highly "
            "stressed or exhausted."),
        in_the_model="Source of the 'goal-directed control' pathway - the "
                     "effortful route, and the one that costs the most.",
        caveat="It is not a 'willpower muscle' and it does not have a fuel "
               "gauge. The ego-depletion literature has largely failed to "
               "replicate.",
    ),
    CorticalRegion(
        "vlPFC", 2, "Ventrolateral prefrontal cortex", "vlPFC", "#0ea5e9",
        what_it_is="The lower outer frontal surface, including the "
                   "inferior frontal gyrus (roughly BA 44/45/47).",
        where_it_is="Front of the brain, lower and to the side - roughly "
                    "above and in front of the temple.",
        contributes_to=(
            "Stopping and selecting. The right inferior frontal gyrus is "
            "one of the most reliable findings in the response-inhibition "
            "literature. This region is also consistently engaged during "
            "AFFECT LABELLING - putting a feeling into words - where its "
            "activity is inversely related to amygdala response."),
        in_the_model="Carries the 'top-down regulation' pathway. Logging "
                     "'named the emotion' or 'paused' strengthens it.",
        caveat="The inverse relationship with amygdala activity is a "
               "correlation observed across people and trials. It does not "
               "establish that this region flips an off-switch, and the "
               "influence is largely indirect via other relays.",
    ),
    CorticalRegion(
        "vmPFC", 3, "Ventromedial prefrontal cortex", "vmPFC", "#3b82f6",
        what_it_is="The lower inner surface of the frontal lobe.",
        where_it_is="Front of the brain, low and on the midline - visible "
                    "only if you hide a hemisphere.",
        contributes_to=(
            "Signalling safety and computing value. It is central to "
            "EXTINCTION RECALL: remembering, tomorrow, that the thing you "
            "faced today turned out to be safe. It works together with the "
            "hippocampus, which makes that safety context-specific."),
        in_the_model="The 'safety' half of the regulation pathway, and the "
                     "structure that exposure practice strengthens most.",
        caveat="Extinction does not erase the original threat memory. It "
               "builds a competing one. This is why fear can return in a "
               "new context, after a long gap, or after a bad experience - "
               "and the simulation reproduces that.",
    ),
    CorticalRegion(
        "mPFC", 26, "Medial prefrontal cortex", "mPFC", "#60a5fa",
        what_it_is="The inner surface of the frontal lobe above the "
                   "ventromedial sector: the medial wall of the superior "
                   "frontal gyrus, anterior to the supplementary motor area.",
        where_it_is="On the midline at the front, facing the other "
                    "hemisphere. Hide a hemisphere to see it.",
        contributes_to="Thinking about mental states - your own and other "
                       "people's. It is the anterior hub of the default mode "
                       "network, and it is unusually active when you "
                       "consider whether a trait describes you, or imagine "
                       "what someone else is thinking.",
        in_the_model="Not modelled as a separate node. The regulation "
                     "pathways in this simulation run through dlPFC and "
                     "vmPFC, which are its neighbours.",
        caveat="Self-referential processing is not the same thing as "
               "self-knowledge, and activity here does not indicate "
               "insight. Rumination engages this region too.",
    ),
    CorticalRegion(
        "dACC", 5, "Dorsal anterior cingulate cortex", "dACC", "#fbbf24",
        what_it_is="The upper part of the anterior cingulate, arching over "
                   "the corpus callosum. The cortical surface of the same "
                   "territory shown as the ACC structure.",
        where_it_is="Medial wall, wrapped around and above the front of the "
                    "corpus callosum.",
        contributes_to="Monitoring: conflict, errors, surprise, pain and "
                       "effort cost. It sits at the junction between "
                       "'something is wrong' and 'do something about it', "
                       "and helps recruit lateral prefrontal control.",
        in_the_model="The bridge. It receives the salience signal and opens "
                     "the competing habitual and regulated routes.",
        caveat="What the dACC computes is genuinely unsettled - conflict, "
               "expected value of control, surprise and effort accounts all "
               "have serious support. This app picks one framing to teach "
               "with, which is not the same as it being correct.",
    ),
    CorticalRegion(
        "sgACC", 6, "Subgenual anterior cingulate", "sgACC", "#f59e0b",
        what_it_is="A small sector of cingulate cortex below the genu - the "
                   "front bend - of the corpus callosum. Roughly Brodmann "
                   "area 25.",
        where_it_is="Medial wall, low and anterior, tucked underneath the "
                    "front of the corpus callosum.",
        contributes_to="Autonomic and affective regulation. It is densely "
                       "connected with the amygdala, hypothalamus and "
                       "brainstem, and is one route by which an emotional "
                       "appraisal becomes a change in heart rate.",
        in_the_model="Part of the threat pathway, carrying the visceral "
                     "component of the simulated response.",
        caveat="Desikan-Killiany has no subgenual parcel. This region is "
               "made by splitting rostral anterior cingulate at z = 0, "
               "which is a geometric proxy for a boundary the atlas does "
               "not draw. Its prominence in depression research also "
               "attracts stronger claims than the evidence supports.",
    ),
    CorticalRegion(
        "aINS", 24, "Anterior insula", "Ant. insula", "#fb7185",
        what_it_is="The front portion of the insula, a lobe of cortex "
                   "folded away inside the lateral sulcus. It is not "
                   "visible from any outside view of the brain.",
        where_it_is="Buried deep in the lateral sulcus, between the frontal "
                    "and temporal lobes. Use x-ray or hide a hemisphere.",
        contributes_to="Interoception - representing the physiological "
                       "state of the body - and the point at which that "
                       "state becomes a felt emotion. It responds to "
                       "hunger, heartbeat, breathlessness, disgust, unfair "
                       "treatment and uncertainty, which is a strange list "
                       "until you notice they all involve noticing how "
                       "you feel.",
        in_the_model="Not modelled as a node. Its function is nevertheless "
                     "what most of the logged practices in this app are "
                     "training: noticing a bodily state before it becomes "
                     "an action.",
        caveat="The anterior insula is one of the most reliably activated "
               "regions in all of neuroimaging, which makes it nearly "
               "useless as evidence for any specific claim. If a region "
               "lights up for almost everything, its lighting up tells you "
               "almost nothing.",
        buried=True,
    ),
    CorticalRegion(
        "pINS", 25, "Posterior insula", "Post. insula", "#9f1239",
        what_it_is="The rear portion of the insula, separated from the "
                   "anterior part by the central insular sulcus.",
        where_it_is="Deep in the lateral sulcus, behind the anterior "
                    "insula.",
        contributes_to="Primary interoceptive and sensory representation: "
                       "the raw afferent signal from the body - "
                       "temperature, pain, itch, visceral state - before it "
                       "has been integrated into a feeling.",
        in_the_model="Not modelled.",
        caveat="The posterior-to-anterior progression from raw signal to "
               "felt emotion is an influential model, not an established "
               "fact, and the split drawn here is a flat approximation to "
               "a sulcus that runs obliquely.",
        buried=True,
    ),
    CorticalRegion(
        "OFC", 4, "Orbitofrontal cortex", "OFC", "#6366f1",
        what_it_is="The frontal cortex resting directly above the eye "
                   "sockets.",
        where_it_is="The underside of the frontal lobes.",
        contributes_to="Representing the current, specific value of "
                       "outcomes and updating it when things change - "
                       "which is what lets behaviour stay goal-directed "
                       "instead of drifting into habit.",
        in_the_model="Grouped with goal-directed control.",
        caveat="",
    ),
    CorticalRegion(
        "motor", 7, "Primary motor cortex", "Motor", "#a3a3a3",
        what_it_is="The precentral gyrus: the cortex whose output descends "
                   "to the spinal cord to drive movement directly. Like the "
                   "somatosensory strip behind it, it carries a body map.",
        where_it_is="Across the top of the brain, immediately in front of "
                    "the central sulcus - roughly under a headphone band.",
        contributes_to="Executing voluntary movement. Planning and "
                       "selecting which movement to make happens mostly in "
                       "the premotor cortex just in front of it, which is "
                       "shown here as a separate region.",
        in_the_model="Not modelled - shown for orientation.",
        caveat="",
    ),
    # There is no generic "parietal cortex" entry any more. Every Desikan
    # parcel of the parietal lobe now belongs to a finer label -
    # somatosensory, superior parietal, supramarginal, TPJ or precuneus - so
    # a catch-all region would open a panel that highlighted nothing.
    CorticalRegion(
        "temporal", 9, "Temporal cortex", "Temporal", "#a8a29e",
        what_it_is="The lobe below the Sylvian fissure.",
        where_it_is="The sides of the brain, by the ears.",
        contributes_to="Hearing, language, object and face recognition, "
                       "and - on its medial surface - memory.",
        in_the_model="Not modelled directly; it houses the amygdala and "
                     "hippocampus, which are.",
        caveat="",
    ),
    CorticalRegion(
        "occipital", 10, "Occipital cortex", "Occipital", "#78716c",
        what_it_is="The lobe at the very back.",
        where_it_is="Rear of the brain.",
        contributes_to="Vision.",
        in_the_model="Not modelled - shown for orientation.",
        caveat="",
    ),
    CorticalRegion(
        "PCC", 11, "Posterior cingulate cortex", "PCC", "#f472b6",
        what_it_is="The rear half of the cingulate belt, wrapped around the "
                   "back of the corpus callosum, together with the "
                   "retrosplenial cortex below it.",
        where_it_is="Medial wall, behind the mid-cingulate, above and "
                    "behind the splenium.",
        contributes_to="Along with medial prefrontal cortex it forms the "
                       "midline core of the default mode network. Activity "
                       "here rises during self-referential thought, "
                       "autobiographical memory and mind-wandering, and "
                       "falls when attention is captured by a demanding "
                       "external task. It is among the most consistently "
                       "reported correlates of experienced meditators, "
                       "typically as reduced activity or altered "
                       "connectivity during self-focused thought.",
        in_the_model="Not a node in the simulation. It is shown because "
                     "self-referential processing is central to what you "
                     "are trying to understand, and it was previously "
                     "invisible in this app.",
        caveat="'The self lives in the PCC' would be badly wrong. It is a "
               "hub with unusually high metabolic cost and dense "
               "connectivity, which is a structural description, not a "
               "psychological one. Reduced default-mode activity is also "
               "seen in states that are not remotely enlightened, "
               "including some forms of anaesthesia and disorders of "
               "consciousness.",
    ),
    CorticalRegion(
        "precuneus", 12, "Precuneus", "Precuneus", "#c084fc",
        what_it_is="The medial parietal lobe, tucked between the two "
                   "hemispheres above the posterior cingulate.",
        where_it_is="Medial surface, behind the paracentral lobule and "
                    "above the parieto-occipital sulcus.",
        contributes_to="First-person perspective taking, mental imagery of "
                       "yourself, episodic memory retrieval and the sense "
                       "of being located somewhere. Frequently co-active "
                       "with the PCC during autobiographical recall.",
        in_the_model="Not modelled. Shown to make the posterior midline "
                     "legible.",
        caveat="It is one of the last cortical regions to be well "
               "characterised, partly because it is buried in the midline "
               "and hard to reach with electrodes. Confident functional "
               "claims about it should be treated cautiously.",
    ),
    CorticalRegion(
        "TPJ", 13, "Temporoparietal junction", "TPJ", "#22c55e",
        what_it_is="Where the temporal and parietal lobes meet, centred on "
                   "the angular gyrus.",
        where_it_is="Lateral surface, above and behind the end of the "
                    "Sylvian fissure.",
        contributes_to="Distinguishing self from other, taking someone "
                       "else's perspective, and reorienting attention when "
                       "something unexpected happens. Damage or "
                       "stimulation here can disrupt the sense of where "
                       "'you' are located, including out-of-body "
                       "experiences.",
        in_the_model="Not modelled, but directly relevant to the concept "
                     "of self you are exploring.",
        caveat="The right TPJ is often labelled 'the theory of mind area'. "
               "That is an oversimplification: the same territory responds "
               "during attentional reorienting that has nothing to do with "
               "other minds, and the debate over whether these are one "
               "function or two adjacent ones is unresolved.",
    ),
    CorticalRegion(
        "FPC", 14, "Frontopolar cortex", "Frontopolar", "#38bdf8",
        what_it_is="The very front of the frontal lobe, Brodmann area 10 - "
                   "the most anterior cortex in the brain.",
        where_it_is="The forward tip of each hemisphere, above the orbital "
                    "surface.",
        contributes_to="Holding one goal in reserve while pursuing "
                       "another, evaluating your own performance, and "
                       "metacognition - judging how well you actually know "
                       "something. Disproportionately enlarged in humans "
                       "compared with other primates.",
        in_the_model="Not modelled directly. Its function - noticing what "
                     "you are doing while you do it - is the capacity this "
                     "whole application is meant to support.",
        caveat="'The seat of human uniqueness' claims about BA10 outrun "
               "the evidence. It is relatively larger in humans, but "
               "relative size is not function, and comparative measurements "
                "across species are contested.",
    ),

    # ---- general anatomical coverage -----------------------------------
    # These are not part of the simulation. They are here so the app can
    # answer "what is that bit?" for the whole cortex rather than only for
    # the regions this particular model happens to use.
    CorticalRegion(
        "premotor", 15, "Premotor cortex", "Premotor", "#cbd5e1",
        what_it_is="A strip of frontal cortex immediately in front of the "
                   "primary motor cortex, including the supplementary motor "
                   "area on the medial wall.",
        where_it_is="Between the prefrontal cortex and the precentral gyrus.",
        contributes_to="Preparing and selecting movements rather than "
                       "executing them: sequencing actions, responding to "
                       "cues that specify which movement to make, and "
                       "withholding one that has been prepared.",
        in_the_model="Not modelled. It is shown separately from the "
                     "prefrontal cortex because it does a different job - "
                     "planning movement, not regulating emotion.",
        caveat="The boundary between premotor and prefrontal cortex is "
               "gradual. The atlas draws a line where cytoarchitecture "
               "changes, but function shades across it.",
    ),
    CorticalRegion(
        "somatosensory", 16, "Somatosensory cortex", "Somatosensory",
        "#e2e8f0",
        what_it_is="Primary somatosensory cortex, the postcentral gyrus. "
                   "It contains an orderly map of the body surface - the "
                   "sensory homunculus.",
        where_it_is="The gyrus immediately behind the central sulcus, "
                    "mirroring the motor strip in front of it.",
        contributes_to="Touch, pressure, vibration, limb position. It also "
                       "matters for emotion more than its name suggests: "
                       "much of what is felt as a feeling is a bodily state "
                       "represented here and in neighbouring cortex.",
        in_the_model="Not modelled, though the bodily component of emotion "
                     "it supports is the thing interoceptive practices are "
                     "usually training.",
        caveat="The homunculus is a real and reproducible map, but the "
               "familiar distorted-figure picture is a simplification - the "
               "map is fragmented and shifts with use.",
    ),
    CorticalRegion(
        "SPL", 17, "Superior parietal lobule", "Sup. parietal", "#a5b4fc",
        what_it_is="The upper part of the parietal lobe, behind the "
                   "somatosensory strip.",
        where_it_is="Top rear of each hemisphere, above the intraparietal "
                    "sulcus.",
        contributes_to="Spatial attention and keeping track of where things "
                       "are relative to the body; guiding reaching and "
                       "shifting the focus of attention.",
        in_the_model="Not modelled.",
        caveat="Attention is not one faculty located here. This region is "
               "one node of a distributed attention network.",
    ),
    CorticalRegion(
        "V1", 18, "Primary visual cortex", "V1", "#818cf8",
        what_it_is="The first cortical stage of vision, also called striate "
                   "cortex for the stripe of myelin visible in it.",
        where_it_is="Buried in and around the calcarine sulcus, on the "
                    "medial surface of the occipital lobe. Mostly hidden "
                    "from the outside - hide a hemisphere to see it.",
        contributes_to="Extracting edges, orientation, motion and binocular "
                       "disparity from the retinal image. It is mapped "
                       "retinotopically: neighbouring points in the visual "
                       "field are neighbouring points of cortex.",
        in_the_model="Not modelled.",
        caveat="V1 is not a passive camera. It receives more input from the "
               "rest of the brain than it does from the eyes, and what you "
               "see is shaped by expectation.",
    ),
    CorticalRegion(
        "A1", 19, "Primary auditory cortex", "A1", "#f0abfc",
        what_it_is="Heschl's gyrus, the first cortical stage of hearing.",
        where_it_is="On the upper surface of the temporal lobe, inside the "
                    "lateral sulcus. Almost entirely hidden from outside.",
        contributes_to="Frequency analysis, timing and the beginnings of "
                       "speech and music perception. Mapped tonotopically, "
                       "by pitch.",
        in_the_model="Not modelled.",
        caveat="Its small size is deceptive; a great deal of auditory "
               "processing happens in surrounding cortex, not here.",
    ),
    CorticalRegion(
        "MTL", 20, "Medial temporal cortex", "Medial temporal", "#34d399",
        what_it_is="Entorhinal and parahippocampal cortex - the cortical "
                   "gateway to the hippocampus.",
        where_it_is="The underside of the medial temporal lobe, wrapped "
                    "around the hippocampus.",
        contributes_to="Almost everything reaching the hippocampus passes "
                       "through here. Entorhinal cortex carries spatial and "
                       "temporal structure; parahippocampal cortex responds "
                       "strongly to scenes and context. This is a large part "
                       "of why a feeling can attach itself to a place.",
        in_the_model="Not modelled directly, but it is the route by which "
                     "the hippocampal 'contextual safety' signal in this "
                     "model would actually reach the rest of the brain.",
        caveat="Entorhinal cortex is among the first places affected in "
               "Alzheimer's disease. That is a genuine finding and not a "
               "reason to interpret anything about your own memory from a "
               "template brain.",
    ),
    CorticalRegion(
        "fusiform", 21, "Fusiform gyrus", "Fusiform", "#fcd34d",
        what_it_is="A long gyrus on the underside of the temporal lobe, "
                   "containing the region often called the fusiform face "
                   "area.",
        where_it_is="Ventral temporal surface, between the parahippocampal "
                    "and inferior temporal gyri.",
        contributes_to="Recognising complex visual forms, faces most "
                       "famously, but also words and other categories of "
                       "expert visual knowledge.",
        in_the_model="Not modelled.",
        caveat="Whether the fusiform face area is face-specific or a general "
               "expertise region is a long-running and still live argument. "
               "Calling it 'the face area' picks a side.",
    ),
    CorticalRegion(
        "STS", 22, "Superior temporal cortex", "Sup. temporal", "#5eead4",
        what_it_is="The superior temporal gyrus and the sulcus beneath it. "
                   "Its posterior left portion is the territory usually "
                   "called Wernicke's area.",
        where_it_is="The upper temporal lobe, running back from the "
                    "temporal pole below the lateral sulcus.",
        contributes_to="Understanding speech; also reading biological "
                       "motion, gaze direction, voices and intentions. "
                       "Together with the TPJ it is central to working out "
                       "what other people are doing and why.",
        in_the_model="Not modelled.",
        caveat="'Wernicke's area' has no agreed boundary and the classical "
               "model of it as the seat of comprehension has not survived "
               "modern imaging and lesion work intact.",
    ),
    CorticalRegion(
        "supramarginal", 23, "Supramarginal gyrus", "Supramarginal",
        "#86efac",
        what_it_is="The parietal gyrus curling around the end of the "
                   "lateral sulcus; with the angular gyrus it forms the "
                   "inferior parietal lobule.",
        where_it_is="Just above and behind the end of the lateral sulcus, "
                    "in front of the angular gyrus.",
        contributes_to="Phonological working memory, tool use and gesture, "
                       "and - in the right hemisphere - separating your own "
                       "emotional state from someone else's when judging "
                       "how they feel.",
        in_the_model="Not modelled.",
        caveat="The empathy findings here come largely from small "
               "stimulation studies. They are interesting rather than "
               "settled.",
    ),
]

CORTICAL_BY_ID: Dict[str, CorticalRegion] = {r.id: r for r in CORTICAL_REGIONS}
CORTICAL_BY_INDEX: Dict[int, CorticalRegion] = {
    r.label_index: r for r in CORTICAL_REGIONS}


# ==========================================================================
#  Groupings
# ==========================================================================
# Some of the most commonly used terms in this field are not parcels at all.
# "Prefrontal cortex" and "dorsal striatum" name collections of regions, and
# drawing either as a single coloured lump would be a category error - there
# is no boundary in the tissue where the dorsal striatum stops and something
# called the ventral striatum begins, only a convention about where to draw
# the line.
#
# So they are offered as selections over regions that already exist, rather
# than as regions in their own right. Selecting one highlights everything it
# contains.
GROUPS: List[Dict[str, Any]] = [
    {
        "id": "PFC",
        "name": "Prefrontal cortex",
        "short": "PFC",
        "color": "#38bdf8",
        "regions": ["dlPFC", "vlPFC", "vmPFC", "mPFC", "OFC", "FPC"],
        "structures": [],
        "what": (
            "All the frontal cortex in front of the motor and premotor "
            "strips. It is not one area: this selection covers six regions "
            "with different connections and different jobs."),
        "why": (
            "The umbrella term is useful because these regions share a "
            "broad role - holding goals in mind and using them to override "
            "what a situation would otherwise pull you into. It is the part "
            "of the brain that matures last, into the mid-twenties."),
        "caveat": (
            "'Strengthening your prefrontal cortex' is not a meaningful "
            "goal, because the PFC is not a muscle and not a unit. Training "
            "effects are usually specific to the task trained, and the "
            "evidence that they generalise is much weaker than the "
            "brain-training industry implies."),
    },
    {
        "id": "latPFC",
        "name": "Lateral prefrontal cortex",
        "short": "Lateral PFC",
        "color": "#22d3ee",
        "regions": ["dlPFC", "vlPFC"],
        "structures": [],
        "what": (
            "The outer surface of the prefrontal cortex: dorsolateral "
            "above, ventrolateral below."),
        "why": (
            "This is the cognitive-control territory. Holding a rule or a "
            "goal available, selecting among competing responses, and "
            "stopping one already under way. In this simulation it is the "
            "source of deliberate regulation."),
        "caveat": (
            "Lateral prefrontal activity indicates effortful control, not "
            "successful control. It also rises when regulation is being "
            "attempted and failing."),
    },
    {
        "id": "dorsal_striatum",
        "name": "Dorsal striatum",
        "short": "Dorsal striatum",
        "color": "#c084fc",
        "regions": [],
        "structures": ["caudate", "putamen"],
        "what": (
            "The caudate nucleus and putamen together. In humans the "
            "internal capsule separates them, which is why they look like "
            "two structures; in many other species they are one."),
        "why": (
            "This is the machinery of habit. As a behaviour is repeated, "
            "control shifts from goal-directed circuits towards this "
            "territory, and the behaviour becomes increasingly triggered by "
            "the situation rather than chosen. That shift is the single "
            "idea this whole application is built around."),
        "caveat": (
            "The dorsomedial/dorsolateral split that carries most of the "
            "goal-directed-to-habitual story comes largely from rodent "
            "work. The human mapping is inferred and contested, and this "
            "atlas cannot resolve it - what is shown here is the whole "
            "dorsal striatum, undivided."),
    },
    {
        "id": "ventral_striatum",
        "name": "Ventral striatum",
        "short": "Ventral striatum",
        "color": "#f472b6",
        "regions": [],
        "structures": ["accumbens"],
        "what": (
            "The nucleus accumbens and surrounding ventral territory, where "
            "the caudate and putamen merge underneath."),
        "why": (
            "Learning what is worth approaching. It receives dopamine from "
            "the ventral tegmental area and is central to how a reward "
            "prediction error updates future behaviour."),
        "caveat": (
            "Calling this 'the pleasure centre' is wrong. Most of its "
            "dopamine signal tracks anticipation and prediction error "
            "rather than enjoyment, and wanting and liking come apart."),
    },
]

GROUPS_BY_ID: Dict[str, Dict[str, Any]] = {g["id"]: g for g in GROUPS}


# ==========================================================================
#  A note the UI shows prominently
# ==========================================================================

NETWORK_NOTE = (
    "These structures are not modules that each 'do' one job. They are "
    "nodes in overlapping, distributed networks. The amygdala is not the "
    "fear centre; the prefrontal cortex is not the rational adult "
    "supervising it. Regulating an emotion is something a whole network "
    "does, and every structure shown here participates in many functions "
    "beyond the one this app highlights."
)

GEOMETRY_NOTE = (
    "This brain is procedurally generated, not scanned. Proportions, lobar "
    "arrangement, the major fissures and the positions of the subcortical "
    "structures follow standard anatomy at roughly centimetre accuracy. The "
    "pattern of gyri and sulci is synthetic - real cortical folding is "
    "individually unique, and the folds here do not correspond to named "
    "gyri. Cortical regions are assigned by geometric rule, not by a "
    "probabilistic atlas."
)


# ===========================================================================
#  Functional networks (Yeo 2011)
# ===========================================================================
# A network is not a place. These are sets of cortical territory whose
# activity rises and falls together, derived from resting-state fMRI in
# 1000 people. They cut across the anatomical regions above, which is
# exactly why they need their own layer.

NETWORK_INFO: Dict[str, Dict[str, str]] = {
    "Default mode": {
        "what": "Medial prefrontal cortex, posterior cingulate, precuneus, "
                "angular gyrus and lateral temporal cortex, active together "
                "when you are not engaged in a demanding external task.",
        "self": "This is the network most associated with thinking about "
                "yourself: remembering your past, imagining your future, "
                "rehearsing conversations, wondering what someone thinks of "
                "you. Mind-wandering recruits it heavily.",
        "caveat": "It is not 'the ego', not 'the monkey mind', and not "
                  "something to be switched off. It supports memory, "
                  "planning and social reasoning. Persistently elevated "
                  "self-referential activity is associated with rumination "
                  "in depression, but that is a pattern of use, not proof "
                  "that the network itself is the problem.",
    },
    "Frontoparietal control": {
        "what": "Lateral prefrontal and lateral parietal cortex, flexibly "
                "engaged when a task requires goals to be held and updated.",
        "self": "Where deliberate self-direction happens. It is the system "
                "that steps in when what you intended differs from what you "
                "are about to do.",
        "caveat": "Often called 'the executive network', which invites the "
                  "homunculus error - imagining a decision-maker inside the "
                  "head. It is a set of coordinating regions, not a chief.",
    },
    "Ventral attention / salience": {
        "what": "Anterior insula and mid-cingulate, responding to events "
                "that are unexpected or personally significant.",
        "self": "Proposed to arbitrate between the default mode and "
                "frontoparietal networks - roughly, deciding whether "
                "something deserves your attention.",
        "caveat": "The 'switching' account is an influential hypothesis, "
                  "not settled fact. Note also that the insula itself is "
                  "not drawn in this application.",
    },
    "Dorsal attention": {
        "what": "Intraparietal sulcus and frontal eye fields, engaged when "
                "attention is deliberately directed somewhere.",
        "self": "Typically anticorrelated with the default mode network: "
                "when focused outward, self-referential activity falls.",
        "caveat": "The anticorrelation is real but its magnitude depends "
                  "on preprocessing choices, and this was the subject of a "
                  "long methodological dispute.",
    },
    "Limbic": {
        "what": "Orbitofrontal and anterior temporal cortex.",
        "self": "Emotional value and its links to memory.",
        "caveat": "Poor signal quality in these areas makes this the least "
                  "reliably delineated of the seven networks.",
    },
    "Somatomotor": {
        "what": "Precentral and postcentral gyri.",
        "self": "Movement and body sensation.",
        "caveat": "",
    },
    "Visual": {
        "what": "Occipital cortex.",
        "self": "Sight.",
        "caveat": "",
    },
}

NETWORK_CAVEAT = (
    "Networks are statistical summaries of correlated activity across 1000 "
    "resting brains, not anatomical objects with edges. The boundaries are "
    "approximations, the seven-network division is one defensible choice "
    "among several, and no network 'does' a psychological function on its "
    "own."
)
