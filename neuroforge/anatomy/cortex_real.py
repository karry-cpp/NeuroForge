"""
Build the cortical surface from real anatomy (FreeSurfer fsaverage).

This is the preferred source of geometry. `mesh.build_cortex` remains as a
fallback for when the assets have not been downloaded, but everything here
is measured rather than invented:

  * the folds are the average folds of 40 real brains, not noise;
  * the region boundaries follow the Desikan-Killiany atlas, not my
    hand-tuned inequalities;
  * sulcal depth is measured, so shading darkens real sulci.

WHAT IS STILL AN APPROXIMATION
------------------------------
The mapping from 35 Desikan-Killiany parcels onto the 11 coarse labels this
app uses is a judgement call, and it is spelled out in DK_TO_LABEL below so
it can be argued with. Two compromises are worth naming explicitly:

  * Desikan-Killiany has no subgenual-ACC parcel. We split
    `rostralanteriorcingulate` at z = 0: below the genu of the corpus
    callosum we call it sgACC. That is a reasonable geometric proxy for a
    boundary that the atlas simply does not draw.
  * `superiorfrontal` is one large parcel spanning both the dorsolateral
    convexity and the medial wall. We assign it to dlPFC, which
    under-represents the medial prefrontal contribution.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

from . import fetch, gifti
from .mesh import CORTICAL_LABELS as L
from .mesh import Mesh

# Desikan-Killiany parcel -> coarse label used by this application.
# Parcels deliberately mapped to "other": unknown and corpuscallosum (the
# medial wall, which is not cortex at all) and insula (excluded by design).
DK_TO_LABEL: Dict[str, str] = {
    "unknown": "other", "corpuscallosum": "other",

    # The insula is a single Desikan parcel. It is split below into anterior
    # and posterior at the level of the central insular sulcus, because the
    # two halves do recognisably different jobs and only the anterior part
    # is the one people mean when they talk about interoception.
    "insula": "aINS",

    # Caudal middle frontal is, functionally, premotor rather than
    # dorsolateral prefrontal. It plans movement; it does not do the
    # cognitive-control job the simulation attributes to dlPFC.
    "caudalmiddlefrontal": "premotor",
    "rostralmiddlefrontal": "dlPFC",
    "superiorfrontal": "dlPFC",

    # Frontal pole gets its own label rather than being swallowed by dlPFC.
    # It is the region most associated with metacognition - thinking about
    # your own thinking - which is central to deliberate self-development.
    "frontalpole": "FPC",

    "parsopercularis": "vlPFC", "parstriangularis": "vlPFC",
    "parsorbitalis": "vlPFC",

    "medialorbitofrontal": "vmPFC",
    "lateralorbitofrontal": "OFC",

    "caudalanteriorcingulate": "dACC", "rostralanteriorcingulate": "dACC",

    "precentral": "motor", "paracentral": "motor",

    # Postcentral gyrus is primary somatosensory cortex - the body map.
    # Filing it under a generic "parietal" label lost the single most
    # cleanly understood piece of cortex there is.
    "postcentral": "somatosensory",
    "superiorparietal": "SPL",
    "supramarginal": "supramarginal",

    # Posterior cingulate and precuneus are the posterior midline core of
    # the default mode network. Leaving them in "other" made the single
    # most self-relevant part of the cortex literally invisible.
    "posteriorcingulate": "PCC", "isthmuscingulate": "PCC",
    "precuneus": "precuneus",

    # Angular gyrus, the parietal component of the temporoparietal
    # junction: self-other distinction and perspective taking.
    "inferiorparietal": "TPJ",

    # The temporal lobe is not one thing. Auditory cortex, the fusiform
    # face area, the medial temporal memory system and the superior temporal
    # sulcus do very different jobs, and the last two matter directly to
    # fear learning and to reading other people.
    "transversetemporal": "A1",
    "entorhinal": "MTL", "parahippocampal": "MTL",
    "fusiform": "fusiform",
    "superiortemporal": "STS", "bankssts": "STS",
    "middletemporal": "temporal", "inferiortemporal": "temporal",
    "temporalpole": "temporal",

    # Pericalcarine cortex is V1, the first cortical stop for vision.
    "pericalcarine": "V1",
    "lateraloccipital": "occipital", "lingual": "occipital",
    "cuneus": "occipital",
}

# ---------------------------------------------------------------------------
#  Yeo 2011 functional networks
# ---------------------------------------------------------------------------
# The Desikan-Killiany atlas describes *anatomy*. Networks are a different
# and complementary description: sets of regions whose activity covaries at
# rest. The default mode network in particular has no single anatomical
# home - it spans medial prefrontal, posterior cingulate, precuneus, angular
# gyrus and lateral temporal cortex - so it can only be shown honestly as a
# network overlay, not as one coloured lump.
#
#   Yeo et al. (2011), J. Neurophysiol. 106(3):1125-1165, n = 1000.
YEO7_NAMES: Dict[int, str] = {
    0: "Unassigned",
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal attention",
    4: "Ventral attention / salience",
    5: "Limbic",
    6: "Frontoparietal control",
    7: "Default mode",
}

YEO7_COLORS: Dict[int, str] = {
    0: "#3a4356", 1: "#7b4fa8", 2: "#4a7fd4", 3: "#3f9e6a",
    4: "#c264d8", 5: "#d8cf8a", 6: "#e8973c", 7: "#e05a6b",
}


# ==========================================================================
#  Destrieux: fine-grained anatomical naming
# ==========================================================================
# Destrieux labels are terse codes like "G_front_sup". Anatomical names are
# a fixed vocabulary with conventional word order ("superior frontal gyrus",
# never "frontal superior gyrus"), so they are written out explicitly rather
# than derived by rule. 75 entries is a small price for being correct.
#
# The G_/S_ distinction is preserved throughout and matters: a gyrus is a
# ridge, a sulcus is the groove between ridges. They are different places.

DESTRIEUX_NAMES: Dict[str, str] = {
    "Unknown": "Unassigned",
    "Medial_wall": "Medial wall (not cortex)",

    "G_and_S_frontomargin": "Frontomarginal gyrus and sulcus",
    "G_and_S_occipital_inf": "Inferior occipital gyrus and sulcus",
    "G_and_S_paracentral": "Paracentral lobule and sulcus",
    "G_and_S_subcentral": "Subcentral gyrus and sulcus",
    "G_and_S_transv_frontopol": "Transverse frontopolar gyri and sulci",
    "G_and_S_cingul-Ant": "Anterior cingulate gyrus and sulcus",
    "G_and_S_cingul-Mid-Ant": "Middle-anterior cingulate gyrus and sulcus",
    "G_and_S_cingul-Mid-Post": "Middle-posterior cingulate gyrus and sulcus",

    "G_cingul-Post-dorsal": "Posterior-dorsal cingulate gyrus",
    "G_cingul-Post-ventral": "Posterior-ventral cingulate gyrus",
    "G_cuneus": "Cuneus",
    "G_front_inf-Opercular": "Inferior frontal gyrus, pars opercularis",
    "G_front_inf-Orbital": "Inferior frontal gyrus, pars orbitalis",
    "G_front_inf-Triangul": "Inferior frontal gyrus, pars triangularis",
    "G_front_middle": "Middle frontal gyrus",
    "G_front_sup": "Superior frontal gyrus",
    "G_Ins_lg_and_S_cent_ins": "Long insular gyrus and central insular sulcus",
    "G_insular_short": "Short insular gyri",
    "G_occipital_middle": "Middle occipital gyrus",
    "G_occipital_sup": "Superior occipital gyrus",
    "G_oc-temp_lat-fusifor": "Fusiform gyrus",
    "G_oc-temp_med-Lingual": "Lingual gyrus",
    "G_oc-temp_med-Parahip": "Parahippocampal gyrus",
    "G_orbital": "Orbital gyri",
    "G_pariet_inf-Angular": "Angular gyrus",
    "G_pariet_inf-Supramar": "Supramarginal gyrus",
    "G_parietal_sup": "Superior parietal lobule",
    "G_postcentral": "Postcentral gyrus",
    "G_precentral": "Precentral gyrus",
    "G_precuneus": "Precuneus",
    "G_rectus": "Gyrus rectus",
    "G_subcallosal": "Subcallosal gyrus",
    "G_temp_sup-G_T_transv": "Transverse temporal gyrus (Heschl's)",
    "G_temp_sup-Lateral": "Superior temporal gyrus, lateral",
    "G_temp_sup-Plan_polar": "Planum polare",
    "G_temp_sup-Plan_tempo": "Planum temporale",
    "G_temporal_inf": "Inferior temporal gyrus",
    "G_temporal_middle": "Middle temporal gyrus",

    "Lat_Fis-ant-Horizont": "Lateral fissure, anterior horizontal ramus",
    "Lat_Fis-ant-Vertical": "Lateral fissure, anterior vertical ramus",
    "Lat_Fis-post": "Lateral fissure, posterior segment",
    "Pole_occipital": "Occipital pole",
    "Pole_temporal": "Temporal pole",

    "S_calcarine": "Calcarine sulcus",
    "S_central": "Central sulcus",
    "S_cingul-Marginalis": "Marginal branch of the cingulate sulcus",
    "S_circular_insula_ant": "Anterior circular insular sulcus",
    "S_circular_insula_inf": "Inferior circular insular sulcus",
    "S_circular_insula_sup": "Superior circular insular sulcus",
    "S_collat_transv_ant": "Anterior transverse collateral sulcus",
    "S_collat_transv_post": "Posterior transverse collateral sulcus",
    "S_front_inf": "Inferior frontal sulcus",
    "S_front_middle": "Middle frontal sulcus",
    "S_front_sup": "Superior frontal sulcus",
    "S_interm_prim-Jensen": "Intermediate primus sulcus (of Jensen)",
    "S_intrapariet_and_P_trans": "Intraparietal and transverse parietal sulcus",
    "S_oc_middle_and_Lunatus": "Middle occipital and lunate sulcus",
    "S_oc_sup_and_transversal": "Superior and transverse occipital sulcus",
    "S_occipital_ant": "Anterior occipital sulcus",
    "S_oc-temp_lat": "Lateral occipitotemporal sulcus",
    "S_oc-temp_med_and_Lingual": "Medial occipitotemporal and lingual sulcus",
    "S_orbital_lateral": "Lateral orbital sulcus",
    "S_orbital_med-olfact": "Medial orbital (olfactory) sulcus",
    "S_orbital-H_Shaped": "Orbital H-shaped sulcus",
    "S_parieto_occipital": "Parieto-occipital sulcus",
    "S_pericallosal": "Pericallosal sulcus",
    "S_postcentral": "Postcentral sulcus",
    "S_precentral-inf-part": "Inferior precentral sulcus",
    "S_precentral-sup-part": "Superior precentral sulcus",
    "S_suborbital": "Suborbital sulcus",
    "S_subparietal": "Subparietal sulcus",
    "S_temporal_inf": "Inferior temporal sulcus",
    "S_temporal_sup": "Superior temporal sulcus",
    "S_temporal_transverse": "Transverse temporal sulcus",
}


def humanise(code: str) -> str:
    """Turn a Destrieux code into the name an anatomist would use."""
    if code in DESTRIEUX_NAMES:
        return DESTRIEUX_NAMES[code]
    # Unrecognised codes are reported verbatim rather than guessed at, so a
    # wrong name is never invented.
    return code.replace("_", " ")


def kind_of(code: str) -> str:
    """'gyrus', 'sulcus', 'both' or '' - the fold type this parcel names."""
    if code.startswith("G_and_S_"):
        return "both"
    if code.startswith("G_"):
        return "gyrus"
    if code.startswith("S_") or code.startswith("Lat_Fis"):
        return "sulcus"
    return ""


def _hemi(side: str) -> Mesh:
    """Load one hemisphere as a labelled Mesh in fsaverage RAS millimetres."""
    p = fetch.paths()
    pos, idx = gifti.read_surface(p[f"pial_{side}"])
    sulc, _ = gifti.read_scalars(p[f"sulc_{side}"])
    keys, names = gifti.read_scalars(p[f"aparc_{side}"])

    # Translate atlas keys -> our coarse label integers, once.
    key_to_label = {}
    for k, nm in names.items():
        coarse = DK_TO_LABEL.get(nm.lower(), "other")
        key_to_label[k] = L[coarse]

    labels: List[int] = [0] * len(keys)
    ra_key = next((k for k, n in names.items()
                   if n.lower() == "rostralanteriorcingulate"), None)
    ins_key = next((k for k, n in names.items()
                    if n.lower() == "insula"), None)
    sf_key = next((k for k, n in names.items()
                   if n.lower() == "superiorfrontal"), None)
    for i, k in enumerate(keys):
        lab = key_to_label.get(k, 0)
        x, y, z = pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]

        # Subgenual split: rostral ACC lying below the genu.
        if ra_key is not None and k == ra_key and z < 0.0:
            lab = L["sgACC"]

        # Insula: anterior vs posterior. The anatomical divider is the
        # central insular sulcus, which runs obliquely; y = 0 is a flat
        # approximation to it, chosen because it sits close to the parcel's
        # own midpoint and is easy to argue with.
        elif ins_key is not None and k == ins_key and y < 0.0:
            lab = L["pINS"]

        # Medial prefrontal: the medial wall of the superior frontal gyrus,
        # anterior to the supplementary motor area. Two conditions, both
        # needed - |x| alone would drag the SMA in, and y alone would take
        # the dorsolateral convexity.
        elif sf_key is not None and k == sf_key and abs(x) < 14.0 and y > 20.0:
            lab = L["mPFC"]

        labels[i] = lab

    m = Mesh(positions=list(pos), indices=list(idx),
             labels=labels, sulc=list(sulc))

    # Fine anatomical naming, when the Destrieux atlas is present.
    dkey = f"destrieux_{side}"
    if dkey in p and os.path.exists(p[dkey]):
        fine, fnames = gifti.read_scalars(p[dkey])
        m.fine = list(fine)
        m.fine_names = {int(k): (humanise(v), kind_of(v))
                        for k, v in fnames.items()}

    # Functional network membership, when the Yeo atlas is present.
    ykey = f"yeo_{side}"
    if ykey in p and os.path.exists(p[ykey]):
        net, _ = gifti.read_scalars(p[ykey])
        m.network = [int(v) for v in net]

    m.compute_normals()
    return m


def build_cortex_real() -> Tuple[Mesh, Mesh]:
    """Return (left, right) hemisphere meshes built from real anatomy.

    Coordinates are FreeSurfer surface RAS in millimetres, which is close
    enough to MNI space that the subcortical anchor coordinates used by the
    simulation land inside the correct structures.
    """
    return _hemi("L"), _hemi("R")


def available() -> bool:
    return fetch.have_assets()
