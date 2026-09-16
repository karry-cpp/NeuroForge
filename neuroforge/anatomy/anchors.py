"""
neuroforge.anatomy.anchors
==========================

The bridge between the SIMULATION layer (`neuroforge.atlas`, which knows
about pathways and connection strengths) and the ANATOMY layer (which knows
about 3-D millimetre coordinates).

Keeping this mapping in one small module is what lets the simulation engine
stay completely independent of the renderer: the model never knows where
anything is in space, and the renderer never knows what a learning rate is.

------------------------------------------------------------------------------
HONESTY
------------------------------------------------------------------------------
An anchor is a single point standing in for a whole structure. Drawing a
curved tube between two anchors is a VISUALISATION OF A FUNCTIONAL
RELATIONSHIP. It is not a tractography result, it does not follow the real
course of any white-matter bundle, and several of the relationships shown
are known to be polysynaptic (relayed through structures not drawn).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Vec3 = Tuple[float, float, float]

# Simulation node id -> right-hemisphere anchor point (mm, RAS).
# Bilateral nodes are mirrored to -x automatically.
ANCHORS: Dict[str, Vec3] = {
    # --- prefrontal (cortical surface, pulled slightly inward so the
    #     pathway tubes sit just under the surface rather than floating)
    "dlPFC": (40, 42, 32),
    "vlPFC": (48, 36, 2),
    "vmPFC": (10, 48, -12),
    "dACC":  (7, 20, 26),

    # --- limbic
    "BLA":   (24, -2, -20),
    "CeA":   (18, -5, -14),
    "HPC":   (27, -24, -16),

    # --- basal ganglia
    "DMS":   (16, 6, 14),      # caudate head/body (associative)
    "DLS":   (28, 0, 2),       # putamen (sensorimotor)
    "NAcc":  (10, 13, -9),     # ventral striatum
    "THAL":  (12, -18, 7),

    # --- neuromodulatory
    "VTA":   (5, -20, -14),
    "LC":    (7, -28, -32),    # pontine tegmentum, approximate
}

# Which anchors are midline-ish and should NOT be mirrored.
MIDLINE = {"VTA"}


def anchor(node_id: str, side: int = 1) -> Vec3:
    """side = +1 right hemisphere, -1 left."""
    p = ANCHORS[node_id]
    if node_id in MIDLINE:
        return (p[0] * (1 if side > 0 else -1) * 0.6, p[1], p[2])
    return (p[0] * side, p[1], p[2])


def anchor_table() -> List[Dict[str, object]]:
    """Serialisable form for the renderer."""
    return [{"id": k, "right": list(v), "midline": k in MIDLINE}
            for k, v in ANCHORS.items()]


# Which anatomical structure should visually light up when a given
# simulation node is active. Cortical entries refer to surface parcels.
NODE_TO_STRUCTURE: Dict[str, str] = {
    "BLA": "amygdala", "CeA": "amygdala",
    "HPC": "hippocampus",
    "DMS": "caudate", "DLS": "putamen", "NAcc": "accumbens",
    "THAL": "thalamus", "VTA": "vta", "LC": "brainstem",
    "dACC": "acc",
}

NODE_TO_CORTEX: Dict[str, str] = {
    "dlPFC": "dlPFC", "vlPFC": "vlPFC", "vmPFC": "vmPFC",
}
