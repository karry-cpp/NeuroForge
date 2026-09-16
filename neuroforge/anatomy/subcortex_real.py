"""
Real subcortical geometry from the FreeSurfer `aseg` segmentation.

Until now the subcortical structures in this app were procedural blobs placed
at atlas coordinates. They were honestly labelled as such, but they were still
the most misleading thing on screen: a learner could reasonably come away
believing a hippocampus really is a smooth bent tube. It is not - it is a
folded, layered sheet rolled into the medial temporal lobe, and it looks like
one once you extract it from real data.

This module replaces those blobs with surfaces extracted from a labelled
volume, so the shapes are measured rather than sculpted.

Coordinate spaces
-----------------
This is the part that is easy to get quietly wrong. The cortical surfaces are
fsaverage, which lives in MNI305. The `aseg` volume is MNI152NLin2009cAsym.
Those are *different spaces*: dropping one into the other unchanged leaves
structures a few millimetres out, which is exactly enough for a hippocampus to
poke through the temporal lobe.

The two are related by a known affine (Fischl et al.), applied below. It is a
small correction - roughly 1.5 mm in y and 1.2 mm in z plus sub-percent
scaling - but it is the difference between structures that sit inside the
cortex and structures that graze it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import fetch, isosurface, nifti
from .mesh import Mesh

# MNI305 -> MNI152, from FreeSurfer (Fischl et al.). We need the other
# direction, so this is inverted once at import time.
_MNI305_TO_152 = (
    (0.9975, -0.0073, 0.0176, -0.0429),
    (0.0146, 1.0009, -0.0024, 1.5496),
    (-0.0130, -0.0093, 0.9971, 1.1840),
)


def _invert_affine(m) -> Tuple[Tuple[float, ...], ...]:
    """Invert a 3x4 rigid-ish affine. Cramer's rule; no numpy available."""
    a, b, c = m[0][0], m[0][1], m[0][2]
    d, e, f = m[1][0], m[1][1], m[1][2]
    g, h, i = m[2][0], m[2][1], m[2][2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-9:
        raise ValueError("singular affine")
    inv = (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )
    t = (m[0][3], m[1][3], m[2][3])
    off = tuple(-(inv[r][0] * t[0] + inv[r][1] * t[1] + inv[r][2] * t[2])
                for r in range(3))
    return tuple(inv[r] + (off[r],) for r in range(3))


_MNI152_TO_305 = _invert_affine(_MNI305_TO_152)


# ==========================================================================
#  aseg label groups
# ==========================================================================
# Standard FreeSurfer aseg ids. Left and right are separate labels, which is
# why each entry lists both - these meshes are built bilaterally and must NOT
# be mirrored again by the caller.
ASEG: Dict[str, Tuple[int, ...]] = {
    "amygdala":     (18, 54),
    "hippocampus":  (17, 53),
    "thalamus":     (10, 49),
    "caudate":      (11, 50),
    "putamen":      (12, 51),
    "pallidum":     (13, 52),
    "accumbens":    (26, 58),
    "brainstem":    (16,),
    "cerebellum":   (8, 47, 7, 46),
    "corpus_callosum": (251, 252, 253, 254, 255),
    "ventricles":   (4, 43, 5, 44, 14, 15),
    "ventral_dc":   (28, 60),
}

# Structures where the segmentation is small enough that heavy blurring would
# dissolve them. Accumbens is ~90 voxels at 2 mm; the pallidum and ventricles
# are similarly thin.
_FINE = {"accumbens", "pallidum", "ventricles", "corpus_callosum"}


def available() -> bool:
    p = fetch.paths().get("aseg")
    return bool(p) and os.path.exists(p) and os.path.getsize(p) > 1024


_volume: Optional[nifti.Volume] = None


def _load() -> nifti.Volume:
    global _volume
    if _volume is None:
        _volume = nifti.read(fetch.paths()["aseg"])
    return _volume


def build(key: str) -> Optional[Mesh]:
    """
    Extract one structure as a bilateral mesh in fsaverage (MNI305) space.

    Returns None if the labels are absent, so callers fall back to procedural
    geometry rather than rendering nothing.
    """
    labels = ASEG.get(key)
    if not labels:
        return None

    fine = key in _FINE
    verts, tris = isosurface.extract(
        _load(), labels,
        blur=1 if fine else 2,
        smooth=4 if fine else 8,
        smooth_amount=0.45 if fine else 0.55,
    )
    if not tris:
        return None

    _to_fsaverage(verts)
    norms = isosurface.normals(verts, tris)
    return Mesh(positions=verts, normals=norms, indices=tris)


def _to_fsaverage(verts: List[float]) -> None:
    """MNI152 -> MNI305, in place."""
    m = _MNI152_TO_305
    for v in range(0, len(verts), 3):
        x, y, z = verts[v], verts[v + 1], verts[v + 2]
        verts[v] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
        verts[v + 1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
        verts[v + 2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]


def centroids(key: str) -> Optional[List[List[float]]]:
    """
    Measured centroids, one per side, in fsaverage space.

    These position the floating 3-D labels. Taking them from the data rather
    than from a hard-coded table means the pin sits on the structure even
    when the structure's shape is not what we assumed.
    """
    labels = ASEG.get(key)
    if not labels:
        return None
    vol = _load()
    nx, ny, nz = vol.shape
    sides: Dict[bool, List[float]] = {}
    counts: Dict[bool, int] = {}
    for k in range(nz):
        for j in range(ny):
            base = nx * (j + ny * k)
            for i in range(nx):
                if vol.data[base + i] in labels:
                    x, y, z = vol.to_world(i, j, k)
                    right = x >= 0
                    acc = sides.setdefault(right, [0.0, 0.0, 0.0])
                    acc[0] += x
                    acc[1] += y
                    acc[2] += z
                    counts[right] = counts.get(right, 0) + 1
    out: List[List[float]] = []
    for right in (True, False):
        if counts.get(right, 0) < 8:
            continue
        n = counts[right]
        p = [sides[right][a] / n for a in range(3)]
        _to_fsaverage(p)
        out.append(p)
    return out or None
