"""
neuroforge.anatomy.mesh
=======================

Procedural generation of a 3-D cerebral mesh with cortical folding, plus
subcortical structures.

------------------------------------------------------------------------------
ANATOMICAL HONESTY  (read this before trusting anything you see)
------------------------------------------------------------------------------
This is a *procedurally generated anatomical approximation*, not a scan and
not a segmented atlas mesh.

  * The overall proportions, the lobar arrangement, the Sylvian and
    longitudinal fissures, and the positions/shapes of the subcortical
    structures are modelled on standard anatomy and placed at approximately
    MNI-like stereotaxic coordinates (millimetres, RAS convention).
  * The gyral/sulcal pattern is *synthetic*. Real cortical folding is
    individually unique; the folds you see here are noise-generated and do
    not correspond to named gyri.
  * Cortical "regions" are defined by geometric rules over position, not by
    cytoarchitecture, myelin mapping or a probabilistic atlas. They are
    teaching approximations, deliberately generous in extent.

Coordinate convention (RAS, millimetres):
    +x = right      +y = anterior     +z = superior
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .noise import Noise3D

Vec3 = Tuple[float, float, float]


# ==========================================================================
#  Small vector helpers
# ==========================================================================

def _norm(v: Vec3) -> Vec3:
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _smin(a: float, b: float, k: float) -> float:
    """Polynomial smooth minimum - blends SDF shapes organically."""
    h = max(0.0, min(1.0, 0.5 + 0.5 * (b - a) / k))
    return b * (1 - h) + a * h - k * h * (1 - h)


def _ellipsoid(p: Vec3, c: Vec3, r: Vec3) -> float:
    """Approximate signed distance to an axis-aligned ellipsoid."""
    dx, dy, dz = (p[0] - c[0]) / r[0], (p[1] - c[1]) / r[1], (p[2] - c[2]) / r[2]
    k = math.sqrt(dx * dx + dy * dy + dz * dz)
    return (k - 1.0) * min(r)


# ==========================================================================
#  Mesh container
# ==========================================================================

@dataclass
class Mesh:
    """Indexed triangle mesh, ready to hand to a GPU."""
    positions: List[float] = field(default_factory=list)   # xyz flat
    normals: List[float] = field(default_factory=list)     # xyz flat
    indices: List[int] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)        # per-vertex region
    # Sulcal depth per vertex, when a real anatomical surface is in use.
    # Negative = gyral crown, positive = sulcal fundus. Empty for the
    # procedural cortex, which has no measured depth to report.
    sulc: List[float] = field(default_factory=list)
    # Fine-grained anatomical parcel per vertex (Destrieux), plus the key ->
    # readable-name table. This is what lets a click answer "superior
    # temporal gyrus" rather than merely "temporal".
    fine: List[int] = field(default_factory=list)
    fine_names: Dict[int, str] = field(default_factory=dict)
    # Yeo 2011 functional network id per vertex (0-7). Empty unless the
    # network atlas is available.
    network: List[int] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    def compute_normals(self) -> None:
        """Area-weighted smooth vertex normals."""
        n = [0.0] * len(self.positions)
        P = self.positions
        for i in range(0, len(self.indices), 3):
            a, b, c = self.indices[i], self.indices[i + 1], self.indices[i + 2]
            pa = (P[a * 3], P[a * 3 + 1], P[a * 3 + 2])
            pb = (P[b * 3], P[b * 3 + 1], P[b * 3 + 2])
            pc = (P[c * 3], P[c * 3 + 1], P[c * 3 + 2])
            # un-normalised cross product = 2 * area * unit normal
            fx, fy, fz = _cross(_sub(pb, pa), _sub(pc, pa))
            for v in (a, b, c):
                n[v * 3] += fx
                n[v * 3 + 1] += fy
                n[v * 3 + 2] += fz
        for i in range(0, len(n), 3):
            l = math.sqrt(n[i] ** 2 + n[i + 1] ** 2 + n[i + 2] ** 2) or 1.0
            n[i] /= l
            n[i + 1] /= l
            n[i + 2] /= l
        self.normals = n


# ==========================================================================
#  Icosphere - uniform triangulation, no pole pinching
# ==========================================================================

def icosphere(subdivisions: int = 5) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts: List[Vec3] = [_norm(v) for v in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    faces: List[Tuple[int, int, int]] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]

    for _ in range(subdivisions):
        cache: Dict[Tuple[int, int], int] = {}
        new_faces: List[Tuple[int, int, int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            hit = cache.get(key)
            if hit is not None:
                return hit
            va, vb = verts[a], verts[b]
            m = _norm(((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2,
                       (va[2] + vb[2]) / 2))
            verts.append(m)
            idx = len(verts) - 1
            cache[key] = idx
            return idx

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return verts, faces


# ==========================================================================
#  The cerebrum as a signed distance field
# ==========================================================================

# Bulk proportions, millimetres. Roughly adult-average cerebral dimensions:
# ~140 mm wide, ~170 mm long, ~120 mm tall.
RX, RY, RZ = 66.0, 80.0, 60.0


def cerebrum_sdf(p: Vec3) -> float:
    """Negative inside, positive outside. Units are approximate mm."""
    x, y, z = p
    ax = abs(x)

    # --- main cerebral mass -------------------------------------------
    d = _ellipsoid(p, (0, 2, 4), (RX, RY, RZ))

    # --- frontal lobes: narrower and slightly tucked under -------------
    d = _smin(d, _ellipsoid(p, (0, 46, 6), (50, 40, 44)), 22.0)

    # --- occipital pole: tapered, tilted down --------------------------
    d = _smin(d, _ellipsoid(p, (0, -56, -2), (46, 32, 38)), 20.0)

    # --- temporal lobes: lateral, inferior, pointing forward -----------
    for s in (-1.0, 1.0):
        d = _smin(d, _ellipsoid(p, (s * 44, -4, -30), (26, 48, 24)), 12.0)

    # --- flatten the inferior surface (the brain sits on the skull base)
    d = max(d, -(z + 46.0))

    # --- LONGITUDINAL FISSURE ------------------------------------------
    # The deep midline cleft dividing the two hemispheres. It is carved from
    # the vertex downward and stops above the corpus callosum, so the
    # hemispheres remain joined inferiorly - which is anatomically correct.
    #
    # Implemented as an SDF *subtraction*: build the solid we want removed
    # (a thin midline slab, open at the top), then d = max(d, -slab).
    cleft_half = 3.2                      # half-width of the cleft, mm
    z_floor = -6.0                        # callosum level: fissure stops here
    slab = max(ax - cleft_half, z_floor - z)
    d = max(d, -slab)

    # --- SYLVIAN (LATERAL) FISSURE -------------------------------------
    # The deep groove separating the temporal lobe below from the frontal
    # and parietal lobes above. Modelled as a tilted, tapering slab on the
    # lateral surface of each hemisphere.
    if -76.0 < y < 56.0:
        plane = z - (-8.0 + 0.20 * y)     # the fissure rises anteriorly
        thick = 3.4
        # only present on the lateral convexity, fading out medially
        lateral = 24.0 - ax               # negative once we are lateral enough
        syl = max(abs(plane) - thick, lateral, -(ax - 12.0))
        d = max(d, -syl)

    # --- CENTRAL SULCUS ------------------------------------------------
    # The landmark separating frontal from parietal lobe. It runs obliquely
    # from near the vertex (about y=-30, z=70) down and forward to meet the
    # Sylvian fissure (about y=+5, z=16).
    cs = z - (24.0 - 1.55 * y - 0.10 * ax)
    d = max(d, -max(abs(cs) - 2.6, ax - 62.0, -(z + 4.0)))

    return d


def _ray_surface(direction: Vec3, lo: float = 6.0, hi: float = 150.0,
                 steps: int = 26) -> float:
    """Binary-search the radius at which the SDF crosses zero."""
    dx, dy, dz = direction
    a, b = lo, hi
    for _ in range(steps):
        m = 0.5 * (a + b)
        if cerebrum_sdf((dx * m, dy * m, dz * m)) < 0.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ==========================================================================
#  Cortical folding
# ==========================================================================

def _folding(p: Vec3, n1: Noise3D, n2: Noise3D) -> float:
    """Displacement in mm applied along the surface normal.

    REAL: cortical folding creates gyri (ridges) and sulci (grooves) which
    roughly triple the cortical surface area packed into the skull, and
    folding is under both genetic and mechanical-buckling influence.
    APPROXIMATION: this specific pattern is synthetic noise. It is not your
    folding pattern and the folds are not named gyri.
    """
    x, y, z = p
    s = 0.021                      # spatial frequency ~ gyral wavelength
    # Ridged noise gives V-shaped valleys, which read as sulci.
    ridge = n1.ridged(x * s, y * s, z * s, octaves=3, gain=0.52)
    detail = n2.fbm(x * s * 2.7, y * s * 2.7, z * s * 2.7, octaves=3)
    fold = ridge * 3.4 + detail * 0.9

    # The medial wall and the region around the corpus callosum are much
    # smoother than the lateral convexity - suppress folding there.
    medial = min(1.0, abs(x) / 12.0)
    fold *= 0.25 + 0.75 * medial
    return fold


# ==========================================================================
#  Cortical parcellation by geometric rule
# ==========================================================================
#
# APPROXIMATION: these are position-based rules, NOT a probabilistic atlas.
# They are intentionally coarse and generous. Do not use them to make any
# anatomical claim beyond "roughly this part of the brain".

CORTICAL_LABELS: Dict[str, int] = {
    "other":      0,
    "dlPFC":      1,
    "vlPFC":      2,
    "vmPFC":      3,
    "OFC":        4,
    "dACC":       5,
    "sgACC":      6,
    "motor":      7,
    "parietal":   8,
    "temporal":   9,
    "occipital": 10,
    # Added for self-referential processing. These were previously either
    # invisible (posterior cingulate was lumped into "other") or buried
    # inside "parietal", which hid the core hubs of the default mode
    # network - the network most implicated in thinking about oneself.
    "PCC":       11,
    "precuneus": 12,
    "TPJ":       13,
    "FPC":       14,
    # Added so the app covers the same ground as a general-purpose brain
    # atlas. Each of these was previously hidden inside a coarser label -
    # primary visual cortex was just "occipital", the entorhinal cortex just
    # "temporal" - which made it impossible to point at them.
    "premotor":      15,
    "somatosensory": 16,
    "SPL":           17,
    "V1":            18,
    "A1":            19,
    "MTL":           20,   # entorhinal + parahippocampal
    "fusiform":      21,
    "STS":           22,   # superior temporal sulcus / Wernicke territory
    "supramarginal": 23,
    # The insula was excluded from earlier versions of this app by design.
    # It is included now because the anterior insula is hard to leave out of
    # any honest account of interoception and of how a bodily state becomes
    # a felt emotion.
    "aINS":          24,
    "pINS":          25,
    # Medial prefrontal cortex. Previously the medial wall of the superior
    # frontal gyrus was assigned wholesale to dlPFC, which was a documented
    # compromise and a bad one: it put the medial surface into a label whose
    # name means "dorsolateral".
    "mPFC":          26,
}


def classify_cortex(p: Vec3) -> int:
    """Assign a coarse region label to a cortical surface point."""
    x, y, z = p
    ax = abs(x)
    medial = ax < 16.0

    # --- cingulate: medial surface, arched around the corpus callosum ---
    if medial and -46.0 < y < 52.0:
        # distance from an arc approximating the callosal outline
        arc = math.hypot((y - 2.0) / 46.0, (z - 6.0) / 32.0)
        if 0.62 < arc < 1.26:
            if y > 18.0 and z < 4.0:
                return CORTICAL_LABELS["sgACC"]      # subgenual ACC
            if y > 4.0:
                return CORTICAL_LABELS["dACC"]       # dorsal / anterior
    # --- prefrontal (anterior to the precentral region) -----------------
    if y > 20.0:
        if z < -14.0:
            return CORTICAL_LABELS["OFC"]            # orbital surface
        if medial and z < 26.0:
            return CORTICAL_LABELS["vmPFC"]
        if z > 22.0:
            return CORTICAL_LABELS["dlPFC"]
        if ax > 30.0:
            return CORTICAL_LABELS["vlPFC"]
        return CORTICAL_LABELS["dlPFC"]
    # --- the rest, coarsely --------------------------------------------
    if y > -6.0 and z > 26.0:
        return CORTICAL_LABELS["motor"]
    if z < -8.0 and ax > 26.0 and y > -58.0:
        return CORTICAL_LABELS["temporal"]
    if y < -58.0:
        return CORTICAL_LABELS["occipital"]
    if z > 10.0:
        return CORTICAL_LABELS["parietal"]
    return CORTICAL_LABELS["other"]


# ==========================================================================
#  Build the cortex
# ==========================================================================

def build_cortex(subdivisions: int = 6, fold: bool = True,
                 seed: int = 20240) -> Mesh:
    """Generate the folded cerebral surface.

    subdivisions=6 -> 40 962 vertices / 81 920 triangles (good detail)
    subdivisions=5 -> 10 242 vertices (fast, for development)
    """
    dirs, faces = icosphere(subdivisions)
    n1, n2 = Noise3D(seed), Noise3D(seed + 977)

    m = Mesh()
    pos, lab = m.positions, m.labels
    for d in dirs:
        r = _ray_surface(d)
        p = (d[0] * r, d[1] * r, d[2] * r)
        if fold:
            disp = _folding(p, n1, n2)
            p = (p[0] + d[0] * disp, p[1] + d[1] * disp, p[2] + d[2] * disp)
        pos.extend(p)
        lab.append(classify_cortex(p))

    for a, b, c in faces:
        m.indices.extend((a, b, c))
    m.compute_normals()
    return m


# ==========================================================================
#  Subcortical structures
# ==========================================================================

def _deform_sphere(subdiv: int, fn) -> Mesh:
    dirs, faces = icosphere(subdiv)
    m = Mesh()
    for d in dirs:
        m.positions.extend(fn(d))
    for a, b, c in faces:
        m.indices.extend((a, b, c))
    m.compute_normals()
    return m


def build_blob(center: Vec3, radii: Vec3, subdiv: int = 3,
               bend: Vec3 = (0.0, 0.0, 0.0), lumpiness: float = 0.0,
               seed: int = 5) -> Mesh:
    """An organic ellipsoidal structure with optional bend and surface lumps."""
    n = Noise3D(seed)

    def fn(d: Vec3) -> Vec3:
        k = 1.0 + (lumpiness * n.fbm(d[0] * 2.2, d[1] * 2.2, d[2] * 2.2, 3)
                   if lumpiness else 0.0)
        x = d[0] * radii[0] * k
        y = d[1] * radii[1] * k
        z = d[2] * radii[2] * k
        t = y / max(radii[1], 1e-6)
        x += bend[0] * t * t
        z += bend[2] * t * t
        return (center[0] + x, center[1] + y, center[2] + z)

    return _deform_sphere(subdiv, fn)


def build_tube(path: Sequence[Vec3], radius_fn, segments: int = 10,
               rings: int = 24, aspect: float = 1.0) -> Mesh:
    """Sweep a circular (or elliptical) cross-section along a Catmull-Rom path.

    Used for the hippocampus, which is a curved elongated structure arcing
    through the medial temporal lobe - an ellipsoid would misrepresent it -
    and for the cingulate cortex, which is a flattened sheet arching around
    the corpus callosum (use aspect < 1 to flatten it).
    """
    def sample(t: float) -> Vec3:
        n = len(path) - 1
        f = t * n
        i = min(n - 1, int(f))
        u = f - i
        p0 = path[max(0, i - 1)]
        p1, p2 = path[i], path[i + 1]
        p3 = path[min(n, i + 2)]
        u2, u3 = u * u, u * u * u
        return tuple(
            0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * u
                   + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u2
                   + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u3)
            for k in range(3))                                   # type: ignore

    m = Mesh()
    prev_up: Vec3 = (0.0, 0.0, 1.0)
    for i in range(rings):
        t = i / (rings - 1)
        c = sample(t)
        nxt = sample(min(1.0, t + 1e-3))
        tan = _norm(_sub(nxt, c)) if t < 1.0 else prev_up
        # parallel-transport-ish frame to avoid twisting
        side = _cross(tan, prev_up)
        if math.hypot(*side) < 1e-4:
            side = _cross(tan, (1.0, 0.0, 0.0))
        side = _norm(side)
        up = _norm(_cross(side, tan))
        prev_up = up
        r = radius_fn(t)
        for j in range(segments):
            a = 2 * math.pi * j / segments
            ca, sa = math.cos(a) * aspect, math.sin(a)
            m.positions.extend((
                c[0] + (side[0] * ca + up[0] * sa) * r,
                c[1] + (side[1] * ca + up[1] * sa) * r,
                c[2] + (side[2] * ca + up[2] * sa) * r))
    for i in range(rings - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c2 = (i + 1) * segments + j
            d2 = (i + 1) * segments + (j + 1) % segments
            m.indices.extend((a, c2, b, b, c2, d2))
    m.compute_normals()
    return m


def mirror_x(mesh: Mesh) -> Mesh:
    """Mirror a structure to the other hemisphere (flip winding order)."""
    out = Mesh()
    out.positions = list(mesh.positions)
    for i in range(0, len(out.positions), 3):
        out.positions[i] = -out.positions[i]
    for i in range(0, len(mesh.indices), 3):
        out.indices.extend((mesh.indices[i], mesh.indices[i + 2],
                            mesh.indices[i + 1]))
    out.compute_normals()
    return out


def merge(*meshes: Mesh) -> Mesh:
    out = Mesh()
    off = 0
    for m in meshes:
        out.positions.extend(m.positions)
        out.normals.extend(m.normals)
        out.indices.extend(i + off for i in m.indices)
        off += m.vertex_count
    return out
