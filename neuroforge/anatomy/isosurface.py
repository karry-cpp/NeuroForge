"""
Surface extraction from label volumes, pure Python.

Marching cubes is the textbook choice, but it needs a 256-entry triangle
table that is tedious to transcribe correctly, and on a binary mask it
produces hard voxel stair-steps. This module uses *naive surface nets*
instead: one vertex per boundary cell, placed at the centroid of the
iso-crossings on that cell's edges, with quads spanning the four cells around
each sign-changing grid edge.

That choice matters for honesty as much as for looks. A segmentation is a
label per 2 mm voxel; a blocky rendering of it is arguably the more literal
picture, but it reads as "this is what the data is like" rather than "this is
what a thalamus is like". Blurring the mask before extraction and relaxing the
mesh afterwards produces a surface that is a fair smooth interpolation of the
segmentation, without inventing detail finer than the voxel grid. The scene
metadata says which atlas it came from and at what resolution.

No numpy, so everything works on flat `array` buffers with manual indexing.
Work is confined to each label's bounding box, which is what keeps this fast
enough to run at build time (a few seconds for the whole subcortex).
"""

from __future__ import annotations

import array
from typing import Dict, List, Sequence, Tuple

from .nifti import Volume

Vec3 = Tuple[float, float, float]

# The 12 edges of a cell, as pairs of corner indices. Corner c encodes
# (dx, dy, dz) as bits 0, 1, 2.
_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # along x
    (0, 2), (1, 3), (4, 6), (5, 7),   # along y
    (0, 4), (1, 5), (2, 6), (3, 7),   # along z
]
_CORNER = [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]


def extract(
    vol: Volume,
    labels: Sequence[int],
    *,
    iso: float = 0.5,
    blur: int = 2,
    smooth: int = 8,
    smooth_amount: float = 0.55,
) -> Tuple[List[float], List[int]]:
    """
    Build a triangle mesh for the union of `labels` in `vol`.

    Returns flat world-space positions and a triangle index list. Empty lists
    if the labels are not present, which callers must handle rather than
    assume geometry exists.
    """
    want = set(labels)
    nx, ny, nz = vol.shape
    data = vol.data

    # ---- bounding box -----------------------------------------------------
    # Marching the whole 1 M-voxel volume for a structure that occupies a few
    # hundred voxels is what would make this too slow to run at build time.
    lo = [nx, ny, nz]
    hi = [-1, -1, -1]
    idx = 0
    for k in range(nz):
        for j in range(ny):
            base = idx
            for i in range(nx):
                if data[base + i] in want:
                    if i < lo[0]: lo[0] = i
                    if i > hi[0]: hi[0] = i
                    if j < lo[1]: lo[1] = j
                    if j > hi[1]: hi[1] = j
                    if k < lo[2]: lo[2] = k
                    if k > hi[2]: hi[2] = k
            idx += nx
    if hi[0] < 0:
        return [], []

    pad = blur + 2
    lo = [max(0, lo[a] - pad) for a in range(3)]
    hi = [min((nx, ny, nz)[a] - 1, hi[a] + pad) for a in range(3)]
    sx = hi[0] - lo[0] + 1
    sy = hi[1] - lo[1] + 1
    sz = hi[2] - lo[2] + 1

    # ---- local scalar field ----------------------------------------------
    f = array.array("f", bytes(4 * sx * sy * sz))
    for k in range(sz):
        for j in range(sy):
            src = lo[0] + nx * ((lo[1] + j) + ny * (lo[2] + k))
            dst = sx * (j + sy * k)
            for i in range(sx):
                if data[src + i] in want:
                    f[dst + i] = 1.0

    for _ in range(blur):
        f = _blur(f, sx, sy, sz)

    # ---- one vertex per boundary cell ------------------------------------
    verts: List[float] = []
    cell_of: Dict[int, int] = {}
    a = vol.affine

    for k in range(sz - 1):
        for j in range(sy - 1):
            for i in range(sx - 1):
                v = [
                    f[(i + cx) + sx * ((j + cy) + sy * (k + cz))]
                    for cx, cy, cz in _CORNER
                ]
                inside = 0
                for c in range(8):
                    if v[c] >= iso:
                        inside |= 1 << c
                if inside == 0 or inside == 255:
                    continue

                ax = ay = az = 0.0
                n = 0
                for e0, e1 in _EDGES:
                    a0, a1 = v[e0], v[e1]
                    if (a0 >= iso) == (a1 >= iso):
                        continue
                    t = (iso - a0) / (a1 - a0) if a1 != a0 else 0.5
                    p0, p1 = _CORNER[e0], _CORNER[e1]
                    ax += p0[0] + t * (p1[0] - p0[0])
                    ay += p0[1] + t * (p1[1] - p0[1])
                    az += p0[2] + t * (p1[2] - p0[2])
                    n += 1
                if not n:
                    continue

                vi = lo[0] + i + ax / n
                vj = lo[1] + j + ay / n
                vk = lo[2] + k + az / n
                cell_of[i + sx * (j + sy * k)] = len(verts) // 3
                verts.extend((
                    a[0][0] * vi + a[0][1] * vj + a[0][2] * vk + a[0][3],
                    a[1][0] * vi + a[1][1] * vj + a[1][2] * vk + a[1][3],
                    a[2][0] * vi + a[2][1] * vj + a[2][2] * vk + a[2][3],
                ))

    if not verts:
        return [], []

    # ---- quads around every sign-changing grid edge ----------------------
    tris: List[int] = []

    def quad(c0, c1, c2, c3, flip):
        try:
            q = (cell_of[c0], cell_of[c1], cell_of[c2], cell_of[c3])
        except KeyError:
            return  # edge touches the padded border; no complete ring of cells
        if flip:
            tris.extend((q[0], q[2], q[1], q[0], q[3], q[2]))
        else:
            tris.extend((q[0], q[1], q[2], q[0], q[2], q[3]))

    for k in range(sz - 1):
        for j in range(sy - 1):
            for i in range(sx - 1):
                here = f[i + sx * (j + sy * k)] >= iso
                # +x edge -> quad in the (y,z) plane
                if i + 1 < sx and (f[(i + 1) + sx * (j + sy * k)] >= iso) != here:
                    if j and k:
                        quad(i + sx * ((j - 1) + sy * (k - 1)),
                             i + sx * (j + sy * (k - 1)),
                             i + sx * (j + sy * k),
                             i + sx * ((j - 1) + sy * k), here)
                # +y edge -> quad in the (x,z) plane
                if j + 1 < sy and (f[i + sx * ((j + 1) + sy * k)] >= iso) != here:
                    if i and k:
                        quad((i - 1) + sx * (j + sy * (k - 1)),
                             i + sx * (j + sy * (k - 1)),
                             i + sx * (j + sy * k),
                             (i - 1) + sx * (j + sy * k), not here)
                # +z edge -> quad in the (x,y) plane
                if k + 1 < sz and (f[i + sx * (j + sy * (k + 1))] >= iso) != here:
                    if i and j:
                        quad((i - 1) + sx * ((j - 1) + sy * k),
                             i + sx * ((j - 1) + sy * k),
                             i + sx * (j + sy * k),
                             (i - 1) + sx * (j + sy * k), here)

    if smooth:
        _relax(verts, tris, smooth, smooth_amount)
    return verts, tris


def _blur(f: array.array, sx: int, sy: int, sz: int) -> array.array:
    """Separable 1-2-1 blur. Three cheap passes beat one 27-tap pass."""
    out = array.array("f", f)
    tmp = array.array("f", f)

    for k in range(sz):
        for j in range(sy):
            row = sx * (j + sy * k)
            for i in range(sx):
                c = f[row + i]
                l = f[row + i - 1] if i else c
                r = f[row + i + 1] if i + 1 < sx else c
                tmp[row + i] = 0.25 * l + 0.5 * c + 0.25 * r

    for k in range(sz):
        for j in range(sy):
            row = sx * (j + sy * k)
            up = sx * ((j - 1) + sy * k)
            dn = sx * ((j + 1) + sy * k)
            for i in range(sx):
                c = tmp[row + i]
                l = tmp[up + i] if j else c
                r = tmp[dn + i] if j + 1 < sy else c
                out[row + i] = 0.25 * l + 0.5 * c + 0.25 * r

    for k in range(sz):
        for j in range(sy):
            row = sx * (j + sy * k)
            bk = sx * (j + sy * (k - 1))
            fw = sx * (j + sy * (k + 1))
            for i in range(sx):
                c = out[row + i]
                l = out[bk + i] if k else c
                r = out[fw + i] if k + 1 < sz else c
                tmp[row + i] = 0.25 * l + 0.5 * c + 0.25 * r
    return tmp


def _relax(verts: List[float], tris: List[int], passes: int, amount: float):
    """
    Laplacian smoothing, in place.

    Surface nets still leaves a faint grid signature; a few relaxation passes
    remove it. The amount is kept modest because aggressive smoothing shrinks
    structures, and a thalamus that has quietly lost a millimetre all round is
    a worse error than a slightly bumpy one.
    """
    n = len(verts) // 3
    adj: List[set] = [set() for _ in range(n)]
    for t in range(0, len(tris), 3):
        a, b, c = tris[t], tris[t + 1], tris[t + 2]
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))

    for _ in range(passes):
        new = list(verts)
        for v in range(n):
            nb = adj[v]
            if len(nb) < 3:
                continue
            sx = sy = sz = 0.0
            for w in nb:
                sx += verts[3 * w]
                sy += verts[3 * w + 1]
                sz += verts[3 * w + 2]
            k = len(nb)
            new[3 * v] += amount * (sx / k - verts[3 * v])
            new[3 * v + 1] += amount * (sy / k - verts[3 * v + 1])
            new[3 * v + 2] += amount * (sz / k - verts[3 * v + 2])
        verts[:] = new


def normals(verts: Sequence[float], tris: Sequence[int]) -> List[float]:
    """Area-weighted vertex normals."""
    n = len(verts) // 3
    out = [0.0] * (3 * n)
    for t in range(0, len(tris), 3):
        a, b, c = tris[t] * 3, tris[t + 1] * 3, tris[t + 2] * 3
        ux, uy, uz = (verts[b] - verts[a], verts[b + 1] - verts[a + 1],
                      verts[b + 2] - verts[a + 2])
        vx, vy, vz = (verts[c] - verts[a], verts[c + 1] - verts[a + 1],
                      verts[c + 2] - verts[a + 2])
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        for p in (a, b, c):
            out[p] += cx
            out[p + 1] += cy
            out[p + 2] += cz
    for v in range(0, 3 * n, 3):
        x, y, z = out[v], out[v + 1], out[v + 2]
        m = (x * x + y * y + z * z) ** 0.5
        if m > 1e-9:
            out[v], out[v + 1], out[v + 2] = x / m, y / m, z / m
        else:
            out[v + 2] = 1.0
    return out
