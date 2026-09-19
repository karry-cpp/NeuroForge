"""
neuroforge.anatomy.tracts
=========================

Route the simulation's pathways through white matter instead of drawing a
bare arc between two points.

------------------------------------------------------------------------------
WHAT THIS IS, AND EMPHATICALLY IS NOT
------------------------------------------------------------------------------
This is NOT tractography. No diffusion data is involved, no fibre orientation
is estimated, and nothing here recovers the course of any real white-matter
bundle.

What it does is much more modest: given two endpoints, it finds a path
between them that prefers to travel through voxels the MNI152 tissue
segmentation says are white matter. The result is a *plausible* route rather
than a straight line that cuts through a ventricle, crosses the midline
through the third ventricle, or briefly leaves the brain.

So the honest claim is "this curve stays inside the white matter", not "this
curve is the uncinate fasciculus". Several of the relationships this app
draws are polysynaptic and have no single bundle at all - see anchors.py.

If the probability map has not been downloaded, every function here returns
None and the renderer falls back to its synthetic arc.
"""

from __future__ import annotations

import heapq
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import fetch, nifti
from .subcortex_real import _invert_affine, _MNI152_TO_305, _MNI305_TO_152

Vec3 = Tuple[float, float, float]

# Below this the voxel is treated as non-white-matter. The map is a partial
# volume estimate, so a hard threshold would break the path wherever a gyrus
# thins; the cost function grades it instead.
_WM_FLOOR = 0.05

# How strongly the search avoids non-white-matter. High enough that a detour
# through the corona radiata beats a shortcut across a ventricle, low enough
# that the path can still leave white matter at both ends - the endpoints are
# nuclei and cortex, which are grey.
_AVOID = 6.0

_volume: Optional[nifti.Volume] = None
_inv_affine = None


def available() -> bool:
    p = fetch.paths().get("wm")
    return bool(p) and os.path.exists(p) and os.path.getsize(p) > 1024


def _load() -> nifti.Volume:
    global _volume, _inv_affine
    if _volume is None:
        _volume = nifti.read(fetch.paths()["wm"])
        _inv_affine = _invert_affine(_volume.affine[:3])
    return _volume


def _to_152(p: Vec3) -> Vec3:
    m = _MNI305_TO_152
    return tuple(m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]
                 for r in range(3))


def _to_305(p: Vec3) -> Vec3:
    m = _MNI152_TO_305
    return tuple(m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]
                 for r in range(3))


def _voxel(p: Vec3) -> Tuple[int, int, int]:
    m = _inv_affine
    return tuple(int(round(
        m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]))
        for r in range(3))


# 26-connected neighbourhood, with the true step length for each offset so
# diagonal moves are not treated as cheap.
_NEIGHBOURS = [
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def route(a_world: Vec3, b_world: Vec3,
          pad: int = 12) -> Optional[List[List[float]]]:
    """
    A* from a to b, preferring white matter. Points in and out are fsaverage
    millimetres; the search happens in the MNI152 voxel grid.

    `pad` grows the searched box around the two endpoints, in voxels. Without
    it the path is trapped in the box that exactly contains them and cannot
    bow around anything.
    """
    if not available():
        return None
    vol = _load()
    nx, ny, nz = vol.shape

    start = _voxel(_to_152(a_world))
    goal = _voxel(_to_152(b_world))

    lo = [max(0, min(start[i], goal[i]) - pad) for i in range(3)]
    hi = [min((nx, ny, nz)[i] - 1, max(start[i], goal[i]) + pad)
          for i in range(3)]
    for i in range(3):
        if not (lo[i] <= start[i] <= hi[i] and lo[i] <= goal[i] <= hi[i]):
            return None

    data = vol.data

    def wm(i: int, j: int, k: int) -> float:
        v = data[i + nx * (j + ny * k)]
        # The map ships as float 0..1 in some releases and uint8 0..255 in
        # others; normalise rather than trusting one of them.
        return v / 255.0 if v > 1.0 else float(v)

    def cost(i: int, j: int, k: int) -> float:
        p = wm(i, j, k)
        if p < _WM_FLOOR:
            p = 0.0
        return 1.0 + _AVOID * (1.0 - p) ** 2

    def h(n: Tuple[int, int, int]) -> float:
        return math.dist(n, goal)

    open_heap = [(h(start), 0.0, start)]
    best: Dict[Tuple[int, int, int], float] = {start: 0.0}
    came: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    seen = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            break
        if cur in seen:
            continue
        seen.add(cur)
        ci, cj, ck = cur
        for dx, dy, dz, step in _NEIGHBOURS:
            ni, nj, nk = ci + dx, cj + dy, ck + dz
            if not (lo[0] <= ni <= hi[0] and lo[1] <= nj <= hi[1]
                    and lo[2] <= nk <= hi[2]):
                continue
            ng = g + step * cost(ni, nj, nk)
            key = (ni, nj, nk)
            if ng < best.get(key, 1e30):
                best[key] = ng
                came[key] = cur
                heapq.heappush(open_heap, (ng + h(key), ng, key))
    else:
        return None

    path = [goal]
    while path[-1] != start:
        prev = came.get(path[-1])
        if prev is None:
            return None
        path.append(prev)
    path.reverse()

    world = [list(_to_305(vol.to_world(*v))) for v in path]
    return _smooth(_thin(world))


def _thin(pts: List[List[float]], keep: int = 14) -> List[List[float]]:
    """Evenly subsample. A voxel path has one point per 2 mm, which is far
    more than a curve through it needs."""
    if len(pts) <= keep:
        return pts
    step = (len(pts) - 1) / (keep - 1)
    return [pts[int(round(i * step))] for i in range(keep)]


def _smooth(pts: List[List[float]], rounds: int = 4) -> List[List[float]]:
    """Moving average over the interior. Voxel paths are staircases; without
    this the tube visibly zig-zags at every diagonal step."""
    out = [list(p) for p in pts]
    for _ in range(rounds):
        nxt = [list(out[0])]
        for i in range(1, len(out) - 1):
            nxt.append([(out[i - 1][c] + 2.0 * out[i][c] + out[i + 1][c]) / 4.0
                        for c in range(3)])
        nxt.append(list(out[-1]))
        out = nxt
    return out
