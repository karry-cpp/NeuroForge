"""
neuroforge.anatomy.noise
========================

Deterministic gradient (Perlin-style) noise in pure Python.

Used to sculpt cortical folding (gyri and sulci) onto the base cerebral
shape. Pure stdlib so the whole app installs with zero dependencies.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

_GRAD3: List[Tuple[float, float, float]] = [
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
]


class Noise3D:
    """Classic 3-D gradient noise with a fixed permutation table."""

    __slots__ = ("_p",)

    def __init__(self, seed: int = 1337):
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self._p = p + p          # doubled to avoid modulo in the hot loop

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _grad(self, h: int, x: float, y: float, z: float) -> float:
        gx, gy, gz = _GRAD3[h % 12]
        return gx * x + gy * y + gz * z

    def noise(self, x: float, y: float, z: float) -> float:
        """Single octave, returns roughly [-1, 1]."""
        p = self._p
        xi, yi, zi = int(math.floor(x)) & 255, \
            int(math.floor(y)) & 255, int(math.floor(z)) & 255
        xf, yf, zf = x - math.floor(x), y - math.floor(y), z - math.floor(z)
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)

        a = p[xi] + yi
        aa, ab = p[a] + zi, p[a + 1] + zi
        b = p[xi + 1] + yi
        ba, bb = p[b] + zi, p[b + 1] + zi

        g = self._grad
        x1 = _lerp(g(p[aa], xf, yf, zf), g(p[ba], xf - 1, yf, zf), u)
        x2 = _lerp(g(p[ab], xf, yf - 1, zf), g(p[bb], xf - 1, yf - 1, zf), u)
        y1 = _lerp(x1, x2, v)

        x3 = _lerp(g(p[aa + 1], xf, yf, zf - 1),
                   g(p[ba + 1], xf - 1, yf, zf - 1), u)
        x4 = _lerp(g(p[ab + 1], xf, yf - 1, zf - 1),
                   g(p[bb + 1], xf - 1, yf - 1, zf - 1), u)
        y2 = _lerp(x3, x4, v)
        return _lerp(y1, y2, w)

    def fbm(self, x: float, y: float, z: float, octaves: int = 4,
            lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """Fractal Brownian motion - stacked octaves. Returns ~[-1, 1]."""
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for _ in range(octaves):
            total += amp * self.noise(x * freq, y * freq, z * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm if norm else 0.0

    def ridged(self, x: float, y: float, z: float, octaves: int = 4,
               lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """Ridged multifractal - produces sharp creases.

        This is what makes the surface read as *sulci* (deep narrow grooves)
        rather than gentle bumps: the absolute value creates V-shaped
        valleys, which is much closer to real cortical folding.
        """
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for _ in range(octaves):
            n = 1.0 - abs(self.noise(x * freq, y * freq, z * freq))
            total += amp * n * n
            norm += amp
            amp *= gain
            freq *= lacunarity
        return (total / norm if norm else 0.0) * 2.0 - 1.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)
