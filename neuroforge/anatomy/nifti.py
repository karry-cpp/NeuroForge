"""
Minimal NIfTI-1 reader, standard library only.

The project has a hard no-dependencies rule, so nibabel is not available and
neither is numpy. Fortunately a NIfTI-1 file is a fixed 348-byte header
followed by a raw voxel array, optionally gzipped, which `struct` and `array`
handle perfectly well.

Only what this app actually needs is implemented: integer label volumes
(`dseg` segmentations) and their voxel-to-world affine. Anything exotic -
scaling factors, time series, RGB voxels - raises rather than guessing.
"""

from __future__ import annotations

import array
import gzip
import struct
from dataclasses import dataclass
from typing import List, Tuple

# NIfTI datatype codes -> (array typecode, bytes per voxel)
_DTYPES = {
    2: ("B", 1),    # uint8
    4: ("h", 2),    # int16
    8: ("i", 4),    # int32
    16: ("f", 4),   # float32
    256: ("b", 1),  # int8
    512: ("H", 2),  # uint16
    768: ("I", 4),  # uint32
}


@dataclass
class Volume:
    """A 3-D label volume plus the affine that places it in world space."""

    data: array.array          # flat, x fastest, then y, then z
    shape: Tuple[int, int, int]
    affine: List[List[float]]  # 4x4, voxel index -> world mm

    def at(self, i: int, j: int, k: int) -> int:
        nx, ny, _ = self.shape
        return self.data[i + nx * (j + ny * k)]

    def to_world(self, i: float, j: float, k: float) -> Tuple[float, float, float]:
        a = self.affine
        return (
            a[0][0] * i + a[0][1] * j + a[0][2] * k + a[0][3],
            a[1][0] * i + a[1][1] * j + a[1][2] * k + a[1][3],
            a[2][0] * i + a[2][1] * j + a[2][2] * k + a[2][3],
        )


def read(path: str) -> Volume:
    """Read a .nii or .nii.gz label volume."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    # The header declares its own size as 348. If that reads back as 348 the
    # file is little-endian; if it reads as the byte-swapped 1543569408 it is
    # big-endian. This is the standard NIfTI endianness trick - there is no
    # explicit flag.
    if struct.unpack("<i", raw[:4])[0] == 348:
        end = "<"
    elif struct.unpack(">i", raw[:4])[0] == 348:
        end = ">"
    else:
        raise ValueError("not a NIfTI-1 file (bad header size)")

    dim = struct.unpack(end + "8h", raw[40:56])
    datatype = struct.unpack(end + "h", raw[70:72])[0]
    scl_slope, scl_inter = struct.unpack(end + "2f", raw[112:120])
    vox_offset = int(struct.unpack(end + "f", raw[108:112])[0])
    qform_code, sform_code = struct.unpack(end + "2h", raw[252:256])

    if dim[0] < 3:
        raise ValueError(f"expected a 3-D volume, got {dim[0]}-D")
    nx, ny, nz = dim[1], dim[2], dim[3]

    if datatype not in _DTYPES:
        raise ValueError(f"unsupported NIfTI datatype {datatype}")
    typecode, width = _DTYPES[datatype]

    n = nx * ny * nz
    buf = array.array(typecode)
    buf.frombytes(raw[vox_offset:vox_offset + n * width])
    if len(buf) != n:
        raise ValueError("voxel data truncated")
    if end == ">":
        buf.byteswap()

    # A label volume should not be scaled. If a file says otherwise, refuse
    # rather than silently returning label ids that are off by a factor.
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        raise ValueError("scaled data is not supported for label volumes")

    if sform_code > 0:
        affine = [
            list(struct.unpack(end + "4f", raw[280:296])),
            list(struct.unpack(end + "4f", raw[296:312])),
            list(struct.unpack(end + "4f", raw[312:328])),
            [0.0, 0.0, 0.0, 1.0],
        ]
    elif qform_code > 0:
        affine = _qform(raw, end, dim)
    else:
        # No spatial information at all. Fall back to pixdim scaling with the
        # origin at the centre, and say so rather than pretending otherwise.
        pix = struct.unpack(end + "8f", raw[76:108])
        affine = [
            [pix[1], 0.0, 0.0, -pix[1] * nx / 2.0],
            [0.0, pix[2], 0.0, -pix[2] * ny / 2.0],
            [0.0, 0.0, pix[3], -pix[3] * nz / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    return Volume(data=buf, shape=(nx, ny, nz), affine=affine)


def _qform(raw: bytes, end: str, dim) -> List[List[float]]:
    """Rebuild the affine from the quaternion form."""
    import math

    qb, qc, qd = struct.unpack(end + "3f", raw[256:268])
    qx, qy, qz = struct.unpack(end + "3f", raw[268:280])
    pix = struct.unpack(end + "8f", raw[76:108])
    qfac = pix[0] if pix[0] != 0 else 1.0
    dx, dy, dz = pix[1], pix[2], pix[3] * qfac

    w2 = 1.0 - (qb * qb + qc * qc + qd * qd)
    a = math.sqrt(w2) if w2 > 1e-7 else 0.0
    b, c, d = qb, qc, qd
    r = [
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
    ]
    s = (dx, dy, dz)
    return [
        [r[0][0] * s[0], r[0][1] * s[1], r[0][2] * s[2], qx],
        [r[1][0] * s[0], r[1][1] * s[1], r[1][2] * s[2], qy],
        [r[2][0] * s[0], r[2][1] * s[1], r[2][2] * s[2], qz],
        [0.0, 0.0, 0.0, 1.0],
    ]
