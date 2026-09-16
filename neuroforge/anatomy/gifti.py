"""
GIFTI (.gii) reader — standard library only.

WHY WRITE THIS RATHER THAN USE nibabel
--------------------------------------
nibabel is the obvious tool, but it requires numpy, and the whole point of
this project's dependency policy is that it runs on a bare Python install.
GIFTI is just XML with base64-encoded (optionally gzipped) binary arrays,
so a competent reader is about 120 lines. We only need three intents:
POINTSET (vertex coordinates), TRIANGLE (faces) and SHAPE/LABEL (per-vertex
scalars), which is a small enough subset to implement honestly.

WHAT THIS IS NOT
----------------
A complete GIFTI implementation. It ignores metadata we do not use, assumes
row-major ordering (which every file we consume uses) and supports only the
data types FreeSurfer's fsaverage actually emits.
"""

from __future__ import annotations

import array
import base64
import gzip
import sys
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# GIFTI datatype code -> (array module typecode, bytes per element)
_TYPES = {
    "NIFTI_TYPE_UINT8":   ("B", 1),
    "NIFTI_TYPE_INT16":   ("h", 2),
    "NIFTI_TYPE_INT32":   ("i", 4),
    "NIFTI_TYPE_UINT32":  ("I", 4),
    "NIFTI_TYPE_FLOAT32": ("f", 4),
    "NIFTI_TYPE_FLOAT64": ("d", 8),
}


@dataclass
class DataArray:
    intent: str
    data: Sequence[float]      # flat
    dims: List[int]            # e.g. [40962, 3]

    @property
    def rows(self) -> int:
        return self.dims[0] if self.dims else 0

    @property
    def cols(self) -> int:
        return self.dims[1] if len(self.dims) > 1 else 1


@dataclass
class Gifti:
    arrays: List[DataArray] = field(default_factory=list)
    # Key -> human-readable name, from <LabelTable>. Present in .label.gii.
    label_names: Dict[int, str] = field(default_factory=dict)

    def by_intent(self, intent: str) -> Optional[DataArray]:
        for a in self.arrays:
            if a.intent == intent:
                return a
        return None


def _decode(el: ET.Element) -> array.array:
    """Decode one <DataArray> payload into a flat array."""
    dtype = el.get("DataType", "NIFTI_TYPE_FLOAT32")
    if dtype not in _TYPES:
        raise ValueError(f"unsupported GIFTI DataType: {dtype}")
    tc, _size = _TYPES[dtype]

    enc = el.get("Encoding", "GZipBase64Binary")
    node = el.find("Data")
    raw_text = (node.text or "") if node is not None else ""

    if enc == "ASCII":
        vals = raw_text.split()
        conv = float if tc in "fd" else int
        return array.array(tc, [conv(v) for v in vals])

    blob = base64.b64decode(raw_text)
    if enc == "GZipBase64Binary":
        try:
            blob = gzip.decompress(blob)
        except OSError:
            # Some writers emit raw zlib rather than a gzip container.
            blob = zlib.decompress(blob)
    elif enc != "Base64Binary":
        raise ValueError(f"unsupported GIFTI Encoding: {enc}")

    out = array.array(tc)
    out.frombytes(blob)

    # GIFTI records its own endianness; swap if it differs from ours.
    endian = el.get("Endian", "LittleEndian")
    file_little = endian.startswith("Little")
    if file_little != (sys.byteorder == "little") and out.itemsize > 1:
        out.byteswap()
    return out


def read(path: str) -> Gifti:
    """Parse a .gii file."""
    root = ET.parse(path).getroot()
    g = Gifti()

    for lt in root.iter("LabelTable"):
        for lab in lt.iter("Label"):
            key = lab.get("Key", lab.get("Index"))
            if key is not None and lab.text:
                g.label_names[int(key)] = lab.text.strip()

    for el in root.iter("DataArray"):
        dims = []
        i = 0
        while (d := el.get(f"Dim{i}")) is not None:
            dims.append(int(d))
            i += 1
        if el.get("ArrayIndexingOrder", "RowMajorOrder") != "RowMajorOrder":
            raise ValueError("only RowMajorOrder GIFTI is supported")
        g.arrays.append(DataArray(
            intent=el.get("Intent", "NIFTI_INTENT_NONE"),
            data=_decode(el),
            dims=dims,
        ))
    return g


def read_surface(path: str):
    """Return (positions, indices) from a *.surf.gii.

    positions is a flat float array [x0,y0,z0, x1,...] in the file's own
    coordinate space (for fsaverage: FreeSurfer surface RAS, in millimetres,
    roughly centred on the middle of the brain).
    """
    g = read(path)
    pts = g.by_intent("NIFTI_INTENT_POINTSET")
    tri = g.by_intent("NIFTI_INTENT_TRIANGLE")
    if pts is None or tri is None:
        raise ValueError(f"{path} is not a surface file")
    return pts.data, tri.data


def read_scalars(path: str):
    """Return (values, label_names) from a *.shape.gii or *.label.gii."""
    g = read(path)
    for want in ("NIFTI_INTENT_LABEL", "NIFTI_INTENT_SHAPE",
                 "NIFTI_INTENT_NONE"):
        a = g.by_intent(want)
        if a is not None:
            return a.data, g.label_names
    raise ValueError(f"{path} has no scalar array")
