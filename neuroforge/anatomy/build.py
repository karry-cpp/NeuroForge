"""
neuroforge.anatomy.build
========================

Assembles the full anatomical scene and caches it to disk in a compact
wire format (base64 float32/uint32 buffers inside JSON).

Generation takes a while (procedural folding over ~40k vertices), so the
result is cached. Delete the cache file to force a rebuild.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
from typing import Any, Dict, List, Tuple

from . import cortex_real, fetch, subcortex_real, tracts
from .anchors import ANCHORS, NODE_TO_CORTEX, NODE_TO_STRUCTURE
from .mesh import (CORTICAL_LABELS, Mesh, build_cortex, merge, mirror_x)
from .structures import (CORTICAL_REGIONS, GEOMETRY_NOTE, GROUPS,
                         NETWORK_CAVEAT,
                         NETWORK_INFO, NETWORK_NOTE, STRUCTURES)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_cache")
# Bump whenever geometry changes, or a stale cache will be served.
# v5: real fsaverage cortex + Desikan-Killiany parcellation + sulcal depth.
# v6: self-referential regions (PCC, precuneus, TPJ, frontopolar) split out
#     of "other"/"parietal", plus Yeo 2011 functional networks.
# v7: subcortical structures extracted from the FreeSurfer aseg segmentation
#     instead of being procedural blobs, plus ventricles and ventral DC.
# v8: insula (anterior/posterior) and medial PFC added; dACC and sgACC given
#     cortical region entries; composite groupings published.
# v9: chat/assistant grounding.
# v10: measured per-hemisphere centroids published for cortical regions, so
#      the sidebar can aim the camera at a surface parcel the way it already
#      does at a subcortical structure.
# v11: pathway endpoints taken from measured geometry instead of a hand-typed
#      table, and the curves between them routed through white matter.
SCENE_VERSION = 11


# --------------------------------------------------------------------------
#  Wire encoding
# --------------------------------------------------------------------------

def _f32(values: List[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _u32(values: List[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}I", *values)).decode()


def _u8(values: List[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}B", *values)).decode()


def _u16(values: List[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}H", *values)).decode()


def encode_mesh(m: Mesh, with_labels: bool = False) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "position": _f32(m.positions),
        "normal": _f32(m.normals),
        "index": _u32(m.indices),
        "count": m.vertex_count,
        "tris": len(m.indices) // 3,
    }
    if with_labels and m.labels:
        d["label"] = _u8(m.labels)
    if m.sulc:
        d["sulc"] = _f32(m.sulc)
    if m.fine:
        d["fine"] = _u16(m.fine)
    if m.network:
        d["network"] = _u8(m.network)
    return d


# --------------------------------------------------------------------------
#  Hemisphere splitting
# --------------------------------------------------------------------------

def split_hemispheres(m: Mesh) -> Tuple[Mesh, Mesh]:
    """Split the cortical mesh into left and right by triangle centroid.

    Lets the UI hide one hemisphere to expose the medial surface, where the
    cingulate, vmPFC and the subcortical structures live.
    """
    out: Dict[int, Mesh] = {0: Mesh(), 1: Mesh()}
    remap: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
    P, N, L = m.positions, m.normals, m.labels

    for i in range(0, len(m.indices), 3):
        tri = (m.indices[i], m.indices[i + 1], m.indices[i + 2])
        cx = sum(P[v * 3] for v in tri) / 3.0
        side = 1 if cx >= 0 else 0          # 1 = right, 0 = left
        dst, rm = out[side], remap[side]
        for v in tri:
            nv = rm.get(v)
            if nv is None:
                nv = dst.vertex_count
                rm[v] = nv
                dst.positions.extend(P[v * 3:v * 3 + 3])
                dst.normals.extend(N[v * 3:v * 3 + 3])
                dst.labels.append(L[v] if L else 0)
                if m.sulc:
                    dst.sulc.append(m.sulc[v])
                if m.fine:
                    dst.fine.append(m.fine[v])
                if m.network:
                    dst.network.append(m.network[v])
            dst.indices.append(nv)
    return out[0], out[1]


# --------------------------------------------------------------------------
#  Scene assembly
# --------------------------------------------------------------------------

def build_scene(subdivisions: int = 6, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()

    # Real anatomy is strongly preferred: the folds, the region boundaries
    # and the sulcal depth are all measured rather than invented. The
    # procedural cortex remains a fallback so the app still runs offline.
    if cortex_real.available():
        if verbose:
            print("[anatomy] loading real cortex (fsaverage)…", flush=True)
        left, right = cortex_real.build_cortex_real()
        geometry_source = "fsaverage"
        n_verts = left.vertex_count + right.vertex_count
        n_tris = (len(left.indices) + len(right.indices)) // 3
    else:
        if verbose:
            print(f"[anatomy] no fsaverage assets; generating procedural "
                  f"cortex (subdiv={subdivisions})…", flush=True)
        cortex = build_cortex(subdivisions=subdivisions)
        left, right = split_hemispheres(cortex)
        geometry_source = "procedural"
        n_verts, n_tris = cortex.vertex_count, len(cortex.indices) // 3

    if verbose:
        print(f"[anatomy]   cortex {n_verts} verts / {n_tris} tris "
              f"[{geometry_source}] ({time.time()-t0:.1f}s)", flush=True)

    meshes: Dict[str, Any] = {
        "cortex_L": encode_mesh(left, with_labels=True),
        "cortex_R": encode_mesh(right, with_labels=True),
    }

    structures: List[Dict[str, Any]] = []
    use_aseg = subcortex_real.available()
    for s in STRUCTURES:
        geo = None
        source = "procedural"
        if use_aseg and s.aseg:
            # Real segmentation. These meshes already cover both sides, so
            # they must not go through the mirror path below.
            geo = subcortex_real.build(s.aseg)
            if geo is not None:
                source = "aseg"
                centroids = subcortex_real.centroids(s.aseg) or [list(s.centroid)]

        if geo is None:
            base = s.builder()
            if not base.indices:
                # A structure with no procedural fallback, and no atlas to
                # build it from. Omit it rather than ship an empty mesh.
                continue
            if s.bilateral:
                geo = merge(base, mirror_x(base))
                centroids = [list(s.centroid),
                             [-s.centroid[0], s.centroid[1], s.centroid[2]]]
            else:
                geo = base
                centroids = [list(s.centroid)]

        meshes[f"struct_{s.id}"] = encode_mesh(geo)
        structures.append({
            "id": s.id, "name": s.name, "short": s.short,
            "system": s.system, "color": s.color, "opacity": s.opacity,
            "order": s.order, "bilateral": s.bilateral,
            "geometry_source": source,
            "centroids": centroids,
            "knowledge": {
                "what_it_is": s.what_it_is,
                "where_it_is": s.where_it_is,
                "contributes_to": s.contributes_to,
                "in_the_model": s.in_the_model,
                "caveat": s.caveat,
            },
        })
        if verbose:
            print(f"[anatomy]   {s.id:<12} {geo.vertex_count:>6} verts",
                  flush=True)

    # Only publish regions that actually exist on the surface that was
    # built. The procedural fallback cortex assigns a much coarser set of
    # labels than the real parcellation does, so advertising every region
    # regardless would give the offline user a list of regions that open a
    # panel and then highlight nothing at all.
    present = set()
    for m in (left, right):
        if m.labels:
            present.update(m.labels)

    # Mean vertex position per label, kept per hemisphere. Averaging the two
    # sides together would put every bilateral region on the midline, which
    # is the one place a lateral region is not.
    acc: Dict[int, Dict[int, List[float]]] = {}
    for side, m in ((0, left), (1, right)):
        if not m.labels:
            continue
        P = m.positions
        for v, li in enumerate(m.labels):
            a = acc.setdefault(li, {}).setdefault(side, [0.0, 0.0, 0.0, 0.0])
            a[0] += P[v * 3]
            a[1] += P[v * 3 + 1]
            a[2] += P[v * 3 + 2]
            a[3] += 1.0

    def _centroids(li: int) -> List[List[float]]:
        # Right side first, to match the ordering the structure list uses.
        out = []
        for side in (1, 0):
            a = acc.get(li, {}).get(side)
            if a and a[3]:
                out.append([round(a[i] / a[3], 2) for i in range(3)])
        return out

    cortical = [{
        "id": r.id, "label_index": r.label_index, "name": r.name,
        "short": r.short, "color": r.color, "buried": r.buried,
        "centroids": _centroids(r.label_index),
        "knowledge": {
            "what_it_is": r.what_it_is, "where_it_is": r.where_it_is,
            "contributes_to": r.contributes_to,
            "in_the_model": r.in_the_model, "caveat": r.caveat,
        },
    } for r in CORTICAL_REGIONS if r.label_index in present]

    # Groupings are filtered the same way as regions: a group is only
    # offered if every part of it was actually built. On the procedural
    # fallback "prefrontal cortex" would otherwise promise six regions and
    # highlight three.
    have_regions = {r["id"] for r in cortical}
    have_structs = {s["id"] for s in structures}
    groups = [g for g in GROUPS
              if all(x in have_regions for x in g["regions"])
              and all(x in have_structs for x in g["structures"])]

    anchors, paths = _pathways(structures, cortical, verbose)

    scene = {
        "version": SCENE_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
        "meshes": meshes,
        "structures": structures,
        "cortical_regions": cortical,
        "groups": groups,
        "cortical_labels": CORTICAL_LABELS,
        "geometry_source": geometry_source,
        # Destrieux parcel id -> [anatomical name, "gyrus"|"sulcus"|"both"].
        # Empty when running on the procedural fallback, which has no
        # anatomical parcels to name.
        "fine_names": {str(k): list(v)
                       for k, v in getattr(left, "fine_names", {}).items()},
        # Yeo 2011 functional networks - the only honest way to show the
        # default mode network, which spans several anatomical regions.
        "networks": [
            {"id": i, "name": cortex_real.YEO7_NAMES[i],
             "color": cortex_real.YEO7_COLORS[i],
             **NETWORK_INFO.get(cortex_real.YEO7_NAMES[i], {})}
            for i in sorted(cortex_real.YEO7_NAMES)
        ] if left.network else [],
        "network_caveat": NETWORK_CAVEAT,
        # Pathway endpoints measured from the built geometry, and the routes
        # between them. Both are empty on the procedural fallback, where the
        # renderer keeps using its own synthetic arcs.
        "anchors_measured": anchors,
        "edge_paths": paths,
        "notes": {
            "geometry": (fetch.NOTICE if geometry_source == "fsaverage"
                         else GEOMETRY_NOTE),
            "network": NETWORK_NOTE,
        },
    }
    if verbose:
        print(f"[anatomy] done in {time.time()-t0:.1f}s", flush=True)
    return scene


def _pathways(structures: List[Dict[str, Any]],
              cortical: List[Dict[str, Any]],
              verbose: bool) -> Tuple[Dict[str, List[float]],
                                      Dict[str, List[List[float]]]]:
    """
    Where each simulation node sits, and how to get between them.

    anchors.py is the seam between the model and the geometry, and it already
    names every simulation node, so resolving those names against measured
    centroids belongs on this side of it. The simulation still knows nothing
    about millimetres.

    The hand-typed table in anchors.py stays as the fallback: it is the only
    thing available when the anatomy has not been downloaded, and a few nodes
    (LC, VTA) have no aseg label to measure.
    """
    # Lazy: the simulation's edge list is only needed to know which pairs to
    # route, and importing it at module scope would make the anatomy package
    # import the model on every build.
    from ..atlas import EDGES

    by_struct = {s["id"]: s for s in structures}
    by_cortex = {c["id"]: c for c in cortical}

    anchors: Dict[str, List[float]] = {}
    for node, fallback in ANCHORS.items():
        found = None
        sid = NODE_TO_STRUCTURE.get(node)
        if sid and by_struct.get(sid, {}).get("centroids"):
            found = by_struct[sid]["centroids"][0]
        if found is None:
            cid = NODE_TO_CORTEX.get(node)
            if cid and by_cortex.get(cid, {}).get("centroids"):
                found = by_cortex[cid]["centroids"][0]
        anchors[node] = [round(v, 2) for v in (found or list(fallback))]

    paths: Dict[str, List[List[float]]] = {}
    if not tracts.available():
        return anchors, paths

    t0 = time.time()
    for e in EDGES:
        a, b = anchors.get(e.src), anchors.get(e.dst)
        if not a or not b:
            continue
        p = tracts.route(tuple(a), tuple(b))
        if p:
            paths[f"{e.src}->{e.dst}"] = [[round(c, 2) for c in q] for q in p]
    if verbose:
        print(f"[anatomy]   routed {len(paths)}/{len(EDGES)} pathways "
              f"through white matter ({time.time()-t0:.1f}s)", flush=True)
    return anchors, paths


# --------------------------------------------------------------------------
#  Cache
# --------------------------------------------------------------------------

def cache_path(subdivisions: int) -> str:
    """Where the built scene is cached.

    The geometry source is part of the key. Without it, a user who ran the
    app before downloading the anatomical assets would keep being served the
    cached procedural brain forever, and `fetch` would appear to do nothing.
    """
    src = "fs" if cortex_real.available() else "proc"
    return os.path.join(
        CACHE_DIR, f"scene_v{SCENE_VERSION}_{src}_s{subdivisions}.json")


def get_scene(subdivisions: int = 6, rebuild: bool = False,
              verbose: bool = True) -> Dict[str, Any]:
    p = cache_path(subdivisions)
    if not rebuild and os.path.exists(p):
        if verbose:
            mb = os.path.getsize(p) / 1e6
            print(f"[anatomy] cache hit: {os.path.basename(p)} ({mb:.1f} MB)",
                  flush=True)
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    scene = build_scene(subdivisions, verbose)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(scene, f, separators=(",", ":"))
    if verbose:
        print(f"[anatomy] cached -> {p} "
              f"({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
    return scene


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the anatomical scene.")
    ap.add_argument("--subdiv", type=int, default=6,
                    help="icosphere subdivisions (5=fast, 6=detailed, 7=heavy)")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    get_scene(a.subdiv, rebuild=a.rebuild)
