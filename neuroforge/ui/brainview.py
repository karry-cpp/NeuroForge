"""
neuroforge.ui.brainview
=======================

The animated brain renderer. Pure Tkinter Canvas, no dependencies.

Rendering contract (deliberately explicit, because this is the part that
could mislead someone):
  * A NODE is a labelled disc standing for a simplified brain region.
    Its position is schematic, not anatomical.
  * An EDGE is a curved line standing for a *hypothetical functional
    connection*. Its thickness and brightness encode the simulation
    variable `w`, i.e. how strong a learned behavioural pattern is
    **in this program**.
  * The travelling dots are an animation of "this route is easy to run".
    They are not action potentials. Nothing here is measured.

The class exposes a small API so the render backend can later be swapped
for Qt/QGraphicsView or moderngl without touching the engine.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Dict, List, Optional, Tuple

from ..atlas import EDGES, PATHWAYS, REGIONS
from ..model import Connectome

BG = "#080b14"
GRID = "#111726"


# ----------------------------------------------------------------- colours
def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % (max(0, min(255, int(r))),
                              max(0, min(255, int(g))),
                              max(0, min(255, int(b))))


def mix(c1: str, c2: str, t: float) -> str:
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(*[a[i] + (b[i] - a[i]) * t for i in range(3)])


# ------------------------------------------------------------------ maths
def bezier(p0, p1, p2, n: int = 18) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        pts.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return pts


def point_on(pts, s: float) -> Tuple[float, float]:
    s = s % 1.0
    i = min(len(pts) - 2, int(s * (len(pts) - 1)))
    f = s * (len(pts) - 1) - i
    x0, y0 = pts[i]
    x1, y1 = pts[i + 1]
    return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


# ------------------------------------------------------------------- view
class BrainView:
    """Draws one brain (current or target) onto a region of a canvas."""

    def __init__(self, canvas: tk.Canvas, title: str, subtitle: str,
                 origin: Tuple[float, float], size: Tuple[float, float],
                 animate: bool = True,
                 on_region_click: Optional[Callable[[str], None]] = None):
        self.cv = canvas
        self.title = title
        self.subtitle = subtitle
        self.ox, self.oy = origin
        self.w, self.h = size
        self.animate = animate
        self.on_region_click = on_region_click
        self.tag = f"bv{id(self)}"
        self.pos: Dict[str, Tuple[float, float]] = {}
        self.paths: Dict[str, List[Tuple[float, float]]] = {}
        self.edge_items: Dict[str, List[int]] = {}
        self.pulse_items: Dict[str, List[int]] = {}
        self.node_items: Dict[str, Tuple[int, int, int]] = {}
        self.highlight: Optional[str] = None
        self.dim_others: bool = False
        self._built = False

    # ------------------------------------------------------------ layout
    def _px(self, rx: float, ry: float) -> Tuple[float, float]:
        pad = 0.09
        return (self.ox + self.w * (pad + rx * (1 - 2 * pad)),
                self.oy + 56 + (self.h - 86) * (pad + ry * (1 - 2 * pad)))

    def resize(self, origin, size) -> None:
        self.ox, self.oy = origin
        self.w, self.h = size
        self.cv.delete(self.tag)
        self._built = False

    # ------------------------------------------------------------- build
    def build(self) -> None:
        cv, T = self.cv, self.tag
        cv.delete(T)
        self.pos = {rid: self._px(r.x, r.y) for rid, r in REGIONS.items()}
        self.paths.clear()
        self.edge_items.clear()
        self.pulse_items.clear()
        self.node_items.clear()

        self._silhouette()

        # header
        cv.create_text(self.ox + self.w / 2, self.oy + 20, text=self.title,
                       fill="#e8f1ff", font=("Segoe UI Semibold", 15),
                       tags=(T,))
        cv.create_text(self.ox + self.w / 2, self.oy + 40, text=self.subtitle,
                       fill="#6d7f9e", font=("Segoe UI", 9), tags=(T,))

        # edges (3 stacked lines = cheap glow)
        for spec in EDGES:
            key = f"{spec.src}->{spec.dst}"
            p0 = self.pos[spec.src]
            p2 = self.pos[spec.dst]
            mx, my = (p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2
            dx, dy = p2[0] - p0[0], p2[1] - p0[1]
            p1 = (mx - dy * spec.curve, my + dx * spec.curve)
            pts = bezier(p0, p1, p2)
            self.paths[key] = pts
            flat = [c for pt in pts for c in pt]
            layers = []
            for wmul, stipple in ((5.0, "gray25"), (2.4, ""), (1.0, "")):
                layers.append(cv.create_line(
                    *flat, fill=GRID, width=1, smooth=True, capstyle="round",
                    stipple=stipple, tags=(T, "edge", key)))
            self.edge_items[key] = layers
            self.pulse_items[key] = [
                cv.create_oval(0, 0, 0, 0, fill="", outline="", tags=(T,))
                for _ in range(3)]

        # nodes on top
        for rid, r in REGIONS.items():
            x, y = self.pos[rid]
            rad = r.radius * min(self.w / 520, 1.15)
            halo = cv.create_oval(x - rad * 1.8, y - rad * 1.8,
                                  x + rad * 1.8, y + rad * 1.8,
                                  fill="", outline="", tags=(T,))
            disc = cv.create_oval(x - rad, y - rad, x + rad, y + rad,
                                  fill="#0e1526", outline=r.color, width=2,
                                  tags=(T, "node", f"node:{rid}"))
            lbl = cv.create_text(x, y, text=rid, fill="#dbe7ff",
                                 font=("Segoe UI Semibold", 9),
                                 tags=(T, "node", f"node:{rid}"))
            self.node_items[rid] = (halo, disc, lbl)
            if self.on_region_click:
                cv.tag_bind(f"node:{rid}", "<Button-1>",
                            lambda _e, k=rid: self.on_region_click(k))
                cv.tag_bind(f"node:{rid}", "<Enter>",
                            lambda _e, k=rid: self.set_highlight(k))
                cv.tag_bind(f"node:{rid}", "<Leave>",
                            lambda _e: self.set_highlight(None))
        self._built = True

    def _silhouette(self) -> None:
        """A soft brain-shaped backdrop so it reads as a brain, not a graph."""
        cx = self.ox + self.w / 2
        cy = self.oy + 56 + (self.h - 86) / 2
        rx, ry = self.w * 0.46, (self.h - 86) * 0.47
        for i, (k, col) in enumerate(((1.00, "#0d1322"), (0.93, "#0b1120"),
                                      (0.86, "#0a0f1c"))):
            pts = []
            for a in range(0, 360, 6):
                t = math.radians(a)
                # gentle brain-ish lobe distortion
                wob = (1 + 0.10 * math.sin(3 * t + 0.6)
                       + 0.05 * math.sin(5 * t))
                pts += [cx + rx * k * wob * math.cos(t),
                        cy + ry * k * wob * math.sin(t) * 0.94]
            self.cv.create_polygon(*pts, fill=col, outline="", smooth=True,
                                   tags=(self.tag,))

    # ------------------------------------------------------------ update
    def set_highlight(self, rid: Optional[str]) -> None:
        self.highlight = rid

    def render(self, cm: Connectome, t: float,
               compare: Optional[Connectome] = None) -> None:
        """Draw the given connectome state. `t` is animation time in seconds.

        If `compare` is given, edges that differ a lot from it are outlined -
        used to show 'here is where CURRENT still differs from TARGET'.
        """
        if not self._built:
            self.build()
        cmp_w = {f"{c.spec.src}->{c.spec.dst}": c.w
                 for c in compare.conns} if compare else None

        for c in cm.conns:
            key = f"{c.spec.src}->{c.spec.dst}"
            pw = PATHWAYS[c.pathway]
            w = c.w
            focus = (self.highlight is None
                     or self.highlight in (c.spec.src, c.spec.dst))
            fade = 1.0 if focus else 0.22

            base = mix("#16203a", pw.color, min(1.0, 0.18 + w * 1.05))
            col = mix(BG, base, fade)
            glow = mix(BG, pw.color, fade * (0.10 + 0.42 * w))
            core = mix(base, "#ffffff", 0.10 + 0.35 * w)

            widths = (1.5 + 13.0 * w ** 1.35, 1.0 + 6.5 * w ** 1.3,
                      0.8 + 2.6 * w)
            cols = (glow, col, mix(BG, core, fade))
            for item, wd, cl in zip(self.edge_items[key], widths, cols):
                self.cv.itemconfig(item, width=max(1, wd), fill=cl)

            # divergence marker vs. the target brain
            if cmp_w is not None and abs(cmp_w[key] - w) > 0.28 and focus:
                self.cv.itemconfig(self.edge_items[key][0],
                                   fill=mix(glow, "#ff9f1c", 0.35))

            # travelling pulses: rate and count scale with strength
            pts = self.paths[key]
            n_active = 0 if w < 0.12 else 1 if w < 0.45 else 2 if w < 0.75 else 3
            speed = 0.09 + 0.34 * w
            for i, item in enumerate(self.pulse_items[key]):
                if not self.animate or i >= n_active or not focus:
                    self.cv.itemconfig(item, fill="", outline="")
                    continue
                s = (t * speed + i / max(1, n_active)
                     + hash(key) % 100 / 100.0) % 1.0
                x, y = point_on(pts, s)
                rr = 1.6 + 3.2 * w
                self.cv.coords(item, x - rr, y - rr, x + rr, y + rr)
                self.cv.itemconfig(item, fill=mix(pw.color, "#ffffff", 0.45),
                                   outline="")
                self.cv.tag_raise(item)

        # node halos breathe with how much traffic they carry
        load: Dict[str, float] = {}
        for c in cm.conns:
            load[c.spec.dst] = load.get(c.spec.dst, 0) + c.w
            load[c.spec.src] = load.get(c.spec.src, 0) + c.w * 0.5
        for rid, (halo, disc, lbl) in self.node_items.items():
            r = REGIONS[rid]
            v = min(1.0, load.get(rid, 0) / 2.6)
            pulse = 0.5 + 0.5 * math.sin(t * 1.6 + hash(rid) % 7)
            hot = self.highlight == rid
            self.cv.itemconfig(
                disc, outline=mix(r.color, "#ffffff", 0.5 if hot else 0.0),
                width=2 + 3.2 * v + (1.6 if hot else 0))
            self.cv.itemconfig(
                halo, fill="", outline=mix(BG, r.color,
                                           0.10 + 0.22 * v * pulse), width=6)
            self.cv.tag_raise(disc)
            self.cv.tag_raise(lbl)
