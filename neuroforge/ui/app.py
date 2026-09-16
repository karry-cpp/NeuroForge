"""
neuroforge.ui.app
=================

STAGE 2 - the full interactive application (Tkinter, stdlib only).

Layout
------
  +--------------------------------------------------------------+------+
  |  header + permanent honesty strip                            |      |
  +--------------------------------------------------------------+ side |
  |   CURRENT BRAIN            |          TARGET BRAIN           | panel|
  |   (animated, clickable)    |          (animated)             |      |
  +--------------------------------------------------------------+      |
  |  TRIGGER -> THREAT -> OLD HABIT / REGULATED -> ... cascade    |      |
  +--------------------------------------------------------------+      |
  |  timeline scrubber + replay controls                          |      |
  +---------------------------------------------------------------------+

Run:  python -m neuroforge
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from .. import DISCLAIMER, __version__
from ..atlas import (EXCLUSIONS, GROUP_COLORS, GROUP_NOTES, PATHWAYS, REGIONS)
from ..engine import Simulation
from ..events import CATEGORIES, EVENTS_BY_ID, events_in
from ..model import Connectome
from .brainview import BG, BrainView, mix

PANEL = "#0c1220"
CARD = "#111a2d"
TXT = "#dbe7ff"
DIM = "#7d8db0"
ACCENT = "#4cc9f0"

CASCADE = [
    ("TRIGGER", "Something happens.", "#8ecae6"),
    ("SALIENCE", "Amygdala flags it as important / threatening.", "#ff6b6b"),
    ("APPRAISAL", "dACC: 'something is wrong'. Two routes open.", "#ffb703"),
    ("OLD ROUTE", "Rumination or autopilot - well-worn, low effort.",
     "#c77dff"),
    ("NEW ROUTE", "Notice → name → pause → let go. Effortful at first.",
     "#4cc9f0"),
    ("REPETITION", "Spaced practice + sleep consolidate what was used.",
     "#80ffdb"),
    ("EASIER ACCESS", "The regulated route becomes the default.", "#b8f2a6"),
]


class App(tk.Tk):
    def __init__(self, sim: Optional[Simulation] = None):
        super().__init__()
        self.title(f"NeuroForge {__version__} — an educational simulation")
        self.geometry("1500x940")
        self.minsize(1180, 780)
        self.configure(bg=BG)

        self.sim = sim or Simulation()
        self.display = Connectome()          # what the LEFT brain shows
        self.replay_day: Optional[int] = None
        self.playing = False
        self.t = 0.0
        self.active_stage = 0
        self._stage_decay = 0.0

        self._style()
        self._build()
        self.sim.subscribe(self._on_sim_change)
        self._on_sim_change()
        self._loop()

    # ------------------------------------------------------------- style
    def _style(self) -> None:
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=PANEL, foreground=TXT,
                    fieldbackground=CARD, bordercolor="#1c2740")
        s.configure("TNotebook", background=PANEL, borderwidth=0)
        s.configure("TNotebook.Tab", background="#0f1729", foreground=DIM,
                    padding=(14, 7), font=("Segoe UI", 9))
        s.map("TNotebook.Tab", background=[("selected", CARD)],
              foreground=[("selected", TXT)])
        s.configure("TFrame", background=PANEL)
        s.configure("TLabel", background=PANEL, foreground=TXT)
        s.configure("Card.TFrame", background=CARD)
        s.configure("Dim.TLabel", foreground=DIM, font=("Segoe UI", 8))
        s.configure("H.TLabel", font=("Segoe UI Semibold", 11))
        s.configure("TScale", background=PANEL)

    # ------------------------------------------------------------- build
    def _build(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True)

        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_header(left)
        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._build_cascade(left)
        self._build_timeline(left)

        side = tk.Frame(root, bg=PANEL, width=400)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self._build_side(side)

        self.cur_view = BrainView(self.canvas, "CURRENT BRAIN",
                                  "what repeated behaviour has built so far",
                                  (0, 0), (10, 10),
                                  on_region_click=self.show_region)
        self.tgt_view = BrainView(self.canvas, "TARGET BRAIN",
                                  "the direction you have chosen to practise",
                                  (0, 0), (10, 10),
                                  on_region_click=self.show_region)

    def _build_header(self, parent) -> None:
        h = tk.Frame(parent, bg=BG)
        h.pack(fill="x", padx=18, pady=(12, 0))
        tk.Label(h, text="NEUROFORGE", bg=BG, fg=TXT,
                 font=("Segoe UI Semibold", 16)).pack(side="left")
        tk.Label(h, text="  neuroplasticity · emotion regulation · habits  "
                         "— educational simulation",
                 bg=BG, fg=DIM, font=("Segoe UI", 9)).pack(side="left")
        self.hud = tk.Label(h, text="", bg=BG, fg=ACCENT,
                            font=("Consolas", 10))
        self.hud.pack(side="right")

        strip = tk.Frame(parent, bg="#141d33")
        strip.pack(fill="x", padx=18, pady=(8, 4))
        tk.Label(strip, text="⚠  Visual metaphor, not measurement.  Nodes = "
                             "simplified regions · lines = hypothetical "
                             "functional connections · thickness = a "
                             "simulation variable for learned-pattern "
                             "strength.  This app does not measure your "
                             "neurons, synapses, gray matter or brain "
                             "activity.",
                 bg="#141d33", fg="#9fb3d9", font=("Segoe UI", 8),
                 anchor="w", justify="left", wraplength=1000
                 ).pack(fill="x", padx=10, pady=4)

    # --------------------------------------------------------- cascade
    def _build_cascade(self, parent) -> None:
        self.casc = tk.Canvas(parent, bg=BG, height=78, highlightthickness=0)
        self.casc.pack(fill="x", padx=18)
        self.casc.bind("<Configure>", lambda e: self._draw_cascade())

    def _draw_cascade(self) -> None:
        c = self.casc
        c.delete("all")
        w = max(c.winfo_width(), 100)
        n = len(CASCADE)
        cw = w / n
        for i, (name, desc, col) in enumerate(CASCADE):
            x = i * cw
            hot = (i == self.active_stage and self._stage_decay > 0)
            glow = self._stage_decay if hot else 0.0
            c.create_rectangle(x + 4, 8, x + cw - 12, 66,
                               fill=mix("#0d1424", col, 0.06 + 0.30 * glow),
                               outline=mix("#1b2540", col, 0.2 + 0.8 * glow),
                               width=1)
            c.create_text(x + 16, 24, text=f"{i+1}. {name}", anchor="w",
                          fill=mix(col, "#ffffff", 0.25 * glow),
                          font=("Segoe UI Semibold", 9))
            c.create_text(x + 16, 46, text=desc, anchor="w", fill="#7d8db0",
                          font=("Segoe UI", 7), width=cw - 34)
            if i < n - 1:
                c.create_text(x + cw - 8, 37, text="›", fill="#33405e",
                              font=("Segoe UI", 14))

    def flash_stage(self, idx: int) -> None:
        self.active_stage = idx
        self._stage_decay = 1.0

    # -------------------------------------------------------- timeline
    def _build_timeline(self, parent) -> None:
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=18, pady=(6, 12))

        self.tl = tk.Canvas(f, bg="#0b1020", height=54,
                            highlightthickness=1,
                            highlightbackground="#1b2540")
        self.tl.pack(fill="x")
        self.tl.bind("<Button-1>", self._scrub)
        self.tl.bind("<B1-Motion>", self._scrub)
        self.tl.bind("<Configure>", lambda e: self._draw_timeline())

        b = tk.Frame(f, bg=BG)
        b.pack(fill="x", pady=(6, 0))
        for txt, cmd in (("▶ Replay evolution", self.toggle_play),
                         ("⏭ Next day", lambda: self.sim.advance_day()),
                         ("⏩ +1 week", lambda: self.sim.advance_day(7)),
                         ("⏩⏩ +1 month", lambda: self.sim.advance_day(30)),
                         ("↻ Live", self.go_live),
                         ("🎲 Demo 12 weeks", self.demo)):
            self._btn(b, txt, cmd).pack(side="left", padx=(0, 6))
        self._btn(b, "💾 Save", self.save).pack(side="right", padx=(6, 0))
        self._btn(b, "📂 Open", self.open).pack(side="right")

    def _btn(self, parent, text, cmd, color="#16203a"):
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg=TXT,
                      activebackground="#1e2c4d", activeforeground="#fff",
                      relief="flat", bd=0, padx=10, pady=5,
                      font=("Segoe UI", 9), cursor="hand2")
        return b

    def _draw_timeline(self) -> None:
        c = self.tl
        c.delete("all")
        W = max(c.winfo_width(), 50)
        H = 54
        days = max(1, self.sim.day)
        c.create_text(8, 10, text="TIMELINE — simulated progress "
                                  "(alignment to target)", anchor="w",
                      fill="#55648a", font=("Segoe UI", 7))
        # alignment curve
        pts = []
        for s in self.sim.history:
            x = 8 + (W - 16) * s.day / days
            y = H - 6 - (H - 24) * s.alignment
            pts += [x, y]
        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT, width=2, smooth=True)
        # event ticks
        for e in self.sim.log:
            cat = EVENTS_BY_ID[e.event_id].category
            col = {"Practice": "#b8f2a6", "Slip": "#ff6b6b",
                   "State": "#33405e"}[cat]
            x = 8 + (W - 16) * e.day / days
            c.create_line(x, H - 5, x, H - 11, fill=col, width=2)
        # playhead
        d = self.sim.day if self.replay_day is None else self.replay_day
        x = 8 + (W - 16) * d / days
        c.create_line(x, 6, x, H - 2, fill="#ffd166", width=1)
        c.create_text(min(W - 34, x + 5), 12,
                      text=f"d{d}", anchor="w", fill="#ffd166",
                      font=("Consolas", 8))

    def _scrub(self, ev) -> None:
        W = max(self.tl.winfo_width(), 50)
        frac = min(1.0, max(0.0, (ev.x - 8) / (W - 16)))
        self.replay_day = int(round(frac * self.sim.day))
        self.playing = False
        self._refresh_display()

    def toggle_play(self) -> None:
        if self.sim.day == 0:
            messagebox.showinfo("Nothing to replay",
                                "Log some events and advance a few days "
                                "first (or press 'Demo 12 weeks').")
            return
        self.playing = not self.playing
        if self.playing and self.replay_day is None:
            self.replay_day = 0

    def go_live(self) -> None:
        self.playing = False
        self.replay_day = None
        self._refresh_display()

    def demo(self) -> None:
        self.sim.simulate_scripted(weeks=12, adherence=0.72)
        self.go_live()

    # ------------------------------------------------------------ side
    def _build_side(self, side) -> None:
        nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.nb = nb

        self.tab_log = ttk.Frame(nb, style="TFrame")
        self.tab_prog = ttk.Frame(nb, style="TFrame")
        self.tab_info = ttk.Frame(nb, style="TFrame")
        self.tab_sci = ttk.Frame(nb, style="TFrame")
        nb.add(self.tab_log, text="Practice log")
        nb.add(self.tab_prog, text="Progress")
        nb.add(self.tab_info, text="Details")
        nb.add(self.tab_sci, text="Honesty")

        self._build_log_tab()
        self._build_progress_tab()
        self._build_info_tab()
        self._build_science_tab()

    # -- log tab ------------------------------------------------------
    def _build_log_tab(self) -> None:
        f = self.tab_log
        tk.Label(f, text="What just happened?", bg=PANEL, fg=TXT,
                 font=("Segoe UI Semibold", 11)).pack(anchor="w", padx=12,
                                                      pady=(10, 2))
        tk.Label(f, text="Each button applies an explicit, inspectable rule. "
                         "Right-click any button to see the rule and the "
                         "evidence behind it.",
                 bg=PANEL, fg=DIM, font=("Segoe UI", 8), wraplength=350,
                 justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x", padx=12)
        tk.Label(row, text="intensity", bg=PANEL, fg=DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        self.intensity = tk.DoubleVar(value=1.0)
        tk.Scale(row, from_=0.25, to=1.5, resolution=0.25, orient="horizontal",
                 variable=self.intensity, bg=PANEL, fg=DIM, troughcolor=CARD,
                 highlightthickness=0, bd=0, length=150,
                 font=("Segoe UI", 7)).pack(side="left", padx=6)

        tk.Label(f, text="note (optional)", bg=PANEL, fg=DIM,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=12)
        self.note = tk.Entry(f, bg=CARD, fg=TXT, insertbackground=TXT,
                             relief="flat", font=("Segoe UI", 9))
        self.note.pack(fill="x", padx=12, pady=(2, 10), ipady=4)

        colors = {"Practice": "#16341f", "Slip": "#331a1a",
                  "State": "#1a2338"}
        for cat in CATEGORIES:
            tk.Label(f, text=cat.upper(), bg=PANEL, fg="#55648a",
                     font=("Segoe UI Semibold", 8)).pack(anchor="w", padx=12,
                                                          pady=(6, 2))
            grid = tk.Frame(f, bg=PANEL)
            grid.pack(fill="x", padx=12)
            for i, ev in enumerate(events_in(cat)):
                b = self._btn(grid, f"{ev.icon} {ev.label}",
                              lambda e=ev.id: self.do_log(e),
                              color=colors[cat])
                b.configure(anchor="w", justify="left", width=22,
                            wraplength=150)
                b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
                b.bind("<Button-3>", lambda _e, k=ev.id: self.show_event(k))
                grid.columnconfigure(i % 2, weight=1)

        tk.Label(f, text="recent", bg=PANEL, fg="#55648a",
                 font=("Segoe UI Semibold", 8)).pack(anchor="w", padx=12,
                                                      pady=(10, 2))
        self.recent = tk.Text(f, height=6, bg="#0a0f1c", fg=DIM, bd=0,
                              relief="flat", font=("Consolas", 8),
                              highlightthickness=0)
        self.recent.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def do_log(self, event_id: str) -> None:
        self.go_live()
        ev = self.sim.log_event(event_id, self.note.get(),
                                self.intensity.get())
        self.note.delete(0, "end")
        stage = {"rumination": 3, "avoidance": 3, "reactive_outburst": 3,
                 "exposure": 4, "regulated_success": 6}.get(event_id)
        if stage is None:
            stage = 4 if ev.category == "Practice" else 1
        self.flash_stage(stage)
        self.show_event(event_id, switch=False)

    # -- progress tab -------------------------------------------------
    def _build_progress_tab(self) -> None:
        f = self.tab_prog
        self.gauge = tk.Canvas(f, height=150, bg=PANEL, highlightthickness=0)
        self.gauge.pack(fill="x", padx=12, pady=(12, 4))
        self.bars = tk.Canvas(f, bg=PANEL, highlightthickness=0)
        self.bars.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _draw_progress(self) -> None:
        cm = self.display
        from ..model import alignment as align_fn
        a = align_fn(cm)
        g = self.gauge
        g.delete("all")
        W = max(g.winfo_width(), 200)
        cx, cy, r = W / 2, 96, 72
        import math
        for k in range(0, 181, 3):
            th = math.radians(180 - k)
            on = (k / 180.0) <= a
            col = mix("#18223a", ACCENT, 1.0 if on else 0.0)
            x1, y1 = cx + r * math.cos(th), cy - r * math.sin(th)
            x2, y2 = cx + (r - 13) * math.cos(th), cy - (r - 13) * math.sin(th)
            g.create_line(x1, y1, x2, y2, fill=col, width=4)
        g.create_text(cx, cy - 18, text=f"{a*100:.1f}%", fill=TXT,
                      font=("Segoe UI Light", 26))
        g.create_text(cx, cy + 6, text="CURRENT → TARGET alignment", fill=DIM,
                      font=("Segoe UI", 8))
        g.create_text(cx, cy + 24,
                      text=f"gap closed since day 0: "
                           f"{self.sim.progress*100:.0f}%   ·   "
                           f"streak {self.sim.streak()}d",
                      fill="#8ecae6", font=("Segoe UI", 8))

        b = self.bars
        b.delete("all")
        W = max(b.winfo_width(), 200)
        y = 10
        b.create_text(4, y, text="PATHWAY STRENGTHS (simulated)", anchor="w",
                      fill="#55648a", font=("Segoe UI Semibold", 8))
        y += 18
        for pid, pw in PATHWAYS.items():
            cur = cm.pathway_strength(pid)
            arrow = "↑ grow" if pw.desirable else "↓ soften"
            b.create_text(4, y, text=pw.label, anchor="w", fill=TXT,
                          font=("Segoe UI", 9))
            b.create_text(W - 4, y, text=f"{cur:.2f} → {pw.target:.2f} "
                                         f"({arrow})", anchor="e",
                          fill=DIM, font=("Consolas", 8))
            y += 15
            b.create_rectangle(4, y, W - 4, y + 9, fill="#0a0f1c", outline="")
            b.create_rectangle(4, y, 4 + (W - 8) * cur, y + 9,
                               fill=pw.color, outline="")
            tx = 4 + (W - 8) * pw.target
            b.create_line(tx, y - 3, tx, y + 12, fill="#ffd166", width=2)
            y += 24
        y += 6
        b.create_text(4, y, anchor="nw", width=W - 8,
                      text="Yellow ticks are the target. Note that the threat "
                           "pathway's target is 0.50, not 0 — a calibrated "
                           "alarm system is the goal, not a silent one.",
                      fill="#55648a", font=("Segoe UI", 8))

    # -- details tab ---------------------------------------------------
    def _build_info_tab(self) -> None:
        self.info = tk.Text(self.tab_info, bg="#0a0f1c", fg=TXT, bd=0,
                            relief="flat", wrap="word", padx=14, pady=14,
                            font=("Segoe UI", 9), highlightthickness=0,
                            spacing1=2, spacing3=6)
        self.info.pack(fill="both", expand=True, padx=8, pady=8)
        for tag, cfg in {
            "h1": {"font": ("Segoe UI Semibold", 13), "foreground": "#ffffff"},
            "h2": {"font": ("Segoe UI Semibold", 9), "foreground": ACCENT,
                   "spacing1": 8},
            "real": {"foreground": "#b8f2a6"},
            "meta": {"foreground": "#ffb703"},
            "dim": {"foreground": DIM, "font": ("Segoe UI", 8)},
        }.items():
            self.info.tag_configure(tag, **cfg)
        self._set_info([("h1", "Click a brain region\n"),
                        ("dim", "…or right-click a log button, to see what "
                                "it represents, what the science actually "
                                "says, and what this app is not claiming.")])

    def _set_info(self, chunks) -> None:
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        for tag, text in chunks:
            self.info.insert("end", text, tag)
        self.info.configure(state="disabled")

    def show_region(self, rid: str) -> None:
        r = REGIONS[rid]
        conns = [c for c in self.display.conns
                 if rid in (c.spec.src, c.spec.dst)]
        lines = [("h1", f"{r.name}\n"),
                 ("dim", f"{r.group} · {GROUP_NOTES[r.group]}\n"),
                 ("h2", "\nIN ONE LINE\n"), (None, r.role + "\n"),
                 ("h2", "\nWHAT THE SCIENCE ACTUALLY SAYS  (real)\n"),
                 ("real", r.science + "\n"),
                 ("h2", "\nWHAT THIS APP IS NOT CLAIMING  (metaphor)\n"),
                 ("meta", r.caveat + "\n"),
                 ("h2", "\nSIMULATED CONNECTIONS\n")]
        for c in sorted(conns, key=lambda c: -c.w):
            d = "→" if c.spec.src == rid else "←"
            other = c.spec.dst if c.spec.src == rid else c.spec.src
            lines.append((None,
                          f"  {d} {other:<6} {c.spec.label:<30} "
                          f"w={c.w:.2f}  (consolidated {c.c*100:.0f}%)\n"))
        lines.append(("dim", "\n'w' is a program variable in [0,1]. It is not "
                             "a synapse count, a volume, or a signal."))
        self._set_info([(t, x) for t, x in lines])
        self.nb.select(self.tab_info)

    def show_event(self, eid: str, switch: bool = True) -> None:
        e = EVENTS_BY_ID[eid]
        self._set_info([
            ("h1", f"{e.icon}  {e.label}\n"),
            ("dim", f"category: {e.category}\n"),
            ("h2", "\nEXACT RULE APPLIED  (metaphor)\n"),
            ("meta", e.rule + "\n"),
            (None, f"\nglobal modifiers: learning rate ×{e.lr_mult:.2f}"
                   f" · reward {e.reward:+.2f} · stress "
                   f"{e.stress_delta:+.2f}\n"),
            ("dim", "…then scaled by intensity, by the inverted-U arousal "
                    "curve, by sleep quality, and by the spacing effect "
                    "(repeats on the same day count for less).\n"),
            ("h2", "\nWHAT THE SCIENCE ACTUALLY SAYS  (real)\n"),
            ("real", e.science + "\n"),
            ("h2", "\nWHAT THIS APP IS NOT CLAIMING  (metaphor)\n"),
            ("meta", (e.caveat or "—") + "\n"),
        ])
        if switch:
            self.nb.select(self.tab_info)

    # -- honesty tab ---------------------------------------------------
    def _build_science_tab(self) -> None:
        t = tk.Text(self.tab_sci, bg="#0a0f1c", fg=TXT, bd=0, wrap="word",
                    padx=14, pady=14, font=("Segoe UI", 9),
                    highlightthickness=0, spacing3=5)
        t.pack(fill="both", expand=True, padx=8, pady=8)
        t.tag_configure("h", font=("Segoe UI Semibold", 10),
                        foreground=ACCENT, spacing1=10)
        t.tag_configure("w", foreground="#ffb703")
        t.insert("end", "What this app is\n", "h")
        t.insert("end", DISCLAIMER + "\n")
        t.insert("end", "Neuroplasticity, stated accurately\n", "h")
        t.insert("end",
                 "Learning changes the nervous system, but mostly by changing "
                 "the *strength and reliability of existing connections* "
                 "(long-term potentiation and depression), by altering "
                 "receptor densities and excitability, and by shifting which "
                 "networks dominate a behaviour. Structural changes — new "
                 "dendritic spines, myelination changes, and measurable "
                 "volume differences — do occur, but typically over weeks to "
                 "months of substantial practice, and imaging effects at the "
                 "group level are small and often over-reported.\n")
        t.insert("end", "'Pruning' is a metaphor here\n", "h")
        t.insert("end",
                 "Synaptic pruning is a real biological process, most "
                 "dramatic in development and adolescence, and it is "
                 "activity-dependent and partly microglia-mediated. Choosing "
                 "not to ruminate today does not prune a synapse. In this "
                 "app, 'weakening' means: a route stops being reinforced and "
                 "loses a competition for a limited input budget at its "
                 "target node. Nothing is deleted — which is also why old "
                 "patterns can return under stress, poor sleep, or after a "
                 "long gap. The simulation models exactly that.\n", "w")
        t.insert("end", "What the model does NOT include\n", "h")
        for name, why in EXCLUSIONS:
            t.insert("end", f"• {name}: {why}\n")
        t.insert("end", "Directional honesty\n", "h")
        t.insert("end",
                 "The TARGET brain is not a picture of a 'healthy brain'. It "
                 "is your chosen direction of practice, written as numbers. "
                 "The threat pathway's target is deliberately non-zero. A "
                 "person with no threat response is not regulated; they are "
                 "in danger.\n", "w")
        t.insert("end", "Not a clinical tool\n", "h")
        t.insert("end",
                 "If you are dealing with trauma, panic, or persistent low "
                 "mood, exposure-style practice is best done with a "
                 "clinician. This program cannot tell whether a practice is "
                 "helping you.\n", "w")
        t.configure(state="disabled")

    # ------------------------------------------------------- lifecycle
    def _on_resize(self, ev) -> None:
        w, h = ev.width, ev.height
        self.cur_view.resize((0, 0), (w / 2, h))
        self.tgt_view.resize((w / 2, 0), (w / 2, h))
        self.canvas.delete("divider")
        self.canvas.create_line(w / 2, 20, w / 2, h - 20, fill="#16203a",
                                dash=(2, 6), tags="divider")

    def _on_sim_change(self) -> None:
        self._refresh_display()
        self.recent.configure(state="normal")
        self.recent.delete("1.0", "end")
        for e in reversed(self.sim.log[-40:]):
            ev = EVENTS_BY_ID[e.event_id]
            note = f"  “{e.note}”" if e.note else ""
            self.recent.insert("end",
                               f"d{e.day:<3} {ev.icon} {ev.label}{note}\n")
        self.recent.configure(state="disabled")

    def _refresh_display(self) -> None:
        if self.replay_day is None:
            self.display.load(self.sim.current.snapshot())
            for a, b in zip(self.display.conns, self.sim.current.conns):
                a.c, a.e = b.c, b.e
            label = f"day {self.sim.day} · live"
        else:
            snap = self.sim.snapshot_at(self.replay_day)
            self.display.load(snap.weights)
            label = f"day {snap.day} · REPLAY ({self.sim.date_for(snap.day)})"
        self.hud.configure(
            text=f"{label}   stress {self.sim.stress:.2f}   "
                 f"sleep {self.sim.sleep:.2f}   "
                 f"align {self.sim.alignment*100:.0f}%")
        self._draw_progress()
        self._draw_timeline()

    def _loop(self) -> None:
        self.t += 0.033
        if self._stage_decay > 0:
            self._stage_decay = max(0.0, self._stage_decay - 0.012)
            self._draw_cascade()
        if self.playing:
            self.replay_day = (self.replay_day or 0) + 1
            if self.replay_day >= self.sim.day:
                self.replay_day, self.playing = self.sim.day, False
            self._refresh_display()
        self.cur_view.render(self.display, self.t, compare=self.sim.target)
        self.tgt_view.render(self.sim.target, self.t)
        self.after(33, self._loop)

    # ------------------------------------------------------ persistence
    def save(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("NeuroForge", "*.json")])
        if p:
            self.sim.save(p)

    def open(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("NeuroForge", "*.json")])
        if not p:
            return
        self.sim = Simulation.load(p)
        self.sim.subscribe(self._on_sim_change)
        self.go_live()
        self._on_sim_change()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
