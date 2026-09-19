# NeuroForge

An interactive 3-D neuroscience laboratory. A rotatable, anatomically real
human brain is the centrepiece; an educational simulation of emotional
regulation, habit formation and neuroplasticity runs inside it.

Built with **zero pip dependencies** — a Python standard-library server, and
Three.js loaded from a CDN.

---

## What it actually is

A model of how **repeated behaviour** shifts the balance between competing
neural pathways, drawn onto a real cortical surface.

It is **not** a measurement of your brain, and the interface says so
continuously. Every panel separates two kinds of claim:

- **REAL** (green) — supported by published neuroscience
- **MODEL** (amber) — this program's invention, for teaching a concept

---

## Quick start

```powershell
python -m neuroforge.anatomy.fetch   # optional: real anatomy (~2.5 MB)
python -m neuroforge                 # serve and open the browser
```

Then open <http://127.0.0.1:8770/>.

Without the fetch step the app still runs, using a procedurally generated
cortex instead.

| Command | Effect |
|---|---|
| `python -m neuroforge` | Serve on port 8770 and open a browser |
| `python -m neuroforge --no-browser` | Serve only |
| `python -m neuroforge --rebuild` | Force a scene rebuild |
| `python -m neuroforge --cli` | Text-only prototype |
| `python -m unittest discover -s tests` | Run the test suite |

**Requirements:** Python 3.9+ and a WebGL2 browser. Nothing else.

---

## The three modes

**Anatomy** — explore structures. Click any region or subcortical structure
for what it is, where it is, what it contributes to, how this app models it,
and a caveat about what that model gets wrong.

**Circuits** — glowing pathways drawn between structures. Thickness and
brightness track simulated connection strength. These are *functional
relationships*, not tractography; several are polysynaptic and are relayed
through structures that are not drawn.

**Simulation** — log a trigger, choose a response, watch simulated pathway
strengths change. Scrub the timeline across day 1 / 7 / 30 / 90 / 180.

One brain throughout, with a **Current / Target / Compare** toggle.

---

## The model

Each connection carries three values:

- `w` — a labile weight that moves quickly
- `c` — a consolidated weight that moves slowly and resists decay
- `e` — an eligibility trace

Inputs converging on a shared target **compete**: strengthening one tends to
weaken its rivals. Critically, that competition is driven by **use, not by the
calendar**. An unwanted pathway does not fade simply because time passed.
Repetition is the whole mechanism.

Effects are deliberately small. One logged practice moves the model by about
**1.7%** of the remaining gap, and no two repetitions land the same way.

Twelve simulated weeks, 24 runs at each level:

| Adherence | Gap closed (mean) | Range across runs |
|---|---|---|
| 90 % | 69.6 % | 65.7 – 72.9 % |
| 70 % | 59.9 % | 51.9 – 73.1 % |
| 40 % | 28.0 % | 16.5 – 42.3 % |
| 10 % | **−17.1 %** | −37.3 – 4.4 % |

Two things that table is meant to show. Sporadic practice is not slow
progress, it is **backwards** — at 10 % adherence the slips outweigh the
reps. And identical adherence does not produce an identical result: attention,
motivation and context all vary, none of them are logged, and the spread at
70 % is wider than the gap between 70 % and 90 %.

### The target is not a "good brain"

The target threat level is **0.50, not zero**. The goal is regulated
responsiveness, not the abolition of fear. A person who cannot feel threat is
not healthy. The application will not let you aim at silence.

Likewise, this app rejects the "prefrontal cortex = good, amygdala = bad"
framing. Both are ordinary parts of an interacting system.

---

## Anatomy

By default the cortical surface is the FreeSurfer **fsaverage** template,
parcellated with the **Desikan-Killiany** atlas, with measured sulcal depth
driving the shading â€” 163,840 triangles of real anatomy. **25 cortical
regions** are labelled, and the **Yeo 2011** seven-network parcellation can be
overlaid on top of them.

Subcortical structures are extracted from the **FreeSurfer `aseg`
segmentation** of the MNI152 template — 12 of the 15 are real measured
geometry rather than placed shapes, read with a stdlib NIfTI parser and
surfaced by a pure-Python isosurface extractor.

Composite terms such as *prefrontal cortex* and *dorsal striatum* are shown
as collections of the regions they contain, rather than drawn as single
invented parcels. There is no boundary in the tissue where they begin.

fsaverage is an average of 40 brains. It is not yours.

The data is downloaded on first use into `neuroforge/_assets/` and is **not
included in this repository** (see [LICENSE](LICENSE)). Without it the app
still runs, but on a procedural cortex with 7 regions and no networks.

---

## Project layout

```
neuroforge/
  anatomy/      geometry, GIFTI + NIfTI readers, isosurface extraction,
                structure knowledge base
  atlas.py      13 regions, 20 connections, 7 pathways
  model.py      dual-rate plasticity + homeostatic competition
  events.py     16 loggable behaviours
  interpret.py  free text -> suggested events (classifier only)
  llm.py        retrieval + streaming chat, grounded in the above
  engine.py     simulation, history, replay, save/load
  server/       stdlib HTTP server + JSON API
  ui/           legacy Tkinter dashboard (superseded)
web/            Three.js renderer
scripts/        headless browser tests
tests/          27 tests
```

The simulation never imports the renderer. `anatomy/anchors.py` is the single
bridge mapping simulation node ids to 3-D coordinates.

---

## The assistant (optional)

There is a chat panel that explains what you are looking at and can highlight
regions in the 3-D view while it answers. It is **off by default** and needs
no account: point it at any OpenAI-compatible server and it works.

```powershell
$env:NEUROFORGE_LLM_BASE  = "http://127.0.0.1:11434/v1"   # Ollama
$env:NEUROFORGE_LLM_MODEL = "qwen3:8b"
python -m neuroforge
```

llama.cpp's `llama-server`, LM Studio and vLLM all expose the same shape; only
the port differs. With nothing configured the panel says so plainly and the
rest of the application is unaffected.

Two deliberate limits:

**It cannot change anything.** Free text becomes a *proposal* you confirm.
Unknown event ids are discarded rather than coerced, so a hallucination cannot
become a change in your record. There is no "log this for me" tool.

**It is given text, not trusted to recall.** Only the regions your question
mentions are retrieved (typically 300-600 tokens), and the model is instructed
to say it does not know rather than invent a mechanism. Replies tag claims as
REAL (established neuroscience) or MODEL (this app's representation).

---

## Scientific integrity

This is a teaching tool built on simplifications. Some worth stating plainly:

- **Neuroplasticity is real; this is not a measurement of yours.** Nothing
  here observes your brain.
- **"Pruning" is not what happens when you skip a rumination.** Synaptic
  pruning is a real, largely developmental, microglia-mediated process. It is
  not a daily consequence of your choices.
- **Extinction is not erasure.** New context-dependent learning inhibits an
  old association, which is why fears return under renewal, spontaneous
  recovery and reinstatement.
- **Behaviour does not identify a region.** Inferring which structure was
  active from an observed behaviour is the reverse-inference fallacy.
- **"21 days to form a habit" has no evidential basis.** Real estimates run to
  weeks or months and vary enormously between people and behaviours.

For contributors, [ENGINEERING.md](ENGINEERING.md) records the
architectural decisions, the coordinate conventions and the bugs already
fixed.

---

## Licence

MIT for the source code. Anatomical template data is downloaded at runtime
under its own terms and is never redistributed here. See [LICENSE](LICENSE).

## Development Note
This repository represents about 3 weeks of local development. The code was developed entirely offline and uploaded to GitHub in a single initial commit.
