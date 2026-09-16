# NeuroForge — engineering brief

**Read this first, and in full, before changing anything in this repository.**
It records not just *what* the code does but *why it is the way it is*, and
which decisions are deliberate and must not be "helpfully" reverted.

---

## 1. What this application is

An interactive 3-D neuroscience laboratory. A large, rotatable, anatomically
real human brain is the centrepiece; an educational simulation of emotional
regulation, habit formation and neuroplasticity lives *inside* it.

The framing that governs every design decision:

> A beautiful interactive 3-D brain that happens to contain an educational
> simulation — **not** a dashboard with a brain diagram in the corner.

If a change would make the brain smaller, flatter, or less central, it is
the wrong change.

### Explicitly rejected designs
- Flat brain silhouettes; circles standing in for regions.
- A network graph pretending to be a brain.
- Two brains side by side. There is **ONE** brain with a
  Current / Target / Compare toggle.

---

## 2. Non-negotiable constraints

| Constraint | Why |
|---|---|
| **Insula is included** (reversed 2024) | Originally excluded by user instruction. The user later asked for the anterior insula by name. It is now split into `aINS` (label 24, `y >= 0`) and `pINS` (label 25, `y < 0`). Interoception cannot be described honestly without it. Do not re-exclude it. |
| **Target threat level is 0.50, not 0** | The goal is regulated responsiveness, not the abolition of fear. A person who cannot feel threat is not healthy. The UI states: *"The app will not let you aim at silence."* |
| **"Simulated pathway strength"** | Never "your neurons", never "your brain changed". |
| **REAL (green) vs MODEL (amber) tags** | Every panel separates established science from this program's invention. Preserve this in any new UI. |
| **Zero pip dependencies** | Runs on a bare Python install. stdlib only. |
| **Anatomical data is never committed** | See `LICENSE`. `neuroforge/_assets/` is gitignored. |

### Forbidden phrases (user-specified)
- "I just pruned your neurons"
- "Your PFC gained gray matter today"
- "This visualization measures your real brain"

---

## 3. Architecture and why

**Python backend + browser renderer, communicating over JSON.**

The user required that the simulation engine be independent of the visual
renderer. A **process boundary** is the strongest possible enforcement of
that: the physics literally cannot reach into the renderer.

### Why not a pure-Python 3-D stack
The machine runs **Python 3.14**, which at the time of writing has no wheels
for VTK, PySide6 or moderngl. Beyond availability:

| Option | Verdict |
|---|---|
| VTK / PyVista | Best scientific 3-D in Python, but heavy native wheels and dated lighting. |
| PySide6 + QOpenGL | Means hand-writing shaders, camera, picking, tone-mapping. Months of work. |
| Plotly | A charting library. Cannot do volumetric glow. |
| **Three.js** | PBR materials, bloom, mature orbit controls, GPU picking, zero native deps. |

The requirement — *futuristic neuroscience laboratory, elegant 3-D lighting,
subtle glow* — is a **real-time rendering problem, not a scientific-plotting
problem**. Three.js r0.169.0 is loaded from a CDN via an importmap.

### Layers
```
anatomy/     geometry + the neuroscience knowledge layer   (Python)
atlas,model,events,engine   the simulation                  (Python, renderer-agnostic)
anchors.py   THE ONLY BRIDGE between node ids and 3-D coordinates
server/      stdlib http.server + JSON API
web/         Three.js renderer — knows nothing about the model
```

`anchors.py` is the seam. If you need geometry in the simulation, or model
state in the geometry, you are about to break the separation — route it
through the API instead.

---

## 4. Coordinate system (a recurring source of bugs)

- Anatomy is in **RAS millimetres**: +x right, +y anterior, +z superior. Z-up.
- Three.js is **Y-up**.
- Reconciled by exactly **one** line: `root.rotation.x = -Math.PI / 2`
  in `viewer.js`. Mapping is RAS (x,y,z) → world (x, z, −y).

Consequences that have already bitten once each:
- `circuits._curve` must use **raw mm**; the curves are children of `root`,
  so converting again double-applies the rotation.
- Camera presets for anterior/posterior were initially swapped.
- Screen-space pins must project through `root.matrixWorld`.

---

## 5. Anatomy: real, with a procedural fallback

**Preferred path** — `anatomy/cortex_real.py`:
- FreeSurfer **fsaverage** pial surface (`den-41k`, 40,962 verts/hemisphere).
- **Desikan-Killiany** atlas for parcellation — real anatomical boundaries.
- Measured **sulcal depth** drives shading (`aSulc` attribute, `uHasSulc`).
- 163,840 triangles total, loads in ~0.6 s.
- Fetch with `python -m neuroforge.anatomy.fetch`.

**Fallback** — `anatomy/mesh.py`: signed-distance-field cerebrum with ridged
multifractal noise for folds. Used when assets are absent. ~7.3 s to build.

`anatomy/gifti.py` is a ~120-line stdlib GIFTI reader (XML + base64/gzip),
written to avoid a numpy dependency.

### Two honest approximations in the atlas mapping
1. Desikan-Killiany has **no subgenual ACC parcel**. We split
   `rostralanteriorcingulate` at z = 0 as a geometric proxy.
2. `superiorfrontal` spans both the dorsolateral convexity and the medial
   wall; assigning it wholly to dlPFC under-represents medial PFC.

Both are documented in `DK_TO_LABEL` so they can be argued with.

> fsaverage is an average of 40 brains. It is not the user's brain. The scene
> metadata says so.

**`SCENE_VERSION` in `build.py` is currently 8. Bump it whenever geometry
changes**, or a stale cache in `neuroforge/_cache/` will be served silently.

### Subcortical geometry

`subcortex_real.py` extracts structures from the FreeSurfer `aseg`
segmentation of MNI152NLin2009cAsym. Eleven of the fifteen structures are
real; `acc`, `sgacc` and `vta` are still procedural (the first two are
cortical, the third is not an aseg label).

`isosurface.py` is naive **surface nets**, not marching cubes: one vertex per
boundary cell, quads around sign-changing grid edges. No 256-entry table to
transcribe, and much smoother output on a label mask. Work is confined to
each label's bounding box, which is what keeps the whole subcortex under five
seconds.

**The two atlases are in different spaces.** fsaverage is MNI305; the aseg
volume is MNI152. Dropping one into the other unchanged leaves structures a
few millimetres out - enough for a hippocampus to poke through the temporal
lobe. `_MNI152_TO_305` corrects it. Do not remove it.

Sanity check after any change: measured centroids should land near the
published coordinates - amygdala ~(23, -5, -21), hippocampus ~(26, -23, -16),
accumbens ~(8, 10, -9). Shape signatures matter too: the caudate must be a
long C, the brainstem a vertical column.

### Only publish regions that exist

`build.py` filters `CORTICAL_REGIONS` to labels actually present in the built
mesh. The procedural fallback cortex assigns far fewer labels than the real
parcellation, so without this filter thirteen regions opened a panel and then
highlighted nothing. Any new region must be verified on **both** geometry
paths with `scripts/coverage.js`.

### Functional networks

`CORTICAL_LABELS` has 15 entries. Indices 11-14 (`PCC`, `precuneus`, `TPJ`,
`FPC`) were added because the parcels that carry self-referential processing
were previously either invisible (posterior and isthmus cingulate fell into
`other`, which draws black) or buried inside a generic `parietal` label.

The default mode network is a separate layer, not a label. It has no single
anatomical home - it spans medial prefrontal, posterior cingulate, precuneus,
angular gyrus and lateral temporal cortex - so it can only be shown honestly
as an overlay derived from resting-state fMRI. The app loads the Yeo 2011
seven-network parcellation (`yeo_L` / `yeo_R` in `fetch.py`) as a per-vertex
`uint8` and paints it through an 8x1 `DataTexture`, exactly as region colours
go through the 16x1 one.

The procedural fallback surface has no resting-state data. In that case
`scene['networks']` is `[]`, the mesh carries no `network` buffer, and the
sidebar group hides itself. `_geomFrom` still uploads a zero-filled
`aNetwork` attribute so the shader compiles. Both paths are smoke-tested.

---

## 5b. Labels, selection, and groups

`CORTICAL_LABELS` in `mesh.py` has 27 entries (0-26). **The region/selection
textures are 32 wide, so label indices must stay <= 31.** Widen both
`uRegionTex` and `uSelTex` together if you ever exceed that.

Label 8 (`parietal`) is deliberately empty: every parcel that fed it was moved
into finer labels. `build.py` filters `CORTICAL_REGIONS` down to labels actually
`present` in the built mesh, so an empty label never produces a clickable region
with zero vertices. That filter exists because `scripts/coverage.js` caught 13
such regions on the procedural path.

### Geometric splits in `cortex_real._hemi()`

The Desikan-Killiany atlas has no parcel for several regions the user asked for,
so three are cut geometrically. **Each threshold was measured from the actual
vertex distribution before being chosen** - do not adjust them by eye.

| Label | Rule | Why this threshold |
|---|---|---|
| `sgACC` (6) | `rostralanteriorcingulate` and `z < 0` | Subgenual means below the genu of the corpus callosum. |
| `pINS` (25) | `insula` and `y < 0` | Parcel spans y -33..25, median -3. |
| `mPFC` (26) | `superiorfrontal` and `abs(x) < 14 and y > 20` | `abs(x) < 14` captures 69% medial. Both conditions are required: x alone drags in SMA, y alone takes the dorsolateral convexity. |

### Multi-label selection

Cortex selection uses a **32x1 `uSelTex` membership mask**, not a single
`uActive` float. This is what allows a group to highlight several labels at
once. `selectMany(labelIndices, structureIds, hit)` is the primitive;
`select(hit)` is its one-element case.

### GROUPS

PFC, lateral PFC, dorsal striatum and ventral striatum are **not parcels**.
They are composite terms, and inventing a mesh for them would mean drawing a
shape no atlas contains. They are instead `GROUPS` in `structures.py`:
selections over regions and structures that already exist. `groupPanel()` in
`ui.js` leads with a "NOT ONE REGION" section listing the components.

`build.py` publishes only groups whose every component is present:

```python
groups = [g for g in GROUPS
          if all(x in have_regions for x in g["regions"])
          and all(x in have_structs for x in g["structures"])]
```

On the procedural fallback the `PFC` group correctly disappears, because `mPFC`
and `FPC` do not exist there. Promising six regions and highlighting three would
be a lie about coverage.

---

## 6. The simulation model

`model.py` implements dual-rate plasticity per connection:
- `w` — labile weight, moves fast
- `c` — consolidated weight, moves slowly, resists decay
- `e` — eligibility trace

Plus **homeostatic competition** between inputs to a shared target.

### The single most important modelling principle

A test (`test_avoidance_moves_away_from_target`) caught a real dishonesty:
homeostatic scaling originally ran globally in `tick_day()`, so an unwanted
pathway faded **merely because time passed**. Fixed by scoping competition to
nodes actually engaged by a logged behaviour. The comment reads:

> *Competition between inputs is driven by use, not by the calendar —
> otherwise an unwanted pathway would fade simply because time passed,
> which would be both wrong and quietly dishonest.*

**Do not reintroduce calendar-driven change.** Repetition is the mechanism.

### Sensitivity (12 weeks, `simulate_scripted`)
| Adherence | Gap closed |
|---|---|
| 90 % | 74.8 % |
| 70 % | 75.7 % |
| 40 % | 40.5 % |
| 10 % | 8.7 % |

Effects are deliberately gradual. Avoid changes that let one action visibly
transform the brain.

---

## 7. Running and testing

```powershell
python -m neuroforge                    # serve + open browser
python -m neuroforge --no-browser       # serve only, port 8770
python -m neuroforge --rebuild          # force scene rebuild
python -m neuroforge --cli              # text prototype
python -m neuroforge.anatomy.fetch      # download real anatomy
python -m unittest discover -s tests    # 11 tests
```

### Browser testing (this is how rendering bugs get caught)
`node --check` validates syntax but **cannot** catch runtime WebGL failures.
`scripts/smoketest.js` drives headless Edge via `playwright-core` and reports
console errors, geometry counts and interaction results.

```powershell
$env:NODE_PATH="$env:TEMP\node_modules"
node scripts/smoketest.js http://127.0.0.1:8770/
```

Expected: `BOOT: ok`, `cortexTris: 163840`, `structures: 13`, `ERRORS: 0`.

---

## 8. Bugs already found and fixed — do not reintroduce

1. **Backtick inside a GLSL comment.** A comment mentioning `` `normal` ``
   inside a template-literal shader terminated the string, breaking the whole
   module. `node --check` passed; only the browser caught it. **Never put
   backticks in shader comments.**
2. **`d - d` in the longitudinal fissure** evaluated to 0.0, flattening the
   procedural brain above z = −14. Fissures are now proper SDF subtractions.
3. **GLSL dynamic uniform-array indexing** (`uRegionColor[li]`) is illegal in
   GLSL ES 1.0 fragment shaders. Replaced with a 16×1 RGBA `DataTexture`
   (RGB = colour, A = live glow).
4. **Cortex `depthWrite` must be `false` in x-ray mode**, or the glass shell
   occludes the subcortical structures inside it.
5. **Side panel covered the action buttons.** The 400 px panel sits at
   `right: 0`; floating controls now shift via `body.panel-open`.
6. **Cache key ignored the geometry source.** Anyone who ran the app before
   downloading the anatomy was served the cached procedural brain forever, so
   `fetch` looked broken. The source is now part of the key.
7. **Focus mode made the brain brighter** (51% → 73% lit). Two causes: the
   fresnel exponent was too low to read as an outline, and - the real culprit
   - the cortex is `DoubleSide`, so on back faces `dot(N,V)` is negative,
   clamps to 0, and fresnel returns 1.0. Fixed with `abs()`. **Measure with
   `readPixels`; do not eyeball a screenshot.**
8. **`readPixels` after the frame is presented returns all zeros.** The
   drawing buffer is cleared on present, so a late read looks like a
   perfectly dark brain rather than a failed measurement. `measureLitFraction`
   defers to the end of the next `composer.render()`.

---

## 9. Environment notes (original machine)

- Python 3.14.0, the only interpreter installed. No numpy.
- Node v25.9.0 present; `playwright-core` installed in `$env:TEMP`.
- Edge at `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`.
- Corporate **Zscaler** proxy blocks GitHub *release-asset* binaries
  (so Git could not be auto-installed), but the GitHub API, PyPI and the
  TemplateFlow S3 bucket are all reachable.

---

## 9b. Visual design rules

The specimen is pink-grey tissue. **Do not light it with coloured lights.**

A full session was lost to this: the tissue albedo was already correct
(`uBase` pink), but every light in the rig was blue (hemisphere `0x5588cc`,
key `0xbfe4ff`, fill `0x6f8fd0`, rim `0x38e0ff`), the shader added two
hardcoded blue bounce terms, the fresnel rim was cyan at 0.45, and bloom at
0.62 smeared all of it across the surface. The result read as a blue object.
Pink paint under four blue lights is a blue brain.

Measure the result, do not judge it by eye. Screenshot the canvas and compare
mean RGB at the centre of the specimen against a corner. Tissue must be
**red-dominant** (currently ~`136,120,118`); the backdrop must be near-neutral
(currently ~`18,21,26`).

| Rule | Why |
|---|---|
| Lights stay near-neutral | The tissue colour must survive the lighting. |
| Bloom strength <= ~0.25 with a high threshold | Bloom should catch genuine highlights (active region, firing pathway), never the whole specimen. Heavy bloom reads as plastic. |
| Broad, dim specular, suppressed in sulci | `pow(...,42)` was a hot pinpoint that looked polished. Sulci are recessed and should not catch a sheen. |
| No science-fiction set dressing | A starfield and a pulsing cyan ground ring were removed. They competed with the anatomy for attention and the reference atlas has neither. |
| UI palette is hue-free charcoal | The old navy palette tinted everything blue and drowned the accent colours that actually carry meaning. |

Honest limitation to keep in mind before promising "better geometry": our
meshes come from fsaverage and a 2 mm `aseg`. That is real measured anatomy,
but it will always look softer than a sculpted mesh. The gap cannot be closed
without inventing geometry.

---

## 9c. Pathways as objects, and comparison

The app exists so the user can log experiences and see what changed. Two
things follow from that, and both were missing for a long time.

### Per-edge history is sent to the client

`DaySnapshot.weights` has always recorded every edge every day, but
`routes._state()` did not serialise it, so the client could only ever draw
"now". It is now included, rounded to 3 dp. Without this there is no
sparkline and no comparison - the whole point of the product.

### Edges are picked against the curve, not the mesh

`Circuits.pickEdge()` samples each curve and compares against the *current*
`uRadius`. Do not replace this with `raycaster.intersectObjects()` on the
tubes: `TubeGeometry` is built at a fixed radius of 1.0 and the shader
inflates it to as much as ~3.8 to show strength, so a mesh raycast would
disagree with what the user can see. Focused-out edges (`uFocus < 0.3`) are
skipped so they cannot swallow clicks.

`viewer.circuits` must be assigned in `main.js` after the circuits are built.
The guard in `_pick()` is `this.circuits?.`, so forgetting it makes pathways
silently unclickable with no error.

### Comparison mode

`Circuits.showDelta(from, to)` paints direction as colour (green stronger,
amber weaker) and magnitude as radius, on a **fixed** +/-0.35 scale.
Normalising to the largest change present would make the biggest mover look
dramatic on a day when nothing much happened.

`update()` returns early while `deltaMode` is set, or the live weights would
immediately overwrite the comparison.

### Testing note that cost real time

When probing from the browser, **snapshot values rather than holding live
references**. `uniforms.uColor.value` is a `THREE.Color` object; a probe that
kept the reference and read `.r/.g/.b` at the end of the run reported the
restored colour and looked like a bug in `showDelta`. The radii were correct
only because they were captured as numbers. Same class of error made
`deltaMode` read as `false` immediately after being set to `true`.

---

## 9d. The language-model boundary

`neuroforge/interpret.py` turns free text into *suggested* events. Read its
docstring before changing anything in it.

**A model may classify. A model may never decide what happens to the brain.**

The effects are the honest part of this app: they come from fixed rules with
stated caveats. If an LLM could invent an effect - "this sounds like it
strengthened your vmPFC" - the app would be generating neuroscience claims
about a real person from a sentence of prose, which is the one thing the
whole project promises not to do. A classifier can be wrong in a way the user
can see and correct. An effect-generator would be wrong in a way nobody could
check.

Consequences that must be preserved:

| Rule | Where |
|---|---|
| Model output is validated against `EVENT_TYPES` and **dropped**, not coerced | `_validate()` |
| `/api/sim/interpret` never mutates the simulation | `routes._interpret` |
| Applying a proposal goes through the same `doLog()` as a button press | `main.js` |
| Network use is opt-in, off by default, key from env only | `llm_config()` |
| A network failure falls back to keywords and says so | `interpret()` |

Configuration is environment-only, never committed:
`NEUROFORGE_LLM` (`openai`|`gemini`), `NEUROFORGE_LLM_KEY`,
`NEUROFORGE_LLM_MODEL`, `NEUROFORGE_LLM_BASE`. The OpenAI path is
OpenAI-compatible, so Groq / OpenRouter / a local llama.cpp server work by
setting `NEUROFORGE_LLM_BASE`.

The offline keyword classifier is deliberately crude and **caps its own
confidence at 0.62**. A keyword hit is evidence that a word appeared, not
evidence that the user meant it. It also checks for negation: "didn't get
angry" is a regulated success, and a naive matcher that saw "angry" would
record the exact opposite of what happened. There is a test for this.

---

## 9e. Concurrency

The server is a `ThreadingHTTPServer`. It used to be a plain `HTTPServer`,
which meant one slow request blocked every other request — including the
static assets and the state polling the page needs to stay responsive. That
was tolerable while every handler was fast. It stops being tolerable the
moment a language model is in the request path.

`routes.SIM` is module-level mutable state shared by all handler threads.
`log_event`, `advance_day` and the outright rebinds in reset/demo/load are
read-modify-write sequences. They are guarded by `routes.STATE_LOCK`, a
reentrant lock, applied with the `@guarded` decorator.

**Apply `@guarded` *below* `@API.get`/`@API.post`**, so the callable that gets
registered in the route table is the locked wrapper rather than the bare
function:

```python
@API.post("/api/sim/log")
@guarded
def _log(payload): ...
```

`/api/sim/log` in particular must hold the lock across its whole body, not
just the mutation: it reads a `before` profile, mutates, then reads `after`.
If another log interleaves, the reported deltas silently attribute someone
else's change to this event.

**`/api/sim/interpret` is deliberately NOT guarded.** It never touches `SIM`,
and it is by far the slowest endpoint. Holding the lock across a model call
would serialise the entire application behind it and give back exactly the
stall that threading was meant to remove. Any future streaming/chat endpoint
must follow the same rule: do the slow work outside the lock, take the lock
only to read or apply a result.

Verified when this landed: 8 concurrent `/api/scene` requests overlapped 5.3×;
40 concurrent logs were all recorded with no errors; and total |delta| across
12 concurrent logs was 0.7835, identical to the sequential baseline.

---

## 10. Status

**Working:** real anatomy, all three modes, view presets, picking, circuits
with GPU-driven thickness, timeline scrubbing, logging with per-pathway delta
readout, save/load, 11/11 tests passing, 0 console errors.

**Superseded:** `neuroforge/ui/` is the original Tkinter dashboard. It still
runs but is no longer the product. Treat as legacy.

**Open ideas:** per-region info sourced from a citable dataset; export of a
session summary.

**Known gap, user-visible:** the log modal still covers the brain, so the
user cannot watch the model respond while they write. Free-text entry now
lives inside that modal, which makes the layout problem more acute rather
than less. Turning logging into a persistent side surface is the largest
remaining UX item. The UI palette and typography have been addressed; the
*layout* has not.

**Deliberately not done:** cranial nerves, meningeal layers, pituitary,
hippocampal subfields. The BrainFacts reference model is hand-sculpted by
named 3-D artists. These cannot be derived from any neuroimaging atlas, so
adding them would mean inventing shapes and presenting them as anatomy.
