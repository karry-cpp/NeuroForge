# Glow

How brain parts glow, where the numbers come from, and how to reuse or tune
it. Everything here is rendering: the glow shows a simulation value, it does
not measure anything.

## The short version

A glowing part is made of two things:

1. **Self-illumination (emission).** The part gives off light in its own
   colour, on top of normal lighting. Structures do it through the
   material's `emissiveIntensity`; cortex regions do it in the cortex shader.
2. **Bloom.** A post-process pass (`UnrealBloomPass`) takes every pixel
   brighter than a threshold and spreads it into a soft halo. This is what
   makes it read as *glow* rather than as a bright colour.

A part only gets a halo once its emission pushes it past the bloom
threshold. Everything below is about controlling how far past it each part
goes.

## The pipeline, end to end

```
server  analyse()  ->  moves: [{part, dir, strength 0..1}]     (engagement.moves_for)
main.js applyVisualise()  ->  glow value g per part             (0 .. ~0.62)
viewer  setStructureGlow(id, g) / setRegionGlow(label, g)
viewer  render loop (structures) / cortex shader (cortex)  ->  emission
bloom pass  ->  halo
```

### 1. Strength (server)

`engagement.moves_for(events)` gives each part the moment pushes a signed
pull from the app's rules, normalised so the strongest part is `1.0`.
Example, "she ignored me and I keep replaying it":
amygdala 1.0, ACC 0.36, brainstem 0.25, hippocampus 0.14.

### 2. Glow value `g` (main.js)

```js
const VIS_GLOW_MIN = 0.12, VIS_GLOW_SPAN = 0.5, VIS_GLOW_FOCUS = 0.75;
g = VIS_GLOW_MIN + VIS_GLOW_SPAN * Math.sqrt(strength);   // parts pushed UP
g = 0;                                                      // parts pushed DOWN
```

- The square root keeps weakly pushed parts visible.
- Parts that are involved but have no direction (insula, parts the model
  named) use `visGlow()`, from the server's `levels`.
- Clicking a part in "In this moment" isolates it but keeps **its own** `g`,
  so the glow still says how hard it is pushed.
- `VIS_GLOW_FOCUS = 0.75` is the "full" glow (see *Reusing it*).

### 3. Strong-glow mode (viewer.js)

`viewer.setStrongGlow(true)` is switched on only in the Visualise tab
(`setViewState`). It changes the emission formulas below; every other view
uses the ordinary selection glow.

### 4. Emission: structures (viewer.js, `render()`)

```js
push = clamp((g - 0.12) / 0.5, 0, 1) ** 2;        // undoes the sqrt: back to strength
want = (base + 0.22 + (g + 0.3) * 0.6             // the usual glow, every part
        + STRUCT_BOOST * push * push)             // boost (0.7), mostly for the top part
       * lumComp * (GLOW_GAIN[id] ?? 1) * hl;     // evened out to the ACC (section 6)
```

- `base` is 0.16 (0.34 with x-ray on).
- **The boost grows with the square of the push.** A part with half the push
  gets about one-sixteenth of the boost, so it goes almost entirely to the
  most involved part (the amygdala in the example). The other parts keep the
  usual glow.
- `lumComp` and `GLOW_GAIN` even out brightness across parts (see below).
- `hl` is the **Highlight** slider (`setHighlight`, 1.0 = 100%, 2.0 = 200%).
  It multiplies everything, which is why 200% looks so much stronger.
- The value is eased toward `want` every frame (`dt * 8`), so changes fade
  in rather than jump.

### 5. Emission: cortex (viewer.js, cortex fragment shader)

```glsl
vec3 ec = rc / max(max(max(rc.r, rc.g), rc.b), 0.05);   // full-strength hue
float lum  = dot(ec, vec3(0.2126, 0.7152, 0.0722));
float comp = clamp(uLumRef / max(lum, 0.05), 0.5, 2.5);  // ACC's yellow = 1
float s = clamp((glow - 0.12) / 0.5, 0.0, 1.0);  s *= s;
lit += ec * (0.38 + 0.6 * (glow + 0.3) + uBoostK * s * s)
       * comp * vGlowGain * uHighlight * uCortexGain;    // uCortexGain = 0.6
```

- `ec` stretches the region's colour to full intensity, so darker swatches
  still reach the bloom threshold.
- The trailing `uCortexGain` (0.6) is because cortex is already lit brighter
  than a structure by the swatch-colour pull just above it in the shader.
- Same shape as the structure formula: usual glow plus a squared boost.

### 6. Evening out brightness across parts

Every part's strong glow is matched to one reference: the **Anterior
cingulate's yellow** (`GLOW_REF_COLOR = 0xfbbf24`). It stays exactly as it
was; everything else is scaled to look as bright. Two layers:

**a. Colour.** At the same emission, colours do not look equally bright: the
eye weighs green-yellow far more than red. Measured brightness of
full-strength swatches:

| Part | Colour | Brightness |
|---|---|---|
| ACC | `#fbbf24` yellow | ~0.57 |
| Hippocampus | `#4ade80` green | ~0.53 |
| Brainstem | `#94a3b8` grey | ~0.36 |
| Amygdala | `#ff5c6e` red-pink | ~0.30 |

Without correction the ACC outshone the amygdala even though the amygdala is
pushed almost three times as hard. So the **whole** strong-glow emission is multiplied by the reference
brightness over the part's own:

- Structures: `lumComp = clamp(refLum / lum, 0.5, 2.5)`, computed once per
  mesh in `_makeStructure` (ACC = 1).
- Cortex: `comp = clamp(uLumRef / lum, 0.5, 2.5)` in the shader, where
  `uLumRef` is the reference hue worked out the same way the shader sees it.
- The boost terms are sized (`STRUCT_BOOST = 0.7`, `uBoostK`) so the ACC's
  own glow is unchanged.

**b. Shape and position (`GLOW_GAIN`).** Colour is not everything: a large
region facing the camera (Planning/dlPFC) adds up a bigger halo, and a region
on the underside (Reward value/OFC) or deep inside (VTA) loses some. These
were measured on screen, per part, against the ACC at Highlight 65% (mean
brightness of the brightest 1500 pixels, labels and panels hidden), and a
correction stored per part id in `GLOW_GAIN` (viewer.js). Missing parts
default to 1. Cortex gains travel to the shader in the blue channel of the
selection texture (`vGlowGain`, range 0..2).

Result at Highlight 65% (ACC = 210): every Regulation part within about
202–217. To re-tune a part by eye in the console:
`NF.viewer.setGlowGain('dlPFC', 0.5)` and select it again.

Not yet tuned: the parts outside the Regulation shortlist (they use the
colour correction only).

### 7. Bloom

```js
new UnrealBloomPass(size, strength 0.24, radius 0.62, threshold 0.88);
this.bloom.strength = anatomy ? 0.20 : 0.46;   // setMode()
```

- **Threshold 0.88**: only pixels brighter than this bloom. That keeps the
  tissue matte and lets only glowing parts radiate.
- **Strength 0.46** in Circuits and Simulation, 0.20 in Anatomy.
- Tone mapping is ACES filmic at exposure 1.0. Very bright emission saturates
  toward white, which is why 200% can wash the hottest part out.

### 8. What makes the focused part look so strong

Clicking a part calls `setEmphasis([...that part], ..., fadeRest=true,
seeThrough=true)`. The glow on the part itself is unchanged, but everything
else drops away:

- Other cortex parts move to the transparent shell pass (`uFaintGhost`) with
  only a trace of their colour.
- Other structures go to 6% opacity and 10% emission.

With nothing competing, the bloom halo stands out against an almost empty
background. Much of the "beautiful" look is this contrast, not extra
emission.

### 9. Zoom

The scene's fog thickens with camera distance, which used to grey the brain
and sink its glow below the bloom threshold when zoomed out. `render()` now
scales the fog density by `VIEW_DIST / distance`, so the glow looks the same
at every zoom.

## Reusing it

### Already reused: the Anatomy tab

Selecting a part in Anatomy (on the brain, its label, or a sidebar row) gives
it the full glow. `viewer.selectMany()` calls `_anatomyGlow()` when
`this.mode === 'anatomy'`:

```js
this.setStrongGlow(any);                 // on while something is selected
this.clearGlow();
for (const i of labels) this.setRegionGlow(i, FULL_GLOW);      // 0.75
for (const id of structs) this.setStructureGlow(id, FULL_GLOW);
this.bloom.strength = any ? SIM_BLOOM : ANATOMY_BLOOM;          // 0.46 / 0.20
```

Bloom goes up to the Simulation level only while something is selected, so
the bare specimen stays matte. Clearing the selection or leaving Anatomy
turns all of it back off. Groups (e.g. "PFC") light every member fully.

### Already reused: the Current tab

`setViewState` turns strong glow on for Current as well as Visualise.
`paintEngagement(V, rows, focus, glow=true)` gives each **more** engaged part
`0.12 + 0.5 * sqrt(score / topScore)` (`engGlowMap`), so the most engaged part
gets the full Visualise glow. **Less** engaged parts get a **pale** glow:
`0.25 * (1 - 0.6 * sqrt(|score| / topLess))`, drawn by `viewer.setPale()`
as a weak emission washed `PALE_MIX` (0.55) toward white, so they read as
fainter and paler, not as "off". Cortex carries the flag in the selection
texture's alpha channel (`vPale`); structures in `userData.pale`.
The values live in `App.engGlow`: the "just changed" flash
(`playEngagementChange`) never drops below them, and the log cascade
(`animateResponse`) restores them when it clears.

### Anywhere else

```js
const V = App.viewer;
V.setStrongGlow(true);                         // the Visualise formulas
V.selectMany(cortexLabelIndices, structureIds, null);
V.setEmphasis(focusLabels, focusStructs, true, true);   // optional: isolate
V.clearGlow();
V.setStructureGlow('amygdala', 0.62);          // per part, 0.12 .. 0.62
V.setRegionGlow(labelIndex, 0.62);
// ... and when leaving the view:
V.setStrongGlow(false);
```

- **Full glow** (the strongest look): pass `VIS_GLOW_FOCUS` (`0.75`). The
  boost is clamped, so anything at or above 0.62 gets all of it.
- **Graded glow**: `0.12 + 0.5 * Math.sqrt(strength)` for a 0..1 strength.
- **No glow, still visible**: select the part but pass `0`.
- Remember to turn strong glow off when leaving, or other views pick up the
  stronger emission. `setMode()` does this for non-Simulation modes.

## Tuning knobs

| What | Where | Effect |
|---|---|---|
| `VIS_GLOW_MIN`, `VIS_GLOW_SPAN` | main.js | Range of `g`. Raising MIN lifts weak parts. |
| `Math.sqrt(strength)` | main.js, `applyVisualise` | Use the strength directly for more contrast between parts. |
| `GLOW_REF_COLOR` | viewer.js | The hue every part is matched to (the ACC's yellow). |
| `GLOW_GAIN` | viewer.js | Per-part correction measured on screen; tune live with `NF.viewer.setGlowGain(id, v)`. |
| `STRUCT_BOOST`, `uBoostK` | viewer.js | Size of the boost for the most involved part. |
| Clamps `0.55..1.8`, `0.6..1.6` | viewer.js | Limit how much a dark or bright colour is corrected. |
| `0.38 + 0.6 * (glow + 0.3)` | viewer.js | The usual glow every part gets. |
| `bloom.strength`, threshold `0.88` | viewer.js | Halo size; lower threshold makes more things glow. |
| Highlight slider | UI, Appearance | Multiplies all emission at runtime. |
