/* ============================================================================
 * viewer.js — the 3-D brain.
 *
 * Owns: renderer, camera, lighting, post-processing, the cortical surface,
 * the subcortical structures, picking, view presets and screen-space pins.
 *
 * Knows NOTHING about the learning simulation. It exposes setters
 * (setMode, setState, highlight…) that main.js drives.
 * ========================================================================= */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { decodeF32, decodeU32, decodeU8, decodeU16 } from './api.js';

// Structures that form the outline of the specimen rather than sitting inside
// it. Muting these with the rest left the brain looking cut off at the bottom.
const SILHOUETTE = new Set(['cerebellum', 'brainstem']);
const GHOST_GREY = new THREE.Color(0x8b9199);

/* ------------------------------------------------------------------ shaders */

const CORTEX_VERT = /* glsl */`
  attribute float aLabel;
  attribute float aSulc;
  attribute float aNetwork;
  uniform float uHasSulc;
  uniform sampler2D uRegionTex;
  uniform sampler2D uSelTex;
  uniform sampler2D uNetTex;
  uniform float uNetActive;
  varying vec3  vN;
  varying vec3  vWorld;
  varying vec3  vViewDir;
  varying vec4  vReg;
  varying vec4  vNetCol;
  varying float vSel;
  varying float vNetOn;
  varying float vHasNet;
  varying float vDepthFold;

  void main() {
    // Sampled per VERTEX with the exact label, and the RESULT is what gets
    // interpolated. Doing these lookups in the fragment shader meant reading
    // an interpolated label: a triangle spanning labels 1 and 26 sweeps
    // through every index between them, and a nearest-neighbour fetch then
    // lit up whichever regions those indices happen to be - thin bands of the
    // selected colour scattered across the brain. Only label 26 was immune,
    // because nothing can interpolate past the highest index in use.
    vReg = texture2D(uRegionTex, vec2((aLabel + 0.5) / 32.0, 0.5));
    vSel = texture2D(uSelTex, vec2((aLabel + 0.5) / 32.0, 0.5)).r;
    vNetCol = texture2D(uNetTex, vec2((aNetwork + 0.5) / 8.0, 0.5));
    vNetOn = (uNetActive >= 0.0 && abs(aNetwork - uNetActive) < 0.5)
             ? 1.0 : 0.0;
    vHasNet = aNetwork > 0.5 ? 1.0 : 0.0;

    vN = normalize(normalMatrix * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    vec4 mv = viewMatrix * world;
    vViewDir = normalize(-mv.xyz);
    // How deep in a sulcus is this vertex? With the real fsaverage surface
    // this is measured sulcal depth, so creases darken because they are
    // genuinely creases. Without it we fall back to distance from the
    // centre, which only approximates the same thing.
    if (uHasSulc > 0.5) {
      vDepthFold = clamp(aSulc * 0.42 + 0.42, 0.0, 1.0);
    } else {
      vDepthFold = clamp((length(position) - 62.0) / 34.0, 0.0, 1.0);
    }
    gl_Position = projectionMatrix * mv;
  }
`;

const CORTEX_FRAG = /* glsl */`
  precision highp float;

  varying vec3  vN;
  varying vec3  vWorld;
  varying vec3  vViewDir;
  varying vec4  vReg;
  varying vec4  vNetCol;
  varying float vSel;
  varying float vNetOn;
  varying float vHasNet;
  varying float vDepthFold;

  uniform vec3  uBase;
  uniform vec3  uDeep;
  uniform float uOpacity;
  uniform float uXray;
  uniform float uTime;
  uniform float uActive;        // currently highlighted label (-1 = none)
  uniform float uDim;           // dim non-highlighted regions
  uniform float uGhostFade;     // 1 = ghosted cortex must also turn see-through
  uniform float uSelLayer;      // 1 = this pass draws ONLY the selection, opaque
  uniform float uGhostAlpha;    // user control: visibility of everything unselected
  uniform float uHighlight;     // user control: strength of the selected region
  uniform sampler2D uRegionTex; // RGB = region colour, A = live glow
  uniform sampler2D uSelTex;    // R > 0.5 = this label is currently selected
  uniform sampler2D uNetTex;    // RGB = Yeo network colour
  uniform float uNetMode;       // 1 = colour cortex by network, not by region
  uniform float uNetActive;     // highlighted network (-1 = none)
  uniform vec3  uKeyDir;
  uniform vec3  uRimColor;

  void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(vViewDir);
    float ndv = clamp(dot(N, V), 0.0, 1.0);

    // ---- base tissue -----------------------------------------------------
    // Gyral crowns catch light, sulcal depths fall into shadow.
    vec3 albedo = mix(uDeep, uBase, 1.0 - vDepthFold);

    // ---- region tint ------------------------------------------------------
    vec4 reg = vReg;
    vec4 net = vNetCol;
    bool netMode = uNetMode > 0.5;

    // In network mode the cortex is painted by functional network instead of
    // by anatomical parcel, because a network like the default mode has no
    // single anatomical home - it is defined by what co-activates, not by
    // where one gyrus ends.
    vec3 rc = netMode ? net.rgb : reg.rgb;
    float glow = netMode
      ? (vHasNet > 0.5 ? 0.62 : 0.0)
      : reg.a;
    bool isActive = netMode ? (vNetOn > 0.5) : (vSel > 0.5);

    // The selection is drawn by a separate opaque pass that writes depth, so
    // that it occludes properly instead of blending with whatever the far
    // hemisphere happens to have drawn. Each pass throws away the other's
    // fragments. Without this the highlighted parcel came out mottled, and
    // the far side's copy of it showed through as a wash of its own colour.
    bool selLayer = uSelLayer > 0.5;
    bool focusing = uDim > 0.5;
    if (selLayer && !isActive) discard;
    if (!selLayer && isActive && focusing) discard;

    // Anything that is not the selection is about to be ghosted, and it must
    // give up its live simulation glow first. The glow feeds a fresnel term
    // further down, so a region that happened to be carrying activity kept a
    // bright rim and read as a white outline drawn around it - which is why
    // only the regions wired to simulation nodes showed the effect.
    bool ghosted = (uDim > 0.5) && !isActive;
    if (ghosted) glow = 0.0;

    float tint = glow * 0.55 + (isActive ? 0.50 * uHighlight : 0.0);
    albedo = mix(albedo, rc, clamp(tint, 0.0, 0.9));
    // Tinting mixes toward a single colour, which erases the gyral/sulcal
    // contrast the albedo above was built from. Re-apply it, or a selected
    // region flattens into one bright block and stops reading as tissue.
    albedo *= 1.0 - vDepthFold * (isActive ? 0.45 : 0.0);

    // ---- lighting --------------------------------------------------------
    float key  = clamp(dot(N, normalize(uKeyDir)), 0.0, 1.0);
    // wrapped diffuse fakes the light bleed of translucent tissue
    float wrap = clamp((dot(N, normalize(uKeyDir)) + 0.55) / 1.55, 0.0, 1.0);
    float fill = clamp(dot(N, normalize(vec3(-0.6, 0.2, -0.5))), 0.0, 1.0);
    float sky  = 0.5 + 0.5 * N.y;

    vec3 lit = albedo * (0.14 + 0.54 * wrap + 0.30 * key * key);
    // Near-neutral bounce. These used to be strongly blue, which is most of
    // why pink-grey tissue rendered as a blue object.
    lit += albedo * fill * 0.17 * vec3(0.82, 0.86, 0.95);
    lit += albedo * sky  * 0.13 * vec3(0.88, 0.89, 0.93);

    // specular sheen (brain tissue is wet). Broad and dim: a tight hot
    // highlight reads as polished plastic, not as a moist membrane.
    vec3 H = normalize(normalize(uKeyDir) + V);
    float spec = pow(clamp(dot(N, H), 0.0, 1.0), 26.0);
    // A tinted region is already bright, so the white sheen on top of it is
    // what tips the selection over into looking like painted plastic.
    float specGain = isActive ? 0.06 : 0.16;
    // Sulci are recessed and should not catch the sheen that crowns do.
    lit += vec3(1.0, 0.97, 0.93) * spec * specGain * (1.0 - vDepthFold * 0.85);

    // ---- fresnel rim -----------------------------------------------------
    // Kept subtle in the solid view. The rim exists to separate the specimen
    // from the backdrop, not to make it look like a hologram.
    float fres = pow(1.0 - ndv, 3.0);
    lit += uRimColor * fres * (0.16 + 1.1 * uXray);
    lit += rc * glow * fres * 1.1;

    // active region gets a travelling shimmer so it reads as "selected"
    if (isActive) {
      float band = sin(vWorld.z * 0.09 - uTime * 2.4) * 0.5 + 0.5;
      lit += rc * band * 0.07 * uHighlight;
      lit += rc * fres * 0.18 * uHighlight;
    }

    // ---- opacity ---------------------------------------------------------
    float a = uOpacity;
    if (uXray > 0.5) a = mix(0.055, 0.40, fres);   // glass shell

    // Focus mode. The unselected cortex stays real tissue and is simply
    // pushed far down in brightness and saturation.
    //
    // It used to collapse to a fresnel silhouette instead. That works on a
    // smooth blob, but this is a 163k-triangle folded surface: every gyral
    // crown grazes the view direction somewhere, so the "outline" picked out
    // all of them at once and read as a wire mesh draped over the brain.
    // There is no exponent that fixes it - the detail is in the geometry.
    float ghost = uDim * (isActive ? 0.0 : 1.0);
    if (ghost > 0.0) {
      float lum = dot(lit, vec3(0.299, 0.587, 0.114));
      vec3 g = mix(vec3(lum), uRimColor, 0.22) * (0.30 * uGhostAlpha);
      lit = mix(lit, g, ghost);

      // Only dissolve the shell when the thing being highlighted is INSIDE
      // it. For a cortical selection, staying opaque is what keeps the
      // surface reading as a surface.
      //
      // A flat base alpha plus a wide, low-exponent rim, not a tight edge
      // term: the rim is there to give the shell a shape, and any narrow
      // falloff brings the wire-mesh look straight back.
      float shell = (0.09 + 0.20 * pow(1.0 - abs(dot(N, V)), 3.0))
                    * uGhostAlpha;
      if (!gl_FrontFacing) shell *= 0.45;
      float clear = mix(1.0, clamp(shell, 0.0, 1.0), uGhostFade);
      a = mix(a, a * clear, ghost);
      // Throwing the invisible fragments away entirely is what makes this
      // work: a fragment that is kept would still write depth and would
      // quietly occlude the structures we are trying to reveal.
      if (a < 0.03) discard;
    } else if (isActive) {
      // Fully opaque, so the parcel reads as solid tissue rather than as a
      // tinted film lying over the shell behind it.
      a = 1.0;
      // The cortex is DoubleSide and writes no depth while the shell is
      // dissolved, so the unlit inside of the far wall can blend over the
      // near one. Darkening it keeps the parcel reading as one surface.
      if (!gl_FrontFacing) lit *= 0.30;
    }

    gl_FragColor = vec4(lit, a);
  }
`;

/* ------------------------------------------------------------------ colours */

const REGION_COLORS = {
  0: 0x000000, 1: 0x22d3ee, 2: 0x0ea5e9, 3: 0x3b82f6, 4: 0x6366f1,
  5: 0xfbbf24, 6: 0xf59e0b, 7: 0xa3a3a3, 8: 0x94a3b8, 9: 0xa8a29e,
  10: 0x78716c,
  // Self-referential regions, given their own colours so they stop being
  // swallowed by "parietal" or, worse, by "other" - which drew them in black.
  11: 0xf472b6,   // posterior cingulate
  12: 0xc084fc,   // precuneus
  13: 0x22c55e,   // temporoparietal junction
  14: 0x38bdf8,   // frontopolar cortex
  // General anatomical coverage. Kept deliberately paler and cooler than
  // the regions above, so the parts the simulation actually uses still
  // read as the foreground.
  15: 0xcbd5e1,   // premotor
  16: 0xe2e8f0,   // somatosensory
  17: 0xa5b4fc,   // superior parietal lobule
  18: 0x818cf8,   // primary visual
  19: 0xf0abfc,   // primary auditory
  20: 0x34d399,   // medial temporal
  21: 0xfcd34d,   // fusiform
  22: 0x5eead4,   // superior temporal
  23: 0x86efac,   // supramarginal
  24: 0xfb7185,   // anterior insula
  25: 0x9f1239,   // posterior insula
  26: 0x60a5fa,   // medial prefrontal
};

/* Yeo 2011 seven-network colours. These are close to the palette used in the
 * original paper, so a reader who has seen the figure recognises them. */
const NETWORK_COLORS = [
  0x2a3444,   // 0 unassigned (medial wall etc.)
  0x7b287d,   // 1 visual
  0x5b9bd5,   // 2 somatomotor
  0x2e9e4f,   // 3 dorsal attention
  0xc43ad6,   // 4 ventral attention
  0xe8e07a,   // 5 limbic
  0xe8913a,   // 6 frontoparietal control
  0xd6474b,   // 7 default mode
];

/* ========================================================================= */

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.clock = new THREE.Clock();
    this.mode = 'anatomy';
    this.selected = null;       // { kind:'cortex'|'structure', id }
    this.hovered = null;
    this.pins = new Map();
    this.structures = new Map();
    this.structMeta = new Map();
    this.hemi = 'both';
    this.showPins = true;
    this.showSubcort = true;
    this.autoRotate = false;
    this._camTween = null;
    this._modeGhost = false;
    this._netActive = false;
    this._xrayOn = false;
    this._onPick = () => {};
    this._onHover = () => {};

    this._initRenderer();
    this._initScene();
    this._initControls();
    this._initPicking();
    window.addEventListener('resize', () => this._resize());
  }

  /* ---------------------------------------------------------- renderer */
  _initRenderer() {
    const r = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: true, alpha: false,
      powerPreference: 'high-performance', stencil: false,
    });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.setSize(innerWidth, innerHeight);
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = r;

    this.camera = new THREE.PerspectiveCamera(
      36, innerWidth / innerHeight, 1, 4000);
    this.camera.position.set(300, 90, 190);

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(
      this.scene || new THREE.Scene(), this.camera));
    // Low strength and a high threshold: bloom should only catch genuine
    // highlights (an active region, a firing pathway), never the whole
    // specimen. Heavy bloom is what made the surface look like plastic.
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(innerWidth, innerHeight), 0.24, 0.62, 0.88);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
  }

  /* ------------------------------------------------------------- scene */
  _initScene() {
    const s = new THREE.Scene();
    // Neutral charcoal, not navy. A blue background pushes a blue cast through
    // the fog and the rim light onto tissue that should read as pink-grey.
    s.background = new THREE.Color(0x16181d);
    s.fog = new THREE.FogExp2(0x16181d, 0.0011);
    this.scene = s;
    this.composer.passes[0].scene = s;

    this.root = new THREE.Group();
    // Our anatomy is RAS (+x right, +y anterior, +z superior); three.js is
    // Y-up. This single rotation maps (x,y,z) -> (x, z, -y) for everything
    // in the scene graph, so all downstream code can stay in millimetres.
    this.root.rotation.x = -Math.PI / 2;
    s.add(this.root);

    // Three-point studio rig, near-neutral. Anatomical illustration wants the
    // tissue colour to survive the lighting, so the key is barely warm and the
    // fill is barely cool instead of both being strongly tinted.
    s.add(new THREE.HemisphereLight(0xdfe4ec, 0x26221f, 0.62));
    const key = new THREE.DirectionalLight(0xfff3e8, 1.45);
    key.position.set(180, 220, 160);
    s.add(key);
    const fill = new THREE.DirectionalLight(0xa8b6c8, 0.48);
    fill.position.set(-200, -60, -150);
    s.add(fill);
    const rim = new THREE.DirectionalLight(0xf2f6ff, 0.62);
    rim.position.set(-90, 120, -230);
    s.add(rim);
    this.keyDir = key.position.clone().normalize();

    this._addBackdrop();
  }

  /**
   * A single large inward-facing sphere carrying a soft vertical gradient,
   * brighter just behind the specimen. This replaces the starfield and the
   * pulsing ground ring: both were science-fiction set dressing that competed
   * with the anatomy for attention, and the reference atlas has neither.
   *
   * BackSide + depthWrite:false + fog:false keeps it purely a backdrop - it
   * must never occlude or tint the brain.
   */
  _addBackdrop() {
    const g = new THREE.SphereGeometry(2600, 32, 24);
    const m = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: {
        uTop:    { value: new THREE.Color(0x0d0f13) },
        uMid:    { value: new THREE.Color(0x24272e) },
        uBottom: { value: new THREE.Color(0x0a0b0e) },
      },
      vertexShader: `varying vec3 vP;
        void main(){ vP = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
      fragmentShader: `varying vec3 vP;
        uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uBottom;
        void main(){
          float h = vP.y * 0.5 + 0.5;
          vec3 c = h < 0.5
            ? mix(uBottom, uMid, smoothstep(0.0, 0.5, h))
            : mix(uMid, uTop, smoothstep(0.5, 1.0, h));
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    this.scene.add(new THREE.Mesh(g, m));
  }

  /* ---------------------------------------------------------- controls */
  _initControls() {
    const c = new OrbitControls(this.camera, this.renderer.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.06;
    c.rotateSpeed = 0.62;
    c.panSpeed = 0.7;
    c.zoomSpeed = 0.85;
    c.minDistance = 95;
    c.maxDistance = 900;
    c.target.set(0, 0, 0);
    c.autoRotateSpeed = 0.55;
    this.controls = c;
  }

  /* ----------------------------------------------------------- picking */
  _initPicking() {
    this.ray = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    let downAt = null;

    this.canvas.addEventListener('pointerdown', (e) => {
      downAt = { x: e.clientX, y: e.clientY, t: performance.now() };
    });
    this.canvas.addEventListener('pointerup', (e) => {
      if (!downAt) return;
      const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
      const dt = performance.now() - downAt.t;
      downAt = null;
      if (moved > 5 || dt > 420) return;      // it was a drag, not a click
      const hit = this._pick(e);
      this._onPick(hit);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      this._lastPointer = e;
      const hit = this._pick(e);
      const id = hit ? `${hit.kind}:${hit.id}` : null;
      if (id !== this._hoverId) {
        this._hoverId = id;
        this.hovered = hit;
        this._onHover(hit, e);
      }
    });
    this.canvas.addEventListener('pointerleave', () => {
      this._hoverId = null; this.hovered = null; this._onHover(null);
    });
  }

  _pick(e) {
    this.pointer.x = (e.clientX / innerWidth) * 2 - 1;
    this.pointer.y = -(e.clientY / innerHeight) * 2 + 1;
    this.ray.setFromCamera(this.pointer, this.camera);

    // Pathways take priority over everything when they are on screen. They
    // are thin, they sit inside the cortex, and in circuit mode they are the
    // thing the user is actually looking at - so they must be clickable
    // rather than being decoration drawn over the anatomy.
    if (this.circuits?.visible) {
      // pickEdge wants the Ray, not the Raycaster that owns it.
      const id = this.circuits.pickEdge(this.ray.ray);
      if (id) return { kind: 'edge', id };
    }

    // subcortical structures take priority - they are small and inside
    if (this.showSubcort) {
      const meshes = [...this.structures.values()].filter(m => m.visible);
      const hits = this.ray.intersectObjects(meshes, false);
      if (hits.length) {
        return { kind: 'structure', id: hits[0].object.userData.id,
                 point: hits[0].point };
      }
    }
    const cortexMeshes = [this.cortexL, this.cortexR].filter(
      m => m && m.visible);
    const hits = this.ray.intersectObjects(cortexMeshes, false);
    if (hits.length) {
      const h = hits[0];
      const labels = h.object.geometry.getAttribute('aLabel');
      const li = Math.round(labels.getX(h.face.a));
      const fineAttr = h.object.geometry.getAttribute('aFine');
      const fine = fineAttr ? Math.round(fineAttr.getX(h.face.a)) : null;
      const netAttr = h.object.geometry.getAttribute('aNetwork');
      const network = netAttr ? Math.round(netAttr.getX(h.face.a)) : null;
      return { kind: 'cortex', id: li, point: h.point, fine, network,
               hemi: h.object.userData.hemi };
    }
    return null;
  }

  onPick(fn)  { this._onPick = fn; }
  onHover(fn) { this._onHover = fn; }

  /* ============================ LOADING ============================== */

  loadScene(scene) {
    this.sceneData = scene;
    const M = scene.meshes;

    // 32x1 RGBA texture: one texel per cortical label.
    // RGB = region colour, A = live activation glow driven by the simulation.
    this.regionData = new Uint8Array(32 * 4);
    for (let i = 0; i < 32; i++) {
      const c = new THREE.Color(REGION_COLORS[i] ?? 0x000000);
      this.regionData[i * 4]     = Math.round(c.r * 255);
      this.regionData[i * 4 + 1] = Math.round(c.g * 255);
      this.regionData[i * 4 + 2] = Math.round(c.b * 255);
      this.regionData[i * 4 + 3] = 0;
    }
    this.regionTex = new THREE.DataTexture(
      this.regionData, 32, 1, THREE.RGBAFormat);
    this.regionTex.magFilter = THREE.NearestFilter;
    this.regionTex.minFilter = THREE.NearestFilter;
    this.regionTex.needsUpdate = true;

    // 8x1 texture: one texel per Yeo network, index 0 being unassigned.
    // Colours are taken from the scene payload where possible, so the legend
    // in the sidebar and the paint on the surface cannot drift apart.
    const netMeta = scene.networks || [];
    const netData = new Uint8Array(8 * 4);
    for (let i = 0; i < 8; i++) {
      const fromScene = netMeta.find(n => n.id === i);
      const c = new THREE.Color(
        fromScene ? fromScene.color : (NETWORK_COLORS[i] ?? 0x000000));
      netData[i * 4]     = Math.round(c.r * 255);
      netData[i * 4 + 1] = Math.round(c.g * 255);
      netData[i * 4 + 2] = Math.round(c.b * 255);
      netData[i * 4 + 3] = 255;
    }
    this.networkTex = new THREE.DataTexture(netData, 8, 1, THREE.RGBAFormat);
    this.networkTex.magFilter = THREE.NearestFilter;
    this.networkTex.minFilter = THREE.NearestFilter;
    this.networkTex.needsUpdate = true;

    // 32x1 membership mask: R = 255 when that label is part of the current
    // selection. A single float uniform could only ever highlight one label,
    // which made composite terms like "prefrontal cortex" unrepresentable.
    this.selData = new Uint8Array(32 * 4);
    this.selTex = new THREE.DataTexture(this.selData, 32, 1, THREE.RGBAFormat);
    this.selTex.magFilter = THREE.NearestFilter;
    this.selTex.minFilter = THREE.NearestFilter;
    this.selTex.needsUpdate = true;

    // The procedural fallback surface carries no resting-state data, so the
    // network overlay is simply unavailable there rather than wrong.
    this.hasNetworks = !!(M.cortex_L && M.cortex_L.network);

    this.cortexUniforms = {
      uBase:   { value: new THREE.Color(0xd9bdb4) },
      uDeep:   { value: new THREE.Color(0x6b4a49) },
      uOpacity:{ value: 1.0 },
      uXray:   { value: 0.0 },
      uTime:   { value: 0.0 },
      uActive: { value: -1.0 },
      uDim:    { value: 0.0 },
      uGhostFade: { value: 0.0 },
      uSelLayer: { value: 0.0 },
      uGhostAlpha: { value: 1.0 },
      uHighlight: { value: 1.0 },
      uKeyDir: { value: this.keyDir },
      uRimColor: { value: new THREE.Color(0x8fa6bd) },
      uRegionTex: { value: this.regionTex },
      uSelTex:    { value: this.selTex },
      uNetTex:    { value: this.networkTex },
      uNetMode:   { value: 0.0 },
      uNetActive: { value: -1.0 },
      // Real anatomical surfaces ship measured sulcal depth; the procedural
      // fallback does not, and the shader approximates it instead.
      uHasSulc: { value: M.cortex_L && M.cortex_L.sulc ? 1.0 : 0.0 },
    };

    this.cortexL = this._makeCortex(M.cortex_L, 'L');
    this.cortexR = this._makeCortex(M.cortex_R, 'R');
    this.root.add(this.cortexL, this.cortexR);

    // Opaque companion pass for the highlighted parcel. Shares geometry and
    // every uniform object with the shell, so it needs no separate updating.
    this.selLayerUniforms = Object.assign({}, this.cortexUniforms,
                                          { uSelLayer: { value: 1.0 } });
    this.selL = this._makeSelLayer(this.cortexL);
    this.selR = this._makeSelLayer(this.cortexR);
    this.root.add(this.selL, this.selR);

    for (const s of scene.structures) {
      this.structMeta.set(s.id, s);
      const mesh = this._makeStructure(M[`struct_${s.id}`], s);
      this.structures.set(s.id, mesh);
      this.root.add(mesh);
    }

    this._buildPins();
    this.setView('lateral', false);
  }

  _geomFrom(enc, withLabels) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position',
      new THREE.BufferAttribute(decodeF32(enc.position), 3));
    g.setAttribute('normal',
      new THREE.BufferAttribute(decodeF32(enc.normal), 3));
    g.setIndex(new THREE.BufferAttribute(decodeU32(enc.index), 1));
    if (withLabels && enc.label) {
      const u8 = decodeU8(enc.label);
      g.setAttribute('aLabel',
        new THREE.BufferAttribute(Float32Array.from(u8), 1));
    }
    if (enc.sulc) {
      g.setAttribute('aSulc',
        new THREE.BufferAttribute(decodeF32(enc.sulc), 1));
    }
    if (enc.fine) {
      // Destrieux parcel id. Not used for shading - only so a click can
      // report the specific gyrus or sulcus that was hit.
      g.setAttribute('aFine',
        new THREE.BufferAttribute(Float32Array.from(decodeU16(enc.fine)), 1));
    }
    if (enc.network) {
      g.setAttribute('aNetwork',
        new THREE.BufferAttribute(Float32Array.from(decodeU8(enc.network)), 1));
    } else {
      // Keep the attribute present but all-zero so the shader compiles and
      // reads "unassigned" everywhere instead of sampling undefined memory.
      const n = g.getAttribute('position').count;
      g.setAttribute('aNetwork',
        new THREE.BufferAttribute(new Float32Array(n), 1));
    }
    g.computeBoundingSphere();
    return g;
  }

  _makeCortex(enc, hemi) {
    const mat = new THREE.ShaderMaterial({
      uniforms: this.cortexUniforms,
      vertexShader: CORTEX_VERT,
      fragmentShader: CORTEX_FRAG,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: true,
    });
    const m = new THREE.Mesh(this._geomFrom(enc, true), mat);
    m.userData = { hemi, kind: 'cortex' };
    m.renderOrder = 2;
    return m;
  }

  /** Opaque pass drawing only the selected parcel, so it writes depth. */
  _makeSelLayer(source) {
    const mat = new THREE.ShaderMaterial({
      uniforms: this.selLayerUniforms,
      vertexShader: CORTEX_VERT,
      fragmentShader: CORTEX_FRAG,
      transparent: false,
      side: THREE.DoubleSide,
      depthWrite: true,
    });
    const m = new THREE.Mesh(source.geometry, mat);
    m.userData = { hemi: source.userData.hemi, kind: 'cortexSel' };
    m.renderOrder = 1;          // before the shell, so the shell depth-tests
    m.visible = false;
    return m;
  }

  _makeStructure(enc, meta) {
    const col = new THREE.Color(meta.color);
    const mat = new THREE.MeshPhysicalMaterial({
      color: col,
      emissive: col,
      emissiveIntensity: 0.16,
      roughness: 0.42,
      metalness: 0.0,
      clearcoat: 0.55,
      clearcoatRoughness: 0.35,
      transmission: 0.0,
      transparent: true,
      opacity: meta.opacity ?? 1.0,
      side: THREE.FrontSide,
    });
    const m = new THREE.Mesh(this._geomFrom(enc, false), mat);
    m.userData = { id: meta.id, kind: 'structure', baseColor: col.clone(),
                   baseOpacity: meta.opacity ?? 1.0 };
    m.renderOrder = 3;
    return m;
  }

  /* -------------------------------------------------------------- pins */
  _buildPins() {
    const layer = document.getElementById('pinLayer');
    layer.innerHTML = '';
    for (const s of this.sceneData.structures) {
      if (s.order > 35) continue;             // skip scaffolding structures
      const el = document.createElement('div');
      el.className = 'pin';
      el.style.color = s.color;
      el.innerHTML = `<span class="knob"></span><span class="txt">${s.short}</span>`;
      el.onclick = () => this._onPick({ kind: 'structure', id: s.id });
      layer.appendChild(el);
      this.pins.set(s.id, {
        el, pos: new THREE.Vector3(...s.centroids[0]), meta: s,
      });
    }
  }

  _updatePins() {
    if (!this.pins.size) return;
    const v = new THREE.Vector3();
    const camPos = this.camera.position;
    for (const [id, pin] of this.pins) {
      const mesh = this.structures.get(id);
      const visible = this.showPins && this.showSubcort && mesh?.visible;
      if (!visible) { pin.el.style.display = 'none'; continue; }
      pin.el.style.display = '';
      v.copy(pin.pos).applyMatrix4(this.root.matrixWorld).project(this.camera);
      const x = (v.x * 0.5 + 0.5) * innerWidth;
      const y = (-v.y * 0.5 + 0.5) * innerHeight;
      pin.el.style.transform =
        `translate(-50%,-50%) translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
      // fade pins on the far side of the brain
      const world = pin.pos.clone().applyMatrix4(this.root.matrixWorld);
      const facing = world.normalize().dot(camPos.clone().normalize());
      pin.el.classList.toggle('faded', facing < -0.15 || v.z > 1);
    }
  }

  /* ============================= STATE =============================== */

  setMode(mode) {
    this.mode = mode;
    const anat = mode === 'anatomy';
    this.cortexUniforms.uOpacity.value = anat ? 1.0 : 0.92;
    // Circuits and simulation need to see inside the brain. They used to do
    // that with the separate x-ray path, which has its own alpha formula and
    // so looked nothing like a selection and ignored the appearance sliders.
    // They now use the same dissolve focus mode uses; x-ray goes back to
    // being purely the user's checkbox.
    this._modeGhost = !anat;
    this._applyGhostState();
    this._styleStructures();
    this.bloom.strength = anat ? 0.20 : 0.46;
  }

  /**
   * Single place that decides how see-through the cortex is.
   *
   * Three things can ask for it: a selection, circuits/simulation mode, and
   * an isolated network. They used to set the uniforms independently, so
   * whichever ran last won and switching modes with something selected
   * produced states neither of them intended.
   */
  _applyGhostState() {
    const sel = this._selState || { any: false };
    const dim = sel.any || this._modeGhost || this._netActive;
    // A network overlay paints the surface; there is nothing inside it to
    // reveal, so it dims without dissolving.
    const fade = sel.any || this._modeGhost;

    this.cortexUniforms.uDim.value = dim ? 1.0 : 0.0;
    this.cortexUniforms.uGhostFade.value = fade ? 1.0 : 0.0;

    const xray = !!this._xrayOn;
    for (const m of [this.cortexL, this.cortexR]) {
      if (m) m.material.depthWrite = !xray && !fade;
    }
    this._syncSelLayer();
  }

  setXray(on, soft = false) {
    this._xrayTarget = on ? 1 : 0;
    this._xrayOn = !!on;
    if (!soft) this.cortexUniforms.uXray.value = this._xrayTarget;
    for (const m of [this.cortexL, this.cortexR]) {
      if (m) m.material.needsUpdate = true;
    }
    // Depth writing is decided in one place, because a dissolved shell needs
    // it off for the same reason x-ray does.
    this._applyGhostState();
    // Structure opacity is owned by _styleStructures, which knows about the
    // selection and the appearance sliders; this only changes the baseline.
    this._styleStructures();
  }

  setHemisphere(h) {
    this.hemi = h;
    if (this.cortexL) this.cortexL.visible = (h === 'both' || h === 'L');
    if (this.cortexR) this.cortexR.visible = (h === 'both' || h === 'R');
    this._syncSelLayer();
  }

  /** The opaque selection pass mirrors the shell, and only runs while focusing. */
  _syncSelLayer() {
    const on = this.cortexUniforms.uDim.value > 0.5;
    if (this.selL) this.selL.visible = on && !!this.cortexL?.visible;
    if (this.selR) this.selR.visible = on && !!this.cortexR?.visible;
  }

  setSubcortical(on) {
    this.showSubcort = on;
    for (const [id, m] of this.structures) m.visible = this._structVisible(id);
  }

  /** Three independent reasons a structure may be hidden, in one place. */
  _structVisible(id) {
    return this.showSubcort
        && !this._hiddenStructs?.has(id)
        && !this._mutedStructs?.has(id);
  }

  toggleStructure(id, on) {
    this._hiddenStructs = this._hiddenStructs || new Set();
    if (on) this._hiddenStructs.delete(id); else this._hiddenStructs.add(id);
    const m = this.structures.get(id);
    if (m) m.visible = this._structVisible(id);
  }

  setPins(on) { this.showPins = on; }  setAutoRotate(on) { this.controls.autoRotate = on; }

  /**
   * Colour the cortex by resting-state functional network instead of by
   * anatomical parcel. Returns false if the loaded geometry has no network
   * data, which is the case for the procedural fallback surface.
   */
  setNetworkMode(on) {
    if (on && !this.hasNetworks) return false;
    this.networkMode = !!on;
    this.cortexUniforms.uNetMode.value = on ? 1.0 : 0.0;
    if (!on) this.selectNetwork(null);
    return true;
  }

  /** Isolate one Yeo network (1..7), or null for all of them at once. */
  selectNetwork(id) {
    const active = (id === null || id === undefined) ? -1 : id;
    this.activeNetwork = active < 0 ? null : active;
    this.cortexUniforms.uNetActive.value = active;
    this._netActive = active >= 0;
    this._applyGhostState();
  }

  /** Highlight a cortical label index or a structure id. */
  select(hit) {
    if (!hit) return this.selectMany([], []);
    if (hit.kind === 'cortex') return this.selectMany([hit.id], [], hit);
    return this.selectMany([], [hit.id], hit);
  }

  /**
   * Highlight any set of cortical labels and structures at once.
   *
   * Single selection is just the one-element case. Composite anatomical
   * terms - "prefrontal cortex", "dorsal striatum" - are collections rather
   * than parcels, and this is what lets them be shown as what they are
   * instead of being drawn as a single invented region.
   */
  selectMany(labelIndices, structureIds, hit = null) {
    const labels = new Set(labelIndices);
    const structs = new Set(structureIds);
    const any = labels.size > 0 || structs.size > 0;
    this.selected = hit || (any ? { kind: 'group' } : null);

    // Buried cortex used to force x-ray on, because selecting the insula on
    // an opaque brain highlighted nothing. The shell now dissolves for every
    // selection, so that is already handled - and forcing x-ray on top of it
    // made the insula the one region that rendered at a different opacity.
    // Measured: the insula is MORE visible without it.

    this.selData.fill(0);
    for (const i of labels) {
      if (i >= 0 && i < 32) this.selData[i * 4] = 255;
    }
    this.selTex.needsUpdate = true;

    // Once the shell dissolves, every faded structure inside shows through it
    // at once. That is useful context when the selection IS a structure, and
    // pure clutter when it is a patch of cortex - fifteen of them stacked
    // compete with the parcel you actually picked.
    const cortexOnly = any && structs.size === 0;
    this._mutedStructs = new Set(
      cortexOnly
        ? [...this.structures.keys()].filter(id => !SILHOUETTE.has(id))
        : []);

    // Kept so the appearance sliders can restyle without a re-selection.
    this._selState = { structs, any, cortexOnly };
    this._applyGhostState();
    this._styleStructures();
  }

  _styleStructures() {
    const { structs, any, cortexOnly } = this._selState
      || { structs: new Set(), any: false, cortexOnly: false };
    const xray = this.cortexUniforms.uXray.value > 0.5;
    const ga = this.cortexUniforms.uGhostAlpha.value;
    // Structures are MeshPhysicalMaterial, so their "highlight" is emissive
    // intensity rather than the shader's uHighlight. Scale it here or the
    // slider silently does nothing to anything in the STRUCTURES list.
    const hl = this.cortexUniforms.uHighlight.value;

    for (const [id, m] of this.structures) {
      const on = structs.has(id);
      const meta = this.structMeta.get(id);
      m.material.emissiveIntensity = on
        ? 1.15 * hl : (xray ? 0.34 : 0.16) * ga;
      // Unselected structures fade almost out too, so the selection is the
      // only solid thing on screen. Kept low because ~15 translucent shells
      // with depthWrite off stack additively and read far stronger than a
      // single one at the same alpha. The silhouette pair is allowed more,
      // so the specimen still reads as one body.
      //
      // Circuits and simulation dim them far less: nothing is selected there,
      // and these are the structures the pathways actually run between, so
      // they are the subject rather than clutter. Without this the cortex
      // went to glass while they stayed solid, and the brain as a whole did
      // not read as transparent at all.
      const modeOnly = !any && this._modeGhost;
      const fade = modeOnly
        ? 0.45 * ga
        : ((cortexOnly && SILHOUETTE.has(id)) ? 0.16 : 0.03) * ga;
      const dim = modeOnly || (any && !on);
      m.material.opacity = dim
        ? Math.min(1, (meta.opacity ?? 1) * fade) : (meta.opacity ?? 1);
      m.material.depthWrite = !dim;
      m.userData.pulse = on;
      m.visible = this._structVisible(id);

      // Kept grey while the cortex is ghosted; a saturated cerebellum beside
      // desaturated cortex stops reading as the same specimen.
      const base = m.userData.baseColor;
      if (base) {
        const mute = cortexOnly && SILHOUETTE.has(id) ? 0.82 : 0.0;
        m.material.color.copy(base).lerp(GHOST_GREY, mute);
        m.material.emissive.copy(base).lerp(GHOST_GREY, mute);
      }
    }
    for (const [id, pin] of this.pins) {
      // _updatePins() hides the pin of any structure that is not visible, so
      // muted ones need no handling here.
      pin.el.classList.toggle('faded', any && !structs.has(id));
    }
  }

  /** How visible everything that is NOT selected stays. 1.0 = default look. */
  setGhostAlpha(v) {
    this.cortexUniforms.uGhostAlpha.value = v;
    this._styleStructures();
  }

  /** How strongly the selected region reads. 1.0 = default look. */
  setHighlight(v) {
    this.cortexUniforms.uHighlight.value = v;
    this._styleStructures();
  }

  /** Per-region activation glow, 0..1, driven by the simulation. */
  setRegionGlow(labelIndex, v) {
    if (labelIndex < 0 || labelIndex > 31) return;
    this.regionData[labelIndex * 4 + 3] =
      Math.round(Math.max(0, Math.min(1, v)) * 255);
    this.regionTex.needsUpdate = true;
  }

  setStructureGlow(id, v) {
    const m = this.structures.get(id);
    if (m) m.userData.glow = v;
  }

  clearGlow() {
    for (let i = 0; i < 32; i++) this.regionData[i * 4 + 3] = 0;
    this.regionTex.needsUpdate = true;
    for (const m of this.structures.values()) m.userData.glow = 0;
  }

  /* ------------------------------------------------------- view presets */
  setView(name, animate = true) {
    const D = 330;
    const V = {
      lateral:   [D * 0.92, D * 0.12, D * 0.30],
      medial:    [-D * 0.18, D * 0.10, D * 0.02],
      superior:  [0.1, D, 0.1],
      inferior:  [0.1, -D, 0.1],
      // RAS +y (anterior) becomes three.js -z after the root rotation
      anterior:  [0, D * 0.14, -D],
      posterior: [0, D * 0.14, D],
    };
    const p = V[name] || V.lateral;
    if (name === 'medial' && this.hemi === 'both') {
      this.setHemisphere('R');
      document.querySelectorAll('#hemiSeg button').forEach(b =>
        b.classList.toggle('active', b.dataset.hemi === 'R'));
    }
    const target = new THREE.Vector3(...p);
    if (name === 'medial') target.set(-D * 0.95, D * 0.05, 0);
    if (!animate) { this.camera.position.copy(target); return; }
    this._camTween = {
      from: this.camera.position.clone(), to: target,
      t: 0, dur: 0.85,
    };
  }

  resetCamera() { this.setView('lateral'); this.controls.target.set(0, 0, 0); }

  /* ============================== LOOP =============================== */

  _resize() {
    this.camera.aspect = innerWidth / innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(innerWidth, innerHeight);
    this.composer.setSize(innerWidth, innerHeight);
  }

  render(dt, t) {
    // camera tween
    if (this._camTween) {
      const c = this._camTween;
      c.t = Math.min(1, c.t + dt / c.dur);
      const e = 1 - Math.pow(1 - c.t, 4);            // quartic ease-out
      this.camera.position.lerpVectors(c.from, c.to, e);
      if (c.t >= 1) this._camTween = null;
    }
    // soft x-ray transition
    if (this._xrayTarget !== undefined) {
      const u = this.cortexUniforms.uXray;
      u.value += (this._xrayTarget - u.value) * Math.min(1, dt * 5.5);
    }
    this.cortexUniforms.uTime.value = t;


    // structure pulse + glow
    const hl = this.cortexUniforms.uHighlight.value;
    const ga = this.cortexUniforms.uGhostAlpha.value;
    const focusing = this.cortexUniforms.uDim.value > 0.5;
    for (const m of this.structures.values()) {
      const g = m.userData.glow || 0;
      const on = m.userData.pulse;
      const pulse = on ? 0.55 + 0.45 * Math.sin(t * 3.4) : 0;
      const base = this.cortexUniforms.uXray.value > 0.5 ? 0.34 : 0.16;
      // The appearance sliders must be applied here, not only where the
      // selection is styled: this loop rewrites emissiveIntensity every
      // frame and would otherwise pull it straight back to the default.
      const want = on
        ? (base + g * 1.5 + pulse * 0.7) * hl
        : (base + g * 1.5) * (focusing ? ga : 1.0);
      m.material.emissiveIntensity +=
        (want - m.material.emissiveIntensity) * Math.min(1, dt * 8);
      if (g > 0.01) {
        const s = 1 + g * 0.07 * (0.6 + 0.4 * Math.sin(t * 5));
        m.scale.setScalar(s);
      } else if (m.scale.x !== 1) {
        m.scale.lerp(new THREE.Vector3(1, 1, 1), Math.min(1, dt * 6));
      }
    }

    this.controls.update();
    this._updatePins();
    this.composer.render();

    // Pixel measurement has to happen here, immediately after the draw and
    // before the browser presents the canvas. Once the frame is presented the
    // drawing buffer is cleared, and a later readPixels returns all zeros -
    // which reads as a perfectly dark brain rather than as a failed measure.
    if (this._pendingMeasure) {
      const done = this._pendingMeasure;
      this._pendingMeasure = null;
      done(this._readLitFraction());
    }
  }

  /** Fraction of the canvas carrying meaningful light. Returns a promise
   *  because the read is deferred to the end of the next rendered frame. */
  measureLitFraction() {
    return new Promise(res => { this._pendingMeasure = res; });
  }

  _readLitFraction() {
    const gl = this.renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let n = 0;
    for (let i = 0; i < px.length; i += 4) {
      if (px[i] + px[i + 1] + px[i + 2] > 96) n++;
    }
    return n / (w * h);
  }
}
