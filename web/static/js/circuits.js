/* ============================================================================
 * circuits.js — glowing functional pathways drawn inside the brain.
 *
 * WHAT THESE ARE:
 *   A visualisation of a *functional relationship* between two structures,
 *   with brightness and thickness driven by a simulation variable.
 *
 * WHAT THESE ARE NOT:
 *   Tractography. These curves do not follow the real course of any
 *   white-matter bundle, and several of the relationships they depict are
 *   polysynaptic — relayed through structures that are not drawn at all.
 * ========================================================================= */

import * as THREE from 'three';

const TUBE_VERT = /* glsl */`
  uniform float uRadius;
  varying vec2  vUv;
  varying vec3  vN;
  varying vec3  vView;

  void main() {
    vUv = uv;
    vN  = normalize(normalMatrix * normal);
    // TubeGeometry is generated at radius 1, so the vertex normal is the
    // radial direction. Displacing along it re-radiuses the tube on the GPU,
    // which lets connection strength drive thickness every frame.
    vec3 p = position + normal * (uRadius - 1.0);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

const TUBE_FRAG = /* glsl */`
  precision highp float;
  varying vec2  vUv;
  varying vec3  vN;
  varying vec3  vView;

  uniform vec3  uColor;
  uniform float uTime;
  uniform float uStrength;   // simulated pathway strength, 0..1
  uniform float uFlow;       // flow speed multiplier
  uniform float uGhost;      // 1 = target-state ghost rendering
  uniform float uFocus;      // 1 = highlighted, 0.25 = dimmed by selection
  uniform float uPulse;      // one-shot travelling pulse, 0..1 position
  uniform float uPulseOn;

  void main() {
    float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vView))), 1.6);

    // travelling flow bands — "this route is easy to run"
    float speed = 0.25 + uStrength * 0.85;
    float band  = sin(vUv.x * 34.0 - uTime * speed * 6.0);
    band = pow(max(band, 0.0), 5.0);

    // core brightness rises with simulated strength
    float core = 0.24 + uStrength * 0.85;
    vec3 col = uColor * (core + band * uStrength * 1.5);
    col += uColor * fres * 0.75;

    // one-shot cascade pulse
    if (uPulseOn > 0.5) {
      float d = abs(vUv.x - uPulse);
      float p = exp(-d * d * 260.0);
      col += (uColor * 1.6 + vec3(0.55)) * p * 2.4;
    }

    float a = (0.10 + uStrength * 0.72) * uFocus;
    if (uGhost > 0.5) {
      // dashed, cool, unmistakably "not the current state"
      float dash = step(0.5, fract(vUv.x * 26.0 - uTime * 0.25));
      a *= 0.34 * dash;
      col = mix(col, vec3(0.45, 0.78, 1.0), 0.55);
    }
    a += band * uStrength * 0.30 * uFocus;

    gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
  }
`;

/* ========================================================================= */

export class Circuits {
  constructor(viewer) {
    this.viewer = viewer;
    this.group = new THREE.Group();
    this.ghostGroup = new THREE.Group();
    viewer.root.add(this.group, this.ghostGroup);
    this.edges = new Map();       // edgeId -> [{mesh, mat, side}]
    this.ghosts = new Map();
    this.pathwayColors = {};
    this.visible = false;
    this.showGhost = false;
    this._pulses = [];
  }

  /* --------------------------------------------------------------- build */
  build(model, scene = {}) {
    this.model = model;
    // Routes measured through white matter, when the anatomy is present.
    this.paths = scene.edge_paths || {};
    const measured = scene.anchors_measured || {};
    const anchors = {};
    // Endpoints from the built geometry beat the hand-typed table: a curve
    // then starts on the structure the panel highlights, not near it.
    for (const a of model.anchors) {
      anchors[a.id] = measured[a.id] ? { ...a, right: measured[a.id] } : a;
    }
    for (const p of model.pathways) this.pathwayColors[p.id] = p.color;

    for (const e of model.edges) {
      const A = anchors[e.src], B = anchors[e.dst];
      if (!A || !B) continue;
      const col = new THREE.Color(this.pathwayColors[e.pathway]);
      const made = [];
      const ghosts = [];

      for (const side of [1, -1]) {
        const curve = this._curve(A, B, e.curve, side, e.id);
        const geo = new THREE.TubeGeometry(curve, 58, 1.0, 9, false);

        const mat = new THREE.ShaderMaterial({
          uniforms: {
            uColor:  { value: col.clone() },
            uTime:   { value: 0 },
            uStrength: { value: 0.2 },
            uFlow:   { value: 1 },
            uGhost:  { value: 0 },
            uFocus:  { value: 1 },
            uRadius: { value: 1.2 },
            uPulse:  { value: 0 },
            uPulseOn:{ value: 0 },
          },
          vertexShader: TUBE_VERT,
          fragmentShader: TUBE_FRAG,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 6;
        mesh.userData = { edge: e.id, side };
        this.group.add(mesh);
        made.push({ mesh, mat, side, curve });

        // ghost copy for the TARGET state overlay
        const gmat = mat.clone();
        gmat.uniforms.uGhost.value = 1;
        gmat.uniforms.uRadius.value = 1.0;
        const gmesh = new THREE.Mesh(geo, gmat);
        gmesh.renderOrder = 5;
        this.ghostGroup.add(gmesh);
        ghosts.push({ mesh: gmesh, mat: gmat, side });
      }
      this.edges.set(e.id, made);
      this.ghosts.set(e.id, ghosts);
    }
    this.group.visible = false;
    this.ghostGroup.visible = false;
  }

  /** Build a curve that arcs *through* the brain rather than across it. */
  _curve(A, B, bow, side, id) {
    // Circuits live under viewer.root, which already applies the
    // RAS -> three.js rotation, so we work in raw millimetres here.

    // A route measured through white matter, if one was published. The
    // right hemisphere is what gets routed; the left is its mirror.
    const pts = this.paths?.[id];
    if (pts && pts.length > 2) {
      return new THREE.CatmullRomCurve3(
        pts.map(p => new THREE.Vector3(p[0] * side, p[1], p[2])),
        false, 'centripetal', 0.5);
    }

    const mirror = (a) => new THREE.Vector3(
      a.midline ? a.right[0] * (side > 0 ? 0.6 : -0.6) : a.right[0] * side,
      a.right[1],
      a.right[2]);

    const p0 = mirror(A);
    const p2 = mirror(B);
    const mid = p0.clone().add(p2).multiplyScalar(0.5);

    // Bow the curve outward from the brain's centre so pathways spread
    // apart and read as distinct arcs instead of overlapping straight lines.
    const out = mid.clone().normalize();
    const dist = p0.distanceTo(p2);
    mid.addScaledVector(out, dist * (0.16 + bow * 0.9));
    // slight lateral offset keeps reciprocal pairs from overlapping exactly
    mid.x += bow * dist * 0.30 * side;

    return new THREE.QuadraticBezierCurve3(p0, mid, p2);
  }

  /**
   * Find the edge nearest the ray.
   *
   * Deliberately not a mesh raycast. The tube geometry has a fixed radius of
   * 1.0 and the shader inflates it to as much as ~3.8 to show strength, so a
   * geometry raycast would miss most of what the user can see and hit - the
   * visual and the hitbox would simply disagree. Sampling the curve and
   * comparing against the *current* rendered radius keeps them consistent.
   */
  pickEdge(ray) {
    if (!this.visible) return null;
    const p = new THREE.Vector3();
    let best = null;

    for (const [id, arr] of this.edges) {
      for (const rec of arr) {
        if (!rec.mesh.visible || !rec.curve) continue;
        // Focused-out pathways are nearly invisible; do not let them
        // swallow clicks meant for the ones on screen.
        if (rec.mat.uniforms.uFocus.value < 0.3) continue;
        const rad = rec.mat.uniforms.uRadius.value;
        const tol = Math.max(2.5, rad * 1.6);

        for (let i = 0; i <= 24; i++) {
          rec.curve.getPoint(i / 24, p);
          this.group.localToWorld(p);
          const d = ray.distanceToPoint(p);
          if (d > tol) continue;
          const along = p.distanceTo(ray.origin);
          if (!best || along < best.along) best = { id, along };
        }
      }
    }
    return best ? best.id : null;
  }

  /**
   * Paint every edge by how much it changed between two days.
   *
   * Colour carries direction (green = stronger, amber = weaker) and radius
   * carries magnitude, so a pathway that barely moved stays thin and grey
   * instead of shouting. Pass null to return to live colours.
   *
   * The scale is fixed at +/-0.35 rather than normalised to the largest
   * change present. Auto-normalising would make the biggest mover look
   * dramatic even on a day when nothing really happened.
   */
  showDelta(from, to) {
    this.deltaMode = !!(from && to);
    for (const [id, arr] of this.edges) {
      const base = new THREE.Color(this.pathwayColors[
        this.model.edges.find(e => e.id === id)?.pathway]);
      let col = base, mag = null;
      if (this.deltaMode) {
        const d = (to[id] ?? 0) - (from[id] ?? 0);
        mag = Math.min(1, Math.abs(d) / 0.35);
        col = new THREE.Color(d >= 0 ? 0x5eead4 : 0xffb454)
          .lerp(new THREE.Color(0x4a5058), 1 - mag);
      }
      for (const { mat } of arr) {
        mat.uniforms.uColor.value.copy(col);
        if (this.deltaMode) {
          mat.uniforms.uRadius.value = 0.45 + mag * 3.4;
          mat.uniforms.uStrength.value = 0.25 + mag * 0.75;
        }
      }
    }
  }

  /* -------------------------------------------------------------- state */
  setVisible(on) {
    this.visible = on;
    this.group.visible = on;
    this.ghostGroup.visible = on && this.showGhost;
  }

  setGhost(on) {
    this.showGhost = on;
    this.ghostGroup.visible = on && this.visible;
  }

  /** Drive thickness + brightness from the simulation's weights. */
  update(weights, targets = null) {
    // In comparison mode radius and brightness encode the *change* between
    // two days, so the live values must not overwrite them.
    if (this.deltaMode) return;
    for (const [id, arr] of this.edges) {
      const w = weights[id] ?? 0;
      for (const { mat } of arr) {
        mat.uniforms.uStrength.value = w;
        // thickness is a gentle power curve so weak pathways stay visible
        mat.uniforms.uRadius.value = 0.45 + Math.pow(w, 1.25) * 3.4;
      }
      if (targets) {
        const t = targets[id] ?? 0;
        for (const { mat } of this.ghosts.get(id)) {
          mat.uniforms.uStrength.value = t;
          mat.uniforms.uRadius.value = 0.45 + Math.pow(t, 1.25) * 3.4;
        }
      }
    }
  }

  /**
   * Dim everything except the given pathway ids (or clear with null).
   *
   * `on` may exceed 1 to push the engaged edges above their normal alpha:
   * contrast built only by dimming makes the whole scene darker, which reads
   * as "the brain went dim" rather than "these lit up".
   */
  focus(pathwayIds, on = 1.0, off = 0.16) {
    for (const e of this.model.edges) {
      const lit = !pathwayIds || pathwayIds.includes(e.pathway);
      for (const { mat } of this.edges.get(e.id) || []) {
        mat.uniforms.uFocus.value = lit ? on : off;
      }
    }
  }

  /** Fire a travelling pulse along a sequence of edges — the cascade. */
  cascade(edgeIds, { stagger = 0.28, speed = 1.5 } = {}) {
    edgeIds.forEach((id, i) => {
      this._pulses.push({ id, t: -i * stagger, speed });
    });
  }

  clearPulses() {
    this._pulses = [];
    for (const arr of this.edges.values())
      for (const { mat } of arr) mat.uniforms.uPulseOn.value = 0;
  }

  /* --------------------------------------------------------------- loop */
  render(dt, t) {
    for (const arr of this.edges.values())
      for (const { mat } of arr) mat.uniforms.uTime.value = t;
    for (const arr of this.ghosts.values())
      for (const { mat } of arr) mat.uniforms.uTime.value = t;

    if (!this._pulses.length) return;
    const alive = [];
    const active = new Set();
    for (const p of this._pulses) {
      p.t += dt * p.speed;
      if (p.t > 1.15) continue;
      alive.push(p);
      if (p.t < 0) continue;
      active.add(p.id);
      for (const { mat } of this.edges.get(p.id) || []) {
        mat.uniforms.uPulseOn.value = 1;
        mat.uniforms.uPulse.value = p.t;
      }
    }
    for (const [id, arr] of this.edges) {
      if (!active.has(id))
        for (const { mat } of arr) mat.uniforms.uPulseOn.value = 0;
    }
    this._pulses = alive;
  }
}
