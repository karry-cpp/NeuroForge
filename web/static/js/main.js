/* ============================================================================
 * main.js — application orchestration.
 *
 * Holds no simulation logic. It fetches state from Python, hands geometry to
 * the Viewer, hands weights to the Circuits, and wires the DOM.
 * ========================================================================= */

import { API } from './api.js';
import { Viewer } from './viewer.js';
import { Circuits } from './circuits.js';
import { Chat } from './chat.js';
import {
  LogDock, Modal, Panel, Timeline, analysisHtml, choicesHtml, edgePanel,
  groupPanel, honestyHtml, interpretingHtml, interpreterHelpHtml,
  networkPanel, proposalsHtml, regionPanel, resultHtml, toast,
  triggerModalHtml,
} from './ui.js';

const $  = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// Unrelated pathways drop to this; the engaged ones sit above their normal
// alpha so the result reads as "these lit up", not "the brain went dark".
const FOCUS_ON = 1.6;
const FOCUS_OFF = 0.10;

/* ------------------------------------------------------------------------
 * Sidebar focus.
 *
 * A curated shortlist of the regions this simulation actually has rules for,
 * so the dock is not thirty rows of anatomy that no logged event ever
 * touches.
 *
 * This is a VIEW filter only. Nothing is removed from the scene, the atlas or
 * the model. The hidden structures still render, are still clickable on the
 * brain itself, and still anchor pathways - caudate, putamen and accumbens
 * carry the entire habit-formation story, which is the idea the application
 * is built around. "Full" restores every row.
 * ---------------------------------------------------------------------- */
const FOCUS = {
  cortex: ['mPFC', 'vmPFC', 'OFC', 'dlPFC', 'vlPFC', 'dACC', 'sgACC', 'PCC',
           'aINS'],
  structures: ['amygdala', 'hippocampus', 'acc', 'sgacc', 'caudate', 'putamen',
               'accumbens', 'thalamus', 'vta', 'brainstem'],
  groups: ['PFC', 'latPFC'],
};

const inFocus = (kind, id) => App.focus === 'all' || FOCUS[kind].includes(id);

/* Visualise keeps only the structures this app has an emotional-regulation
 * story for. With all fourteen present the middle of the brain was a pile of
 * overlapping lumps under eleven pins.
 *
 * The striatum is caudate + putamen because that is what it is here: there is
 * no single striatum mesh, and structures.py is explicit that the dorsal /
 * ventral split is a drawing convention rather than a boundary in tissue.
 * Cerebellum earns its place by anchoring the bottom of the specimen - without
 * it the brain reads as floating.
 *
 * The cortical half of the same shortlist is FOCUS.cortex: mPFC, vmPFC, OFC,
 * dlPFC, vlPFC, dACC, sgACC, PCC, aINS. Parcels are part of the cortex mesh, so they need no
 * cast - they simply appear when selected. */
const VIS_CAST = ['amygdala', 'hippocampus', 'acc', 'sgacc',
                  'caudate', 'putamen',
                  'brainstem', 'cerebellum'];
const VIS_MAX_PARTS = 3;
// Visualise glow by involvement level 0..1: the busiest part glows most,
// golden white. Focus lifts one part to the top.
const VIS_GLOW_MIN = 0.12, VIS_GLOW_SPAN = 0.5, VIS_GLOW_FOCUS = 0.75;
const visGlow = (id, focus) => id === focus ? VIS_GLOW_FOCUS
  : VIS_GLOW_MIN + VIS_GLOW_SPAN * (App.visLevels[id] ?? 0.25);

const App = {
  scene: null, model: null, state: null,
  mode: 'anatomy', viewState: 'current', replayDay: null,
  playing: false, playT: 0,
  focus: 'regulation',
  visSeq: 0,
  visParts: [], visFocus: null, visLevels: {},
  visMoves: [], visAlts: [], visEvents: [], visPreview: null,
  visReveal: Infinity, visTimers: [],
  curFocus: null, curNote: null, disputePart: null, intent: 'usual',
  // Change over time: null | 'since' | a horizon in days.
  pathView: null, pathFocus: null, pathDrawn: null, projScenario: 'keep',
  projection: null, projLoading: null, projFailed: null, logSeq: 0,
  viewer2: null, camLead: 1,
  eventById: {}, nodeById: {}, cortexById: {}, cortexByIndex: {},
  structById: {},
};
window.NF = App;                       // handy for debugging in the console
// Exposed so the headless smoke test can drive the same code paths a user
// would, without depending on where the brain happens to land on screen.
window.NF.debug = { showCortex: (...a) => showCortex(...a),
                    showStructure: (...a) => showStructure(...a),
                    showNetwork: (id) => showNetwork(id),
                    showGroup: (id) => {
                      const g = (App.scene.groups || [])
                        .find(x => x.id === id);
                      if (g) showGroup(g);
                      return !!g;
                    },
                    litFraction: () => litFraction(),
                    showEdge: (id) => showEdge(id),
                    compare: (a, b) => {
                      refreshCompareOptions();
                      $('#cmpFrom').value = String(a);
                      $('#cmpTo').value = String(b);
                      runCompare();
                      return !!App.compareOn;
                    },
                    endCompare: () => endCompare(),
                    setFocus: (f) => { App.focus = f; buildDock();
                                       return App.focus; },
                    dockRows: () => ({
                      cortex: $$('#cortexList .structrow').length,
                      structures: $$('#structList .structrow').length,
                      groups: $$('#groupList .structrow').length,
                    }),
                    clickCortexRow: (id) => {
                      const r = App.cortexById[id];
                      const row = r && $(`#cortexList [data-cortex="`
                        + `${r.label_index}"]`);
                      if (row) row.click();
                      return !!row;
                    },
                    fineName: (id) => fineName(id) };

/** Fraction of canvas pixels that are meaningfully lit. Used by the smoke
 *  test to check that "focus" actually removes ink rather than adding it. */
function litFraction() {
  return App.viewer.measureLitFraction();
}

/* ======================================================================= */
/*  BOOT                                                                    */
/* ======================================================================= */

async function boot() {
  const msg = $('#loaderMsg'), bar = $('#loaderBar');
  const step = (t, p) => { msg.textContent = t; bar.style.width = p + '%'; };

  try {
    step('contacting the simulation engine…', 8);
    App.model = await API.model();

    step('receiving cortical surface…', 26);
    App.scene = await API.scene();

    step('building 3-D scene…', 58);
    App.viewer = new Viewer($('#gl'));
    App.viewer.loadScene(App.scene);

    step('laying out functional pathways…', 76);
  App.circuits = new Circuits(App.viewer);
  App.circuits.build(App.model, App.scene);
  // The viewer owns picking, so it needs to know about the circuits to be
  // able to hit-test them. Without this the `this.circuits?.` guard in
  // _pick() is always falsy and pathways are silently unclickable.
  App.viewer.circuits = App.circuits;    step('reading simulation state…', 90);
    App.state = await API.state();

    index();
    wire();
    applyState();
    setMode('anatomy');

    // Chat needs the scene indexed first: highlighting a region by id goes
    // through the same lookup tables the panels use.
    App.chat = new Chat({
      onShow: (ids) => chatHighlight(ids),
      onLog: (id, intensity, note) => doLog(id, intensity, note),
      onClose: () => chatUnlabel(),
      labelOf: (id) => {
        const p = App.structById[id] || App.cortexById[id];
        return p?.label || partLabel(id);
      },
    });
    await App.chat.init();

    step('ready', 100);
    setTimeout(() => $('#loader').classList.add('done'), 420);
    startLoop();

    setTimeout(() => toast(
      'Drag to rotate · scroll to zoom · click any structure',
      'Everything shown is a simulation, not a measurement.'), 1400);
  } catch (err) {
    msg.innerHTML = `<span style="color:#ff5c6e">${err.message}</span><br>
      <span style="font-size:10px">Check the terminal running the server.</span>`;
    console.error(err);
  }
}

function index() {
  for (const e of App.model.events) App.eventById[e.id] = e;
  for (const n of App.model.nodes) App.nodeById[n.id] = n;
  for (const r of App.scene.cortical_regions) {
    App.cortexById[r.id] = r;
    App.cortexByIndex[r.label_index] = r;
  }
  for (const s of App.scene.structures) App.structById[s.id] = s;
  window.__NF_EVENTCAT = Object.fromEntries(
    App.model.events.map(e => [e.id, e.category]));
}

/* ======================================================================= */
/*  WIRING                                                                  */
/* ======================================================================= */

function wire() {
  const V = App.viewer;

  /* ---- picking ---- */
  V.onPick((hit) => {
    // Current's engagement view is a reading, not a selection: clicks inspect.
    if (currentLocked()) {
      if (!hit) {
        if (App.curFocus || App.curNote != null) focusCurrentPart(null);
        return;
      }
      if (hit.kind === 'edge') return;
      const id = hit.kind === 'structure'
        ? hit.id : App.cortexByIndex[hit.id]?.id;
      if (engagedRows().some(r => r.part === id)) focusCurrentPart(id);
      if (hit.kind === 'structure') showStructure(hit.id, true);
      else showCortex(hit.id, true, hit.fine, hit.network);
      return;
    }
    // Visualise highlights stay until Reset. A click focuses a lit part; the
    // panel stays shut, since the explanation already says what each part does.
    if (visualiseLocked()) {
      if (!hit) { focusVisualisePart(null); return; }
      if (hit.kind === 'edge') return;
      const id = hit.kind === 'structure'
        ? hit.id : App.cortexByIndex[hit.id]?.id;
      if (App.visParts.includes(id)) {
        focusVisualisePart(App.visFocus === id ? null : id);
      }
      return;
    }
    if (!hit) { clearSelection(); Panel.close(); return; }
    if (hit.kind === 'edge') { showEdge(hit.id); return; }
    V.select(hit);
    markDockRow(hit.kind === 'structure'
      ? $(`#structList .structrow[data-id="${hit.id}"]`)
      : $(`#cortexList .structrow[data-cortex="${hit.id}"]`));
    if (App.mode === 'anatomy') {
      const id = hit.kind === 'structure'
        ? hit.id : App.cortexByIndex[hit.id]?.id;
      if (id) V.setPinFilter([id]);
    }
    if (hit.kind === 'structure') showStructure(hit.id);
    else showCortex(hit.id, false, hit.fine, hit.network);
  });

  V.onHover((hit, e) => {
    const tip = $('#hoverTip');
    document.body.style.cursor = hit ? 'pointer' : '';
    if (!hit) { tip.classList.add('hidden'); return; }
    let name, sub;
    if (hit.kind === 'structure') {
      const s = App.structById[hit.id];
      name = s.name; sub = s.system;
    } else {
      const r = App.cortexByIndex[hit.id];
      if (!r) { tip.classList.add('hidden'); return; }
      name = r.name;
      // With the real atlas we can name the exact fold under the cursor,
      // which is far more informative than the coarse functional label.
      const f = fineName(hit.fine);
      sub = f ? f : 'cortical region';
    }
    tip.innerHTML = `<div class="h">${name}</div><div class="s">${sub}</div>`;
    tip.style.left = e.clientX + 'px';
    tip.style.top = e.clientY + 'px';
    tip.classList.remove('hidden');
  });

  /* ---- modes ---- */
  $$('#modebar button').forEach(b =>
    b.onclick = () => setMode(b.dataset.mode));

  /* ---- views ---- */
  $$('.viewbtn').forEach(b => b.onclick = () => {
    $$('.viewbtn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    V.setView(b.dataset.view);
  });

  /* ---- hemisphere ---- */
  $$('#hemiSeg button').forEach(b => b.onclick = () => {
    $$('#hemiSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    V.setHemisphere(b.dataset.hemi);
  });

  /* ---- toggles ---- */
  $('#chkXray').onchange    = (e) => V.setXray(e.target.checked, true);
  $('#chkSubcort').onchange = (e) => V.setSubcortical(e.target.checked);
  $('#chkLabels').onchange  = (e) => V.setPins(e.target.checked);
  $('#chkSpin').onchange    = (e) => V.setAutoRotate(e.target.checked);

  /* ---- appearance sliders ---- */
  wireLookControls();

  /* ---- structure list ---- */
  buildDock();

  $$('#focusSeg button').forEach(b => b.onclick = () => {
    $$('#focusSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    App.focus = b.dataset.focus;
    App.viewer.select(null);
    Panel.close();
    buildDock();
  });

  /* ---- state toggle ---- */
  $$('#stateSeg button').forEach(b => b.onclick = () => selectState(b.dataset.state));

  wireComposer();

  /* ---- actions ---- */
  $('#btnTrigger').onclick  = openTrigger;
  $('#btnHonesty').onclick  = () =>
    Modal.open(honestyHtml(App.model, App.scene.notes));
  $('#btnTimeline').onclick = () =>
    $('#timeline').classList.toggle('hidden');
  $('#btnMenu').onclick     = openMenu;

  // Closing the panel keeps the highlight. The usual reason to dismiss it is
  // to look at the region it was covering.
  $('#panelClose').onclick  = () => Panel.close();
  $('#logClose').onclick    = () => { endPreview(); LogDock.close(); };
  $('#btnChat').onclick     = () => App.chat?.toggle();
  $('#chatClose').onclick   = () => App.chat?.close();
  $('#modalClose').onclick  = Modal.close;
  $('#modalWrap').onclick   = (e) => {
    if (e.target.id === 'modalWrap') Modal.close();
  };

  /* ---- timeline ---- */
  App.timeline = new Timeline($('#tlCanvas'), scrubTo);
  $('#tlPlay').onclick  = togglePlay;
  $('#tlLive').onclick  = goLive;
  $('#tlDay1').onclick  = () => advance(1);
  $('#tlWeek').onclick  = () => advance(7);
  $('#tlMonth').onclick = () => advance(30);

  /* ---- compare two days ---- */
  $('#cmpGo').onclick  = runCompare;
  $('#cmpOff').onclick = endCompare;

  /* ---- keyboard ---- */
  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (Modal.isOpen()) Modal.close();
      else if (LogDock.isOpen()) { endPreview(); LogDock.close(); }
      else if (Panel.isOpen()) Panel.close();
      else if (V.selected && !visualiseLocked() && !currentLocked()) clearSelection();
    }
    // Anywhere the user can type. Guarding only INPUT meant the log dock's
    // <textarea> swallowed every space into the replay toggle, and typing a
    // digit switched mode out from under the sentence being written.
    const t = e.target;
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
        || t.tagName === 'SELECT' || t.isContentEditable) return;
    const k = e.key.toLowerCase();
    if (k === '1') setMode('anatomy');
    if (k === '2') setMode('circuits');
    if (k === '3') setMode('simulation');
    if (k === 'r') V.resetCamera();
    if (k === 'l') openTrigger();
    if (k === ' ') { e.preventDefault(); togglePlay(); }
  });
}

/* The default mode network has no single anatomical home - it is defined by
 * what co-activates at rest, spanning medial prefrontal, posterior cingulate,
 * precuneus, angular gyrus and lateral temporal cortex. So it is offered as an
 * overlay over the surface rather than as one more coloured lump. */
function buildNetworkList() {
  const nets = App.scene.networks || [];
  const group = $('#netGroup');
  const chk = $('#chkNetworks');
  if (!nets.length) {
    // Procedural fallback geometry carries no resting-state data.
    group.classList.add('hidden');
    return;
  }
  const host = $('#netList');
  host.innerHTML = nets.filter(n => n.id > 0).map(n => `
      <div class="structrow" data-net="${n.id}">
        <span class="sw" style="background:${n.color};color:${n.color}"></span>
        <span class="nm">${n.name}</span>
      </div>`).join('');

  host.querySelectorAll('.structrow').forEach(row => {
    row.onclick = () => {
      const id = Number(row.dataset.net);
      const same = App.viewer.activeNetwork === id;
      host.querySelectorAll('.structrow')
        .forEach(x => x.classList.remove('on'));
      if (same) { App.viewer.selectNetwork(null); Panel.close(); return; }
      row.classList.add('on');
      App.viewer.selectNetwork(id);
      showNetwork(id);
    };
  });

  chk.onchange = (e) => {
    const ok = App.viewer.setNetworkMode(e.target.checked);
    if (!ok) { e.target.checked = false; return; }
    host.classList.toggle('hidden', !e.target.checked);
    if (!e.target.checked) {
      host.querySelectorAll('.structrow')
        .forEach(x => x.classList.remove('on'));
      Panel.close();
    }
  };
}

function showNetwork(id) {
  const n = (App.scene.networks || []).find(x => x.id === id);
  if (!n) return;
  Panel.open(networkPanel(n, App.scene.network_caveat));
}

/* Appearance sliders. 100% is exactly the shipped look, so the defaults are
 * not a preference - they are the calibrated values. Persisted per browser. */
const LOOK_KEY = 'nf.look';
const LOOK_DEFAULT = { ghost: 100, hi: 100 };

function wireLookControls() {
  let saved = LOOK_DEFAULT;
  try {
    saved = { ...LOOK_DEFAULT, ...JSON.parse(localStorage.getItem(LOOK_KEY)) };
  } catch { /* corrupt or unavailable storage falls back to defaults */ }

  const ghost = $('#rngGhost'), hi = $('#rngHi');

  const apply = (persist) => {
    const g = Number(ghost.value), h = Number(hi.value);
    $('#ghostVal').textContent = g + '%';
    $('#hiVal').textContent = h + '%';
    App.viewer.setGhostAlpha(g / 100);
    App.viewer.setHighlight(h / 100);
    if (persist) {
      try { localStorage.setItem(LOOK_KEY, JSON.stringify({ ghost: g, hi: h })); }
      catch { /* private mode - the sliders still work for this session */ }
    }
  };

  ghost.value = saved.ghost;
  hi.value = saved.hi;
  apply(false);

  ghost.oninput = () => apply(true);
  hi.oninput = () => apply(true);
  $('#btnResetLook').onclick = () => {
    ghost.value = LOOK_DEFAULT.ghost;
    hi.value = LOOK_DEFAULT.hi;
    apply(true);
  };
}

/** Drop the highlight and un-light whichever dock row pointed at it. */
function clearSelection() {
  App.viewer.select(null);
  if (App.mode === 'anatomy') App.viewer.setPinFilter(null);
  markDockRow(null);
  $$('#netList .structrow').forEach(r => r.classList.remove('on'));
}

/** One selection across Systems, Cortex and Structures: lighting a row
 *  in one section un-lights the others. */
function markDockRow(row) {
  $$('#cortexList .structrow, #structList .structrow, #groupList .structrow')
    .forEach(r => r.classList.toggle('on', r === row));
}

/* Rebuild every dock list against the current focus. */
function buildDock() {
  buildGroupList();
  buildCortexList();
  buildStructureList();
  buildNetworkList();
  App.viewer.setPinScope(App.focus === 'all'
    ? null : [...FOCUS.cortex, ...FOCUS.structures]);

  const hidden = (App.scene.structures || []).length
    + (App.scene.cortical_regions || []).length
    - (FOCUS.structures.length + FOCUS.cortex.length);
  $('#focusNote').textContent = App.focus === 'all'
    ? 'Every region in the atlas.'
    : `${hidden} more on the brain, unlabelled — click them directly, `
      + 'or switch to Full.';
}

/* Composite anatomical terms. "Prefrontal cortex" and "dorsal striatum" are
 * collections of regions, not parcels, so they are offered as selections over
 * things that already exist rather than drawn as invented regions. */
function buildGroupList() {  const groups = (App.scene.groups || [])
    .filter(g => inFocus('groups', g.id));
  const host = $('#groupList');
  if (!groups.length) { host.classList.add('hidden'); return; }
  host.classList.remove('hidden');
  host.innerHTML = '<div class="docktitle">SYSTEMS</div>' +
    groups.map(g => `
      <div class="structrow" data-group="${g.id}">
        <span class="sw" style="background:${g.color};color:${g.color}"></span>
        <span class="nm">${g.short}</span>
      </div>`).join('');

  host.querySelectorAll('.structrow').forEach(row => {
    row.onclick = () => {
      const g = groups.find(x => x.id === row.dataset.group);
      const already = row.classList.contains('on');
      markDockRow(already ? null : row);
      if (already) {
        App.viewer.selectMany([], []);
        if (App.mode === 'anatomy') App.viewer.setPinFilter(null);
        Panel.close();
        return;
      }
      if (App.mode === 'anatomy') {
        App.viewer.setPinFilter([...g.regions, ...g.structures]);
      }
      showGroup(g);
    };
  });
}

function showGroup(g) {
  const idx = g.regions
    .map(id => App.cortexById[id])
    .filter(Boolean)
    .map(r => r.label_index);
  App.viewer.selectMany(idx, g.structures);
  const parts = [
    ...g.regions.map(id => App.cortexById[id]).filter(Boolean)
      .map(r => ({ name: r.name, color: r.color })),
    ...g.structures.map(id => App.structById[id]).filter(Boolean)
      .map(s => ({ name: s.name, color: s.color })),
  ];
  Panel.open(groupPanel(g, parts));
}

/* Surface parcels. These were always clickable on the brain, but several of
 * them - medial prefrontal, subgenual and posterior cingulate, anterior
 * insula - sit on the medial wall or inside a sulcus, so on an opaque brain
 * there was no way to reach them from the outside. This gives them a row.
 * Selecting a buried one makes the viewer switch to x-ray, as picking does. */
function buildCortexList() {
  const host = $('#cortexList');
  const order = FOCUS.cortex;
  const items = (App.scene.cortical_regions || [])
    .filter(r => inFocus('cortex', r.id))
    .slice()
    .sort((a, b) => {
      const ia = order.indexOf(a.id), ib = order.indexOf(b.id);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
          || a.label_index - b.label_index;
    });

  if (!items.length) { host.classList.add('hidden'); return; }
  host.classList.remove('hidden');
  host.innerHTML = '<div class="docktitle">CORTEX</div>' +
    items.map(r => `
      <div class="structrow" data-cortex="${r.label_index}">
        <span class="sw" style="background:${r.color};color:${r.color}"></span>
        <span class="nm">${r.label || r.short}</span>
        ${r.buried ? '<span class="tagmini" title="Folded inside a sulcus '
          + 'or on the medial wall - selecting it turns the surface to '
          + 'glass">buried</span>'
          : ''}
      </div>`).join('');

  host.querySelectorAll('.structrow').forEach(row => {
    row.onclick = () => {
      const li = Number(row.dataset.cortex);
      if (visualiseLocked() || currentLocked()) {
        const id = App.cortexByIndex[li]?.id;
        if (visualiseLocked() && App.visParts.includes(id)) {
          focusVisualisePart(id);
        }
        if (currentLocked() && engagedRows().some(r => r.part === id)) {
          focusCurrentPart(id);
        }
        showCortex(li, true);
        return;
      }
      const already = row.classList.contains('on');
      markDockRow(already ? null : row);
      if (already) {
        App.viewer.select(null);
        if (App.mode === 'anatomy') App.viewer.setPinFilter(null);
        Panel.close();
        return;
      }
      App.viewer.select({ kind: 'cortex', id: li });
      if (App.mode === 'anatomy') {
        const id = App.cortexByIndex[li]?.id;
        if (id) App.viewer.setPinFilter([id]);
      }
      showCortex(li);
    };
  });
}

function buildStructureList() {
  const host = $('#structList');
  const items = App.scene.structures
    .filter(s => inFocus('structures', s.id))
    .slice().sort((a, b) => a.order - b.order);
  const sys = [...new Set(items.map(s => s.system))];
  items.sort((a, b) => sys.indexOf(a.system) - sys.indexOf(b.system));
  host.innerHTML = '<div class="docktitle">STRUCTURES</div>' +
    items.map((s, i) => `
      ${s.system !== items[i - 1]?.system
        ? `<div class="docksub">${s.system}</div>` : ''}
      <div class="structrow" data-id="${s.id}">
        <span class="sw" style="background:${s.color};color:${s.color}"></span>
        <span class="nm">${s.label || s.short}</span>
        <span class="eye" data-eye="${s.id}">◉</span>
      </div>`).join('');

  host.querySelectorAll('.structrow').forEach(row => {
    const id = row.dataset.id;
    row.onclick = (e) => {
      if (e.target.dataset.eye) {
        const on = row.classList.toggle('off');
        App.viewer.toggleStructure(id, !on);
        return;
      }
      if (visualiseLocked() || currentLocked()) {
        if (visualiseLocked() && App.visParts.includes(id)) {
          focusVisualisePart(id);
        }
        if (currentLocked() && engagedRows().some(r => r.part === id)) {
          focusCurrentPart(id);
        }
        showStructure(id, true);
        return;
      }
      const already = row.classList.contains('on');
      markDockRow(already ? null : row);
      if (already) {
        App.viewer.select(null);
        if (App.mode === 'anatomy') App.viewer.setPinFilter(null);
        Panel.close();
        return;
      }
      App.viewer.select({ kind: 'structure', id });
      if (App.mode === 'anatomy') App.viewer.setPinFilter([id]);
      showStructure(id);
    };
  });
}

/* ======================================================================= */
/*  MODES                                                                   */
/* ======================================================================= */

function setMode(mode) {
  App.mode = mode;
  $$('#modebar button').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode));
  App.viewer.setMode(mode);

  const sim = mode === 'simulation';
  const circ = mode !== 'anatomy';

  endPaths();
  App.circuits.setVisible(circ);
  $('#statebar').classList.toggle('hidden', !sim);
  $('#btnTimeline').classList.toggle('hidden', !sim);
  // Every Simulation tab is about the brain; the timeline starts hidden and
  // the Timeline button brings it up.
  $('#timeline').classList.add('hidden');
  $('#metricWrap').style.opacity = sim ? '1' : '.32';

  // The side panel and the Compare split belong to one mode; they no longer
  // sit inside #statebar, so they do not get hidden with it.
  if (!sim) {
    $('#simPanel').classList.add('hidden');
    setSplit(false);
    App.viewer.setStructureCast(null);
    App.viewer.setGlassMode(false);
    App.viewer.setPinFilter(null);
    App.viewer.setPinBadges(null);
    App.viewer.setStrongGlow(false);
  }

  if (mode === 'anatomy') {
    Panel.close();
    clearSelection();
    App.circuits.clearPulses();
  }

  if (mode === 'circuits') {
    App.circuits.update(uniformWeights(0.55));
    App.circuits.setGhost(false);
    toast('Circuit view',
      'Curves show functional relationships, not white-matter tracts.');
  }
  if (sim) {
    applyState();
    setViewState(App.viewState);
  }
}

function uniformWeights(v) {
  const w = {};
  for (const e of App.model.edges) w[e.id] = v;
  return w;
}

function setViewState(s) {
  const prev = App.viewState;
  App.viewState = s;
  const hint = $('#stateHint');
  const vis = s === 'visualise';
  const reading = READING.has(s);

  $('#simPanel').classList.remove('hidden');
  // Compare has nothing obvious to add, so it has no box.
  $('#composer').classList.toggle('hidden', s === 'compare');
  $('#visBody').classList.toggle('hidden', !vis);
  $('#visReset').classList.toggle('hidden', !vis);
  App.viewer.setStrongGlow(vis || s === 'current');
  $('#curPanel').classList.toggle('hidden', vis);
  if (prev !== s && DEFAULT_INTENT[s]) setIntent(DEFAULT_INTENT[s]);
  // Tubes are the Circuits mode's subject. Here the subject is parts, and
  // twenty lit tubes through the middle of the brain drown them out.
  App.circuits.setVisible(false);
  if (prev !== s) endPaths();
  App.viewer.setStructureCast(vis ? VIS_CAST : null);
  // Until something is named, nothing is the subject: the structures should
  // be as see-through as the cortex, not floating solid in front of it.
  App.viewer.setGlassMode(true);
  App.viewer.setPinFilter(vis ? App.visParts : null);
  if (!reading) App.viewer.setPinBadges(null);
  setSplit(s === 'compare');
  // applyState() re-enters here; put the highlight back rather than lose it.
  if (vis && App.visParts.length) applyVisualise(App.visFocus);
  // The timeline is a wide bar across the lower half; it sits on top of the
  // brain and the panel. It can still be opened from the Timeline button.
  if (prev !== s) $('#timeline').classList.add('hidden');
  $('#btnTimeline').classList.toggle('hidden', vis);

  if (prev === 'visualise' && !vis) clearVisualise();
  if (READING.has(prev) && prev !== s) {
    App.curFocus = null;
    App.curNote = null;
    clearSelection();
    App.viewer.clearGlow();
  }

  hint.textContent = {
    visualise: 'what the rules associate with your words \u00b7 nothing is '
      + 'recorded',
    current: 'more / less engaged than a typical starting point \u00b7 a '
      + 'model, not a measurement',
    target: 'how the brain you are aiming for would be engaged \u00b7 a '
      + 'model, not a measurement',
    compare: 'where you are, beside where you are aiming \u00b7 turn either '
      + 'brain and both follow',
  }[s] || '';
  if (reading) renderReading();
}

/** Compare puts a second, camera-linked brain beside the first. */
function setSplit(on) {
  const was = document.body.classList.contains('split');
  if (on === was) return;
  document.body.classList.toggle('split', on);
  if (on && !App.viewer2) {
    const V2 = new Viewer($('#gl2'), { pinLayer: $('#pinLayer2') });
    V2.loadScene(App.scene);
    V2.setMode('simulation');
    V2.setGlassMode(true);
    V2.setTrueColor(true);
    // Either brain can be turned; the one last touched leads.
    App.viewer.controls.addEventListener('start', () => { App.camLead = 1; });
    V2.controls.addEventListener('start', () => { App.camLead = 2; });
    App.viewer2 = V2;
  }
  App.camLead = 1;
  App.viewer.resize();
  App.viewer2?.resize();
}

function syncCameras() {
  const [a, b] = App.camLead === 2
    ? [App.viewer2, App.viewer] : [App.viewer, App.viewer2];
  b.camera.position.copy(a.camera.position);
  b.controls.target.copy(a.controls.target);
}

/* ======================================================================= */
/*  STATE APPLICATION                                                       */
/* ======================================================================= */

/* The model resolves this to about a fifth of the range, not to a tenth of a
 * percent, so the headline says what it actually knows. */
function progressBand(pct) {
  if (pct < -2) return 'moving away';
  if (pct < 5) return 'just started';
  if (pct < 20) return 'early days';
  if (pct < 45) return 'some of the way';
  if (pct < 70) return 'over halfway';
  if (pct < 90) return 'most of the way';
  return 'at target';
}

function applyState(s = App.state) {
  App.state = s;
  $('#dayNum').textContent = App.replayDay ?? s.day;
  $('#practiceNum').textContent = s.total_practices;
  $('#streakNum').textContent = s.streak;
  // Fraction of the day-0 gap closed, not raw similarity to the target. The
  // old metric opened at 58.7% on a fresh simulation, because an ordinary
  // brain already partly resembles the target - which read as though the user
  // had started more than halfway done.
  const pct = (s.progress * 100);
  $('#alignBar').style.width = Math.max(0, Math.min(100, pct)) + '%';
  const band = progressBand(pct);
  $('#alignBand').textContent = band;
  $('#alignBand').classList.toggle('negative', pct < -0.05);
  // Rounded to 5: the model does not know this to a tenth of a percent, and
  // printing "69.6%" claims that it does. Small non-zero values get "<5%"
  // rather than "~0%", which would read as "nothing happened".
  $('#alignVal').textContent =
    Math.abs(pct) < 0.05 ? ''
    : Math.abs(pct) < 2.5 ? (pct > 0 ? '<5%' : '>-5%')
    : `~${Math.round(pct / 5) * 5}%`;
  $('#tlDay').textContent = `day ${App.replayDay ?? s.day}`;
  App.timeline.set(s, App.replayDay);
  refreshCompareOptions();
  if (App.mode === 'simulation') setViewState(App.viewState);
  if (Panel.isOpen() && App.viewer.selected) {
    const h = App.viewer.selected;
    if (h.kind === 'structure') showStructure(h.id, true);
    else showCortex(h.id, true);
  }
}

/* ======================================================================= */
/*  PANELS                                                                  */
/* ======================================================================= */

function connectionsFor(nodeIds) {
  const out = [];
  const pathColor = Object.fromEntries(
    App.model.pathways.map(p => [p.id, p.color]));
  for (const e of App.model.edges) {
    if (!nodeIds.includes(e.src) && !nodeIds.includes(e.dst)) continue;
    const other = nodeIds.includes(e.src) ? e.dst : e.src;
    const dir = nodeIds.includes(e.src) ? '→' : '←';
    out.push({
      label: `${dir} ${other} · ${e.label}`,
      w: App.state?.weights?.[e.id] ?? 0,
      color: pathColor[e.pathway],
    });
  }
  return out.sort((a, b) => b.w - a.w);
}

function showStructure(id, quiet = false) {
  const s = App.structById[id];
  if (!s) return;
  const nodes = App.model.nodes
    .filter(n => n.structure === id).map(n => n.id);
  Panel.open(regionPanel(s, 'structure', connectionsFor(nodes),
                         App.scene.notes.network));
  if (!quiet) highlightNodes(nodes);
}

/** Anatomical name of a Destrieux parcel id, or null. */
function fineName(id) {
  if (id === null || id === undefined) return null;
  const e = App.scene.fine_names?.[String(id)];
  if (!e || !e[0] || e[0] === 'Unassigned') return null;
  return e[0];
}

function showCortex(labelIndex, quiet = false, fine = null, network = null) {
  const r = App.cortexByIndex[labelIndex];
  if (!r) { Panel.close(); return; }
  const nodes = App.model.nodes
    .filter(n => n.cortex === r.id).map(n => n.id);
  const net = (network > 0)
    ? (App.scene.networks || []).find(n => n.id === network) : null;
  Panel.open(regionPanel(r, 'cortex', connectionsFor(nodes),
                         App.scene.notes.network, fineName(fine), net));
  if (!quiet) highlightNodes(nodes);
}

/**
 * Highlight regions the assistant referred to.
 *
 * Ids arrive from the model, so they are treated as untrusted: anything not
 * in the scene is ignored rather than thrown. The server already filters
 * these, but the renderer should not depend on that being true.
 *
 * Deliberately does not open the context panel. The chat panel is already
 * occupying that side of the screen, and swapping it out mid-sentence would
 * hide the answer the user is reading.
 */
function chatHighlight(ids) {
  const V = App.viewer;
  if (!V || !ids?.length) return;

  const labels = [];
  const structs = [];
  const named = [];
  for (const id of ids) {
    if (App.cortexById[id]) labels.push(App.cortexById[id].label_index);
    else if (App.structById[id]) structs.push(id);
    else continue;
    named.push(id);
  }
  if (!labels.length && !structs.length) return;

  if (!('chatPrevPins' in App)) {
    App.chatPrevPins = V._pinFilter ? [...V._pinFilter] : null;
  }
  V.setPinFilter(named);

  // selectMany() already handles buried cortex: it flips to x-ray when a
  // hidden region is selected and restores the previous mode afterwards.
  V.selectMany(labels, structs);
  
  V.clearGlow();
  for (const s of structs) V.setStructureGlow(s, 0.8);
  for (const l of labels) V.setRegionGlow(l, 0.8);
}

function chatUnlabel() {
  if (!('chatPrevPins' in App)) return;
  App.viewer?.setPinFilter(App.chatPrevPins);
  delete App.chatPrevPins;
}

/* ------------------------------------------------------------- compare */
/**
 * Populate the two day pickers from recorded history.
 *
 * Only days the engine actually snapshotted are offered. Letting the user
 * pick an arbitrary date and then silently showing the nearest one would
 * misattribute change to a day it did not happen on.
 */
function refreshCompareOptions() {
  const h = App.state?.history || [];
  const from = $('#cmpFrom'), to = $('#cmpTo');
  if (!from || !to) return;
  if (h.length < 2) {
    from.innerHTML = to.innerHTML = '<option>no history yet</option>';
    from.disabled = to.disabled = true;
    $('#cmpGo').disabled = true;
    return;
  }
  from.disabled = to.disabled = false;
  $('#cmpGo').disabled = false;
  const opts = h.map(x => `<option value="${x.day}">day ${x.day}</option>`)
    .join('');
  from.innerHTML = opts;
  to.innerHTML = opts;
  from.value = h[0].day;
  to.value = h[h.length - 1].day;
}

function runCompare() {
  const h = App.state?.history || [];
  // The pickers are rebuilt on every state change, but a state that arrives
  // by some other route would leave them stale - and a stale picker makes
  // this button silently do nothing, which is the worst failure mode.
  if ($('#cmpFrom').options.length !== h.length) refreshCompareOptions();
  const a = h.find(x => String(x.day) === $('#cmpFrom').value);
  const b = h.find(x => String(x.day) === $('#cmpTo').value);
  if (!a || !b || !a.weights || !b.weights) {
    toast('Cannot compare', 'Those days are not in the recorded history.');
    return;
  }

  if (App.mode !== 'simulation') setMode('simulation');
  App.compareOn = true;
  App.circuits.setVisible(true);
  App.circuits.focus(null);
  App.circuits.showDelta(a.weights, b.weights);
  $('#cmpOff').classList.remove('hidden');
  $('#cmpLegend').classList.remove('hidden');

  // Say plainly what moved, rather than leaving the user to read colours.
  let up = 0, down = 0, biggest = null;
  for (const e of App.model.edges) {
    const d = (b.weights[e.id] ?? 0) - (a.weights[e.id] ?? 0);
    if (d > 0.01) up++; else if (d < -0.01) down++;
    if (!biggest || Math.abs(d) > Math.abs(biggest.d)) biggest = { e, d };
  }
  const sign = biggest && biggest.d >= 0 ? '+' : '';
  toast(`Day ${a.day} \u2192 ${b.day}`,
        `${up} pathways stronger, ${down} weaker. Largest change: ` +
        `${biggest.e.id} ${sign}${(biggest.d * 100).toFixed(0)}%. ` +
        `Simulated values.`, 5200);
}

function endCompare() {
  App.compareOn = false;
  App.circuits.showDelta(null, null);
  App.circuits.update(App.state.weights);
  $('#cmpOff').classList.add('hidden');
  $('#cmpLegend').classList.add('hidden');
}

/**
 * Open the panel for one simulated connection, with its own history.
 *
 * The series comes from the engine's per-day snapshots, so this is the real
 * recorded trajectory of that edge rather than a redrawing of the current
 * value.
 */
function showEdge(edgeId) {
  const e = App.model.edges.find(x => x.id === edgeId);
  if (!e) return;
  const p = App.model.pathways.find(x => x.id === e.pathway);
  const anchor = (id) => {
    const n = App.model.nodes.find(x => x.id === id);
    return n?.label || id;
  };
  const hist = App.state?.history || [];
  const series = hist
    .map(h => h.weights?.[edgeId])
    .filter(v => typeof v === 'number');
  const now = App.state?.weights?.[edgeId] ?? 0;
  if (series.length) series.push(now);

  App.viewer.select(null);
  App.circuits.focus([e.pathway]);
  Panel.open(edgePanel(e, p, anchor(e.src), anchor(e.dst), series, now));
}

function highlightNodes(nodeIds) {  if (!nodeIds.length || App.mode === 'anatomy') return;
  const paths = new Set();
  for (const e of App.model.edges)
    if (nodeIds.includes(e.src) || nodeIds.includes(e.dst))
      paths.add(e.pathway);
  App.circuits.focus([...paths]);
}

/* ======================================================================= */
/*  THE LOGGING FLOW                                                        */
/* ======================================================================= */

function openTrigger() {
  if (App.mode !== 'simulation') setMode('simulation');
  goLive();
  App.chat?.close();
  App.lastLit = null;
  App.lastAnalysis = null;
  LogDock.open(triggerModalHtml());
  $('#choiceHost').innerHTML =
    choicesHtml(App.model.events, App.model.categories);

  // Hovering a response shows what it engages, before committing to it. The
  // dock leaves the brain visible, which is the whole reason this stopped
  // being a modal.
  //
  // Previews stay disarmed until the pointer actually moves: the dock slides
  // in under a stationary cursor, and the button that lands beneath it fires
  // mouseenter on its own. That dimmed every other pathway to 0.16 the
  // instant the panel opened, which read as a flicker.
  let armed = false;
  $('#logdock').addEventListener('pointermove', () => { armed = true; },
                                 { once: true });

  $$('#choiceHost .choice').forEach(b => {
    const ev = App.eventById[b.dataset.event];
    b.onclick = () => doLog(b.dataset.event);
    b.onmouseenter = b.onfocus = () => {
      if (!armed) return;
      b.classList.add('previewing');
      previewEvent(ev);
    };
    b.onmouseleave = b.onblur = () => {
      if (!armed) return;
      b.classList.remove('previewing');
      endPreview();
    };
  });

  wireInterpreter();
  showInterpreterHelp();

  // the trigger itself: light salience, then fire the cascade
  App.viewer.setStructureGlow('amygdala', 1.0);
  App.circuits.cascade(
    ['BLA->CeA', 'BLA->dACC'], { stagger: 0.18, speed: 1.8 });
  setTimeout(() => App.viewer.setStructureGlow('amygdala', 0.35), 1600);
}

/** Which structures, regions and pathways an event's rule touches. */
function eventTargets(ev) {
  const acts = ev?.activations || {};
  const edges = [], pathways = [], lit = [];
  for (const e of App.model.edges) {
    const a = acts[e.pathway];
    if (!a) continue;
    if (!pathways.includes(e.pathway)) pathways.push(e.pathway);
    if (a > 0) edges.push(e.id);
    for (const nid of [e.src, e.dst]) {
      const n = App.nodeById[nid];
      if (n) lit.push([n, Math.abs(a)]);
    }
  }
  return { edges, pathways, lit };
}

/** Show an event's reach without applying it. */
function previewEvent(ev) {
  if (!ev) return;
  const { pathways, lit } = eventTargets(ev);
  App.viewer.clearGlow();
  for (const [n, a] of lit) {
    const g = Math.min(1, a) * 0.7;
    if (n.structure) App.viewer.setStructureGlow(n.structure, g);
    if (n.cortex) {
      const r = App.cortexById[n.cortex];
      if (r) App.viewer.setRegionGlow(r.label_index, g);
    }
  }
  App.circuits.focus(pathways.length ? pathways : null);
}

function endPreview() {
  App.viewer.clearGlow();
  // Fall back to the interpreted entry's lighting rather than to nothing,
  // or hovering a choice once would wipe the result the user is reading.
  if (App.lastAnalysis) { highlightAnalysis(App.lastAnalysis); return; }
  if (App.lastLit?.length) { lightFor(App.lastLit); return; }
  App.circuits.focus(null);
}

/**
 * Wire the free-text box inside the log modal.
 *
 * The interpreter only ever fills in a suggestion. Applying it goes through
 * the same doLog() path as pressing a button, so there is exactly one route
 * by which anything in this app changes.
 */
function wireInterpreter() {
  const go = $('#nlGo'), out = $('#nlOut'), src = $('#nlSource');
  if (!go) return;

  go.onclick = async () => {
    const text = $('#nlText').value.trim();
    if (!text) { $('#nlText').focus(); return; }
    go.disabled = true;
    go.classList.add('is-busy');
    go.textContent = 'Reading';
    App.lastAnalysis = null;        // stale result must not outlive its entry
    App.lastLit = null;
    out.classList.remove('hidden');
    out.innerHTML = interpretingHtml(
      App.llm?.enabled ? 'Reading your entry'
                       : 'Matching phrases against the 16 known events');
    try {
      const r = await API.interpret(text);
      const interp = r.interpretation;
      out.innerHTML = proposalsHtml(interp);
      src.textContent = interp.source.startsWith('llm:')
        ? interp.source.replace('llm:', 'model: ')
        : 'offline keyword matching';
      // The user's own words are kept as the note, so the record is theirs
      // and not the classifier's paraphrase of it.
      $$('#nlOut .nl-prop').forEach(b => b.onclick = () => {
        doLog(b.dataset.event, parseFloat(b.dataset.intensity) || 1.0, text);
      });
      const ids = interp.proposals.map(p => p.event).slice(0, 2);
      // Light the brain from the proposals immediately: classification takes
      // well under a second, the prose does not, and waiting for the prose
      // would leave the brain inert while the user is reading the result.
      lightFor(ids);
      if (ids.length) $('#pickWrap')?.removeAttribute('open');
      showAnalysis(text, ids);
    } catch (e) {
      out.innerHTML =
        `<div class="nl-empty">Could not interpret that (${e.message}).
         Pick an event below instead.</div>`;
    } finally {
      go.disabled = false;
      go.classList.remove('is-busy');
      go.textContent = 'Interpret';
    }
  };

  $('#nlText').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) go.click();
  });
}

/**
 * Say which of the two interpreters is actually going to run, and how to
 * change that.
 *
 * Asked for on every open rather than cached in the page: the user may have
 * started LM Studio since last time, and a panel that still claimed "no
 * model" would be wrong in the one direction that matters. The server side
 * caches the port probe, so repeat opens are free.
 *
 * It also warms that cache, which is why this is called on open and not
 * lazily when the disclosure is expanded - the first Interpret then does not
 * pay for the probe.
 */
async function showInterpreterHelp() {
  const host = $('#nlHow');
  if (!host) return;
  host.innerHTML = interpreterHelpHtml(App.llm || null);
  let cfg = null;
  try {
    cfg = (await API.interpretConfig()).llm;
  } catch {
    cfg = { enabled: false };     // unreachable server: the offline path is
  }                               // what they will get, so say so
  App.llm = cfg;
  // The dock may have been closed, or reopened, while that was in flight.
  const now = $('#nlHow');
  if (now) now.innerHTML = interpreterHelpHtml(cfg);
}

/**
 * Fetch the reading and light up what it names.
 *
 * Runs after the proposals are already on screen: it is the slow call, and
 * making the user wait for prose before seeing the classification would
 * make the whole box feel broken on a cold model.
 */
async function showAnalysis(text, eventIds) {
  const out = $('#nlOut');
  if (!out || !eventIds.length) return;
  const slot = document.createElement('div');
  slot.className = 'an-slot';
  slot.innerHTML = interpretingHtml('Working out what it engages');
  out.appendChild(slot);
  try {
    const r = await API.analyse(text, eventIds);
    const a = r.analysis;
    slot.innerHTML = analysisHtml(a);
    App.lastAnalysis = a;
    highlightAnalysis(a);
  } catch (e) {
    slot.remove();
  }
}

/**
 * Brighten what an entry engages, straight from the event rules.
 *
 * Same treatment as hovering a response, but driven by the classifier, so
 * the brain reacts the moment the entry is understood.
 */
function lightFor(eventIds) {
  const evs = eventIds.map(id => App.eventById[id]).filter(Boolean);
  if (!evs.length) return;
  App.lastLit = eventIds;
  App.viewer.clearGlow();
  const pathways = new Set();
  const best = new Map();
  for (const ev of evs) {
    const { pathways: ps, lit } = eventTargets(ev);
    ps.forEach(p => pathways.add(p));
    for (const [n, a] of lit) {
      if (!best.has(n) || best.get(n) < a) best.set(n, a);
    }
  }
  for (const [n, a] of best) {
    const g = Math.min(1, a) * 0.85;
    if (n.structure) App.viewer.setStructureGlow(n.structure, g);
    if (n.cortex) {
      const r = App.cortexById[n.cortex];
      if (r) App.viewer.setRegionGlow(r.label_index, g);
    }
  }
  App.circuits.focus(pathways.size ? [...pathways] : null);
}

/* ======================================================================= */
/*  COMPOSER                                                                */
/* ======================================================================= */

/* One box for everything typed in Simulation. It used to be three - the
 * Visualise box, Current's patterns box and the log dock's free text - all
 * running the same classifier with different consequences. What it does now
 * follows the tab, and the label above it says so. */
const INTENTS = {
  usual: { label: 'HOW YOU USUALLY REACT', go: 'Add',
           ph: 'e.g. I bottle things up at work and snap at home' },
  goal:  { label: 'WHO YOU WANT TO BE', go: 'Add',
           ph: 'e.g. I want to pause before I answer when I am criticised' },
  show:  { label: 'WHAT IS HAPPENING RIGHT NOW', go: 'Explain',
           ph: 'e.g. my chest is tight and I keep replaying the argument' },
};
const DEFAULT_INTENT = { current: 'usual', target: 'goal', visualise: 'show' };

function setIntent(k) {
  App.intent = k;
  $('#composerLabel').textContent = INTENTS[k].label;
  $('#composerText').placeholder = INTENTS[k].ph;
  $('#composerGo').textContent = INTENTS[k].go;
  $('#composerMsg').textContent = '';
}

/** Fit a textarea to its text, up to a cap; past that it scrolls. */
function growBox(el) {
  el.style.height = 'auto';
  const cap = Math.round(window.innerHeight * 0.4);
  el.style.height = Math.min(el.scrollHeight + 2, cap) + 'px';
  el.style.overflowY = el.scrollHeight + 2 > cap ? 'auto' : 'hidden';
  const n = el.value.length, max = Number(el.maxLength);
  const count = el.parentElement?.querySelector('.box-count');
  if (count) count.textContent = max > 0 && n > max * 0.8 ? `${n} / ${max}` : '';
}

function wireComposer() {
  const box = $('#composerText'), go = $('#composerGo'), msg = $('#composerMsg');
  setIntent('usual');
  // Any textarea marked data-grow, including ones rendered later.
  document.addEventListener('input', (e) => {
    if (e.target.matches?.('textarea[data-grow]')) growBox(e.target);
  });

  go.onclick = async () => {
    const text = box.value.trim();
    if (text.length < 4) { box.focus(); return; }
    const k = App.intent;
    msg.textContent = '';

    if (k === 'show') { runVisualise(text); return; }

    go.disabled = box.disabled = true;
    go.textContent = 'Reading\u2026';
    let r;
    try {
      r = await (k === 'goal' ? API.addGoal(text) : API.addNote(text));
    } catch (err) {
      r = { added: false, reason: err.message };
    }
    go.disabled = box.disabled = false;
    go.textContent = INTENTS[k].go;
    if (!r.added) { msg.textContent = r.reason || 'Could not add that.'; return; }
    box.value = '';
    growBox(box);
    App.curFocus = null;
    // Bring forward what the new note touches, so cause and effect meet.
    App.curNote = (k === 'goal' ? r.state.goal_effects
                                : r.state.note_effects)?.length - 1;
    if (!(App.curNote >= 0)) App.curNote = null;
    const before = App.state?.engagement;
    applyState(r.state);
    if (App.curNote != null) focusNote(App.curNote);
    if (k === 'usual') {
      playEngagementChange(before, Object.keys(
        r.state.profile?.notes?.at(-1)?.events || {}));
    }
    const read = (r.read_as || []).map(p => p.label.toLowerCase()).join(', ');
    const by = (r.source || '').startsWith('llm:')
      ? `read by ${r.source.slice(4)}` : 'matched by keyword';
    $('#composerMsg').textContent = `Added - ${by} as: ${read}.`;
  };

  box.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go.click(); }
  });
  $('#visReset').onclick = () => { clearVisualise(); box.focus(); };
}

/* ======================================================================= */
/*  VISUALISE                                                               */
/* ======================================================================= */

/** Move the segmented control and the view together. */
function selectState(s) {
  $$('#stateSeg button').forEach(b =>
    b.classList.toggle('active', b.dataset.state === s));
  setViewState(s);
}

function clearVisualise() {
  App.visSeq++;                          // strand any reply still in flight
  stopMoment();
  endPaths();
  App.visParts = [];
  App.visFocus = null;
  App.visMoves = [];
  App.visAlts = [];
  App.visEvents = [];
  App.visPreview = null;
  $('#visOut').innerHTML = '';
  clearSelection();
  App.viewer.clearGlow();
  App.viewer.setPinBadges(null);
  const vis = App.viewState === 'visualise';
  App.viewer.setPinFilter(vis ? [] : null);
  if (vis) App.viewer.setStructureCast(VIS_CAST);
}

function visualiseLocked() {
  return App.mode === 'simulation' && App.viewState === 'visualise'
      && App.visParts.length > 0;
}

/* ======================================================================= */
/*  CURRENT / TARGET / COMPARE                                              */
/* ======================================================================= */

// Tabs that read out engagement rather than drawing connections.
const READING = new Set(['current', 'target', 'compare']);

const onBrain = (rows) => (rows || [])
  .filter(r => App.structById[r.part] || App.cortexById[r.part]);

function readingRows() {
  return onBrain(App.viewState === 'target'
    ? App.state?.target_engagement : App.state?.engagement);
}

const engagedRows = () => readingRows().filter(r => r.level !== 'typical');

function currentLocked() {
  return App.mode === 'simulation' && READING.has(App.viewState)
      && (engagedRows().length > 0
          || onBrain(App.state?.target_engagement).some(r => r.level !== 'typical'));
}

function renderReading() {
  stopEngAnim();
  const s = App.viewState;
  if (s === 'compare') { renderCompare(); return; }
  const target = s === 'target';
  const rows = readingRows();
  paintEngagement(App.viewer, rows, focusIds(), !target);
  $('#curPanel').innerHTML = target
    ? targetPanelHtml(rows) : currentPanelHtml(rows, !!App.state?.profile);
  wireReadingPanel();
  showPartPaths();
}

/** A focused part, or every part the focused note pushes. */
function focusIds() {
  if (App.curFocus) return [App.curFocus];
  return App.curNote == null ? [] : noteTags(App.curNote).map(e => e.part);
}

/** The rows that name this note as a reason, so the note and the rows say
 *  the same thing; failing that, the parts its events push hardest. */
function noteTags(i) {
  const cited = readingRows().filter(r => r.level !== 'typical'
    && (r.why?.refs || []).includes(`note:${i}`));
  if (cited.length) return cited.map(r => ({ part: r.part, dir: r.level }));
  const fx = (App.viewState === 'target' ? App.state?.goal_effects
                                         : App.state?.note_effects) || [];
  return (fx[i] || []).filter(e => partOf(e.part));
}

function renderCompare() {
  const cur = onBrain(App.state?.engagement);
  const tgt = onBrain(App.state?.target_engagement);
  const f = App.curFocus ? [App.curFocus] : [];
  paintEngagement(App.viewer, cur, f);
  if (App.viewer2) paintEngagement(App.viewer2, tgt, f);
  $('#curPanel').innerHTML = comparePanelHtml(cur, tgt);
  wireReadingPanel();
  showPartPaths();
}

/** Click and Enter/Space for a non-button that holds buttons of its own. */
function pressable(el, fn, skip = 'button') {
  el.onclick = (e) => { if (!e.target.closest(skip)) fn(); };
  el.onkeydown = (e) => {
    if (e.target === el && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      fn();
    }
  };
}

function focusNote(i) {
  App.curFocus = null;
  App.curNote = i;
  renderReading();
  const el = $('#curPanel .cur-note.on');
  if (!el) return;
  el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

function wireReadingPanel() {
  $$('#curPanel .eng-row, #curPanel .cmp-row[data-part]').forEach(b => {
    b.classList.toggle('on', b.dataset.part === App.curFocus);
    pressable(b, () => focusCurrentPart(
      App.curFocus === b.dataset.part ? null : b.dataset.part),
      'button, a, textarea, .eng-levers, .eng-dispute');
  });
  $$('#curPanel [data-dispute]').forEach(a => a.onclick = (e) => {
    e.preventDefault();
    App.disputePart = App.curFocus;
    renderReading();
    $('#disputeText')?.focus();
  });
  const dClose = $('#curPanel [data-dispute-close]');
  if (dClose) dClose.onclick = () => { App.disputePart = null; renderReading(); };
  const dAdd = $('#curPanel [data-dispute-add]');
  if (dAdd) dAdd.onclick = async () => {
    const text = $('#disputeText').value.trim();
    if (text.length < 4) { $('#disputeText').focus(); return; }
    dAdd.disabled = true;
    dAdd.textContent = 'Reading\u2026';
    let r;
    try { r = await API.addNote(text); }
    catch (err) { r = { added: false, reason: err.message }; }
    if (!r.added) {
      dAdd.disabled = false;
      dAdd.textContent = 'Add';
      $('#disputeMsg').textContent = r.reason || 'Could not add that.';
      return;
    }
    App.disputePart = null;
    const before = App.state?.engagement;
    applyState(r.state);
    focusNote(r.state.profile.notes.length - 1);
    playEngagementChange(before, Object.keys(
      r.state.profile.notes.at(-1)?.events || {}));
  };
  $$('#curPanel [data-edit]').forEach(b => b.onclick = (e) => {
    e.preventDefault();
    openReread('current', Number(b.dataset.edit));
  });
  $$('#curPanel [data-edit-goal]').forEach(b => b.onclick = (e) => {
    e.preventDefault();
    openReread('target', Number(b.dataset.editGoal));
  });
  $$('#curPanel .cur-note[data-note]').forEach(n => {
    const i = Number(n.dataset.note);
    pressable(n, () => focusNote(App.curNote === i ? null : i));
  });
  $$('#curPanel [data-ref]').forEach(b => b.onclick = (e) => {
    e.preventDefault();
    const [kind, val] = b.dataset.ref.split(':');
    if (kind === 'note') focusNote(Number(val));
    else if (kind === 'answer') {
      openQuiz(App.viewState === 'target' ? 'target' : 'current',
               val.split('/')[0]);
    }
  });
  $$('#curPanel [data-log]').forEach(b =>
    b.onclick = () => doLog(b.dataset.log));
  $$('#curPanel [data-quiz]').forEach(b =>
    b.onclick = () => openQuiz(b.dataset.quiz || 'current'));
  const night = $('#curPanel [data-night-close]');
  $$('#curPanel [data-pview]').forEach(b =>
    b.onclick = () => setPathView(b.dataset.pview));
  $$('#curPanel [data-scen]').forEach(b => b.onclick = () => {
    App.projScenario = b.dataset.scen;
    renderReading();
  });
  $$('#curPanel .pr-row').forEach(r => pressable(r, () => {
    const id = r.dataset.pathway;
    App.pathFocus = App.pathFocus === id ? null : id;
    if (App.pathView == null) App.pathView = 'since';
    App.pathDrawn = null;
    renderReading();
  }));
  if (night) night.onclick = () => {
    App.overnightSeen = App.state.day;
    renderReading();
  };
  const clear = $('#curPanel [data-clear]');
  if (clear) clear.onclick = async () => {
    if (!confirm('Clear your description and everything logged since?')) return;
    App.curFocus = null;
    App.curNote = null;
    applyState(await API.reset(true));
  };
  const clearT = $('#curPanel [data-clear-target]');
  if (clearT) clearT.onclick = async () => {
    if (!confirm('Clear your target? Progress goes back to the app\'s '
                 + 'default aims.')) return;
    App.curFocus = null;
    App.curNote = null;
    applyState((await API.clearTarget()).state);
  };
  $$('#curPanel [data-del]').forEach(b => b.onclick = async (e) => {
    e.preventDefault();
    App.curNote = null;
    App.disputePart = null;
    applyState((await API.deleteNote(Number(b.dataset.del))).state);
  });
  $$('#curPanel [data-del-goal]').forEach(b => b.onclick = async () => {
    App.curNote = null;
    applyState((await API.deleteGoal(Number(b.dataset.delGoal))).state);
  });
}

/**
 * Show which parts a brain has more or less engaged than typical.
 *
 * Part colours stay the part's own, so the brain reads the same as in every
 * other view. More engaged parts come forward, less engaged ones stay
 * visible but faint, typical ones drop to glass, and the arrows on the
 * labels say which is which. With `glow` (the Current tab) each more
 * engaged part also glows in its own colour, graded by its score with the
 * Visualise formula; less engaged parts get a faint, pale glow. Clicking
 * a part (or a note) brings it forward without moving the camera.
 */
function paintEngagement(V, rows, focusIds = [], glow = false) {
  const lit = rows.filter(r => r.level !== 'typical');
  const focus = new Set(focusIds.filter(id => lit.some(r => r.part === id)));
  const isForward = (r) => focus.size ? focus.has(r.part) : r.level === 'more';

  const labels = [], structs = [], badges = {};
  const up = { labels: [], structs: [] };
  for (const r of lit) {
    badges[r.part] = r.level === 'more' ? '\u25b2' : '\u25bc';
    const forward = isForward(r);
    if (App.structById[r.part]) {
      structs.push(r.part);
      if (forward) up.structs.push(r.part);
    } else {
      const li = App.cortexById[r.part].label_index;
      labels.push(li);
      if (forward) up.labels.push(li);
    }
  }

  V.clearGlow();
  App.engGlow = glow ? engGlowMap(lit) : {};
  if (lit.length) {
    V.selectMany(labels, structs, null);
    V.setEmphasis(up.labels, up.structs, true);
  } else {
    V.selectMany([], []);
  }
  for (const [part, g] of Object.entries(App.engGlow)) setPartGlow(V, part, g);
  const less = glow ? lit.filter(r => r.level === 'less') : [];
  V.setPale(
    less.filter(r => App.cortexById[r.part])
      .map(r => App.cortexById[r.part].label_index),
    less.filter(r => App.structById[r.part]).map(r => r.part));
  V.setPinFilter(lit.map(r => r.part));
  V.setPinBadges(badges);
}

/** More engaged parts by score, scaled so the most engaged gets full
 *  Visualise glow; the square root keeps mild ones visible. Less engaged
 *  parts get a pale glow that fades the further below typical they are. */
const PALE_GLOW_MAX = 0.25;
function engGlowMap(lit) {
  const top = (lvl) => Math.max(1e-6, ...lit
    .filter(r => r.level === lvl).map(r => Math.abs(r.score)));
  const more = top('more'), less = top('less');
  return Object.fromEntries(lit.map(r => [r.part, r.level === 'more'
    ? VIS_GLOW_MIN + VIS_GLOW_SPAN * Math.sqrt(Math.max(0, r.score) / more)
    : PALE_GLOW_MAX * (1 - 0.6 * Math.sqrt(Math.abs(r.score) / less))]));
}

function setPartGlow(V, part, g) {
  if (App.structById[part]) V.setStructureGlow(part, g);
  else if (App.cortexById[part]) V.setRegionGlow(App.cortexById[part].label_index, g);
}

// Glow by order of change, not size: scores move ~0.05, which reads as
// the same brightness everywhere if mapped directly.
const ENG_PEAK = [0.55, 0.45, 0.38, 0.32];
const ENG_STEP_MS = 150;

function stopEngAnim() {
  const a = App.engAnim;
  if (!a) return;
  a.timers.forEach(clearTimeout);
  cancelAnimationFrame(a.raf);
  App.engAnim = null;
  App.viewer.clearGlow();
}

/**
 * What a new reaction changed, played once: the route its events take,
 * then each moved part switching from its old reading to its new one,
 * biggest change first, with a brief glow on parts that went up. Settles
 * on the ordinary reading; nothing keeps pulsing.
 */
function playEngagementChange(beforeRows, eventIds) {
  stopEngAnim();
  if (App.mode !== 'simulation' || App.viewState !== 'current') return;
  const V = App.viewer;
  const after = readingRows();
  const old = Object.fromEntries(onBrain(beforeRows).map(r => [r.part, r]));
  const moved = after
    .map(r => ({ r, d: r.score - (old[r.part]?.score || 0),
                 was: old[r.part]?.level || 'typical' }))
    .filter(x => x.r.level !== x.was || Math.abs(x.d) >= TREND_STEADY)
    .sort((a, b) => Math.abs(b.d) - Math.abs(a.d))
    .slice(0, ENG_PEAK.length);
  if (!moved.length) return;

  const anim = App.engAnim = { timers: [], raf: 0, glow: {} };
  const at = (ms, fn) => anim.timers.push(setTimeout(fn, ms));
  const mix = Object.fromEntries(after.map(r =>
    [r.part, old[r.part] || { ...r, level: 'typical', score: 0 }]));
  const paintMix = () => paintEngagement(V, Object.values(mix), [], true);
  const tag = (part, on) => $$(`#curPanel .cur-fx em[data-part="${part}"]`)
    .forEach(e => e.classList.toggle('lit', on));

  const edges = [...new Set(eventIds.flatMap(
    e => eventTargets(App.eventById[e]).edges))];
  let t0 = 0;
  if (edges.length) {
    clearTimeout(App.pathTimer);
    App.circuits.update(App.state.weights);
    App.circuits.spotlight(edges);
    App.circuits.cascade(edges, { stagger: 0.13, speed: 1.35 });
    App.pathTimer = setTimeout(pathsDone, 2600);
    t0 = 900;
  }
  paintMix();

  moved.forEach((x, i) => at(t0 + i * ENG_STEP_MS, () => {
    mix[x.r.part] = x.r;
    paintMix();
    tag(x.r.part, true);
    if (x.d > 0) anim.glow[x.r.part] = { peak: ENG_PEAK[i],
                                         start: performance.now() };
  }));
  const end = t0 + moved.length * ENG_STEP_MS + 900;
  at(end, () => {
    paintEngagement(V, after, focusIds(), true);
    for (const g of Object.values(anim.glow)) g.fade = performance.now();
    moved.forEach(x => tag(x.r.part, false));
  });
  at(end + 1000, () => {
    stopEngAnim();
    paintEngagement(V, after, focusIds(), true);
  });

  const frame = () => {
    const now = performance.now();
    for (const [part, g] of Object.entries(anim.glow)) {
      const k = Math.min(1, (now - g.start) / 350)
        * (g.fade ? Math.max(0, 1 - (now - g.fade) / 900) : 1);
      const peak = App.structById[part] ? g.peak : g.peak + 0.2;
      // The flash rides on the steady glow, never below it.
      setPartGlow(V, part, Math.max(App.engGlow?.[part] ?? 0, peak * k));
    }
    anim.raf = requestAnimationFrame(frame);
  };
  anim.raf = requestAnimationFrame(frame);
}

function focusCurrentPart(id) {
  if (id !== App.curFocus) App.disputePart = null;
  App.curFocus = id;
  App.curNote = null;
  renderReading();
}

/** Let the user say which events a note describes. Only the choice of
 *  events is theirs; what those events do stays with the rules. */
function openReread(kind, i) {
  const target = kind === 'target';
  const note = (target ? App.state?.target_profile : App.state?.profile)
    ?.notes?.[i];
  if (!note) return;
  const had = new Set(Object.keys(note.events || {}));
  const cats = {};
  for (const e of App.model.events) (cats[e.category] ||= []).push(e);
  Modal.open(`
    <div class="m-title">How should this be read?</div>
    <div class="m-sub">\u201c${escHtml(note.text)}\u201d<br>Tick up to three
      things it describes. The app's rules decide what each one does.</div>
    <form id="reread">
      ${Object.entries(cats).map(([c, evs]) => `
        <fieldset class="qz-q">
          <legend>${escHtml(c)}</legend>
          <div class="qz-opts">${evs.map(e => `
            <label class="qz-opt">
              <input type="checkbox" value="${escHtml(e.id)}"
                ${had.has(e.id) ? 'checked' : ''}>
              <span>${escHtml(e.label)}</span>
            </label>`).join('')}</div>
        </fieldset>`).join('')}
      <div class="qz-foot">
        <span id="rereadMsg"></span>
        <button type="submit" class="primary">Save</button>
      </div>
    </form>`);
  $('#reread').onsubmit = async (e) => {
    e.preventDefault();
    const ids = $$('#reread input:checked').map(x => x.value);
    const msg = $('#rereadMsg');
    if (!ids.length) { msg.textContent = 'Pick at least one, or remove it.'; return; }
    if (ids.length > 3) { msg.textContent = 'Up to three.'; return; }
    const r = await (target ? API.editGoal : API.editNote)(i, ids);
    if (!r.ok) { msg.textContent = r.reason; return; }
    Modal.close();
    App.disputePart = null;
    const before = App.state?.engagement;
    applyState(r.state);
    focusNote(i);
    if (!target) playEngagementChange(before, ids);
    toast('Reading corrected', 'Rebuilt from the events you picked.');
  };
}

/** "Doesn't feel right": what this row rests on, each with its fix, and a
 *  box to describe what is missing. Never sets a part directly. */
function disputeHtml(r) {
  if (App.disputePart !== r.part) {
    return `<a href="#" class="eng-dispute-open" data-dispute>Doesn't feel
      right?</a>`;
  }
  const reasons = r.why?.reasons || [];
  const items = (r.why?.refs || []).map((ref, i) => {
    const [kind, val] = ref.split(':');
    const fix = kind === 'note'
      ? `<a href="#" data-edit="${escHtml(val)}">fix how it was read</a>
         \u00b7 <a href="#" data-del="${escHtml(val)}">remove</a>`
      : kind === 'answer'
        ? `<a href="#" data-ref="${escHtml(ref)}">change answer</a>`
      : kind === 'rarely' ? 'if you do, describe it below'
      : kind === 'log' ? 'logged by you' : '';
    return `<li>${escHtml(reasons[i] || '')}<span>${fix}</span></li>`;
  });
  return `
    <div class="eng-dispute">
      <div class="cur-group">DOESN'T FEEL RIGHT?</div>
      ${items.length ? `<p>This reading rests on:</p><ul>${items.join('')}</ul>`
        : ''}
      <p>Correct any of these, or describe what is missing - what you
        actually do, in your own words.</p>
      <textarea id="disputeText" rows="2" maxlength="1200" data-grow
        placeholder="e.g. I usually stay calm when I am criticised"></textarea>
      <div class="vis-row">
        <button class="cur-go" data-dispute-add>Add</button>
        <button class="cur-ghost" data-dispute-close>Cancel</button>
        <span class="cur-msg" id="disputeMsg"></span>
      </div>
    </div>`;
}

// Below this a part reads as steady; logging moves scores about 0.01/week.
const TREND_STEADY = 0.005;

/** Sparkline over recorded days, and which way the last week went. */
function trendOf(part) {
  const t = App.state?.engagement_trend;
  const ys = t?.parts?.[part];
  if (!ys || ys.length < 2) return null;
  const days = t.days, last = days[days.length - 1];
  const j = days.findIndex(d => d >= last - 7);
  if (days[j] === last) return null;
  const d = ys[ys.length - 1] - ys[j];
  const word = Math.abs(d) < TREND_STEADY ? 'steady'
    : d > 0 ? 'rising' : 'falling';
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 0.02) { const m = (hi + lo) / 2; lo = m - 0.01; hi = m + 0.01; }
  const W = 40, H = 12;
  const pts = ys.map((y, k) => `${(k / (ys.length - 1) * W).toFixed(1)},${
    (H - 1 - (y - lo) / (hi - lo) * (H - 2)).toFixed(1)}`).join(' ');
  return {
    svg: `<svg class="eng-spark ${word}" width="${W}" height="${H}"
      viewBox="0 0 ${W} ${H}" aria-hidden="true"><polyline points="${pts}"/></svg>`,
    text: `${word} since day ${days[j]}`,
  };
}

// Scores in this model run to about +-0.5; the bar is full at that.
const ENG_FULL = 0.5;

const partOf = (id) => App.structById[id] || App.cortexById[id];

/** Plain name plus the anatomical one, e.g. "Value (vmPFC)". The sidebar
 *  label ends in the anatomical name; skip it when the plain name is it. */
function partNames(id) {
  const p = partOf(id);
  const plain = p?.short || p?.name || id;
  const anat = /\(([^)]+)\)\s*$/.exec(p?.label || '')?.[1];
  return [plain, anat && anat.toLowerCase() !== plain.toLowerCase() ? anat : ''];
}
function fullName(id) {
  const [plain, anat] = partNames(id);
  return anat ? `${plain} (${anat})` : plain;
}
/** For narrow name columns: the anatomical name on its own small line. */
function nameHtml(id) {
  const [plain, anat] = partNames(id);
  return escHtml(plain) + (anat ? `<small>${escHtml(anat)}</small>` : '');
}
const nameOf = (r) => escHtml(fullName(r.part));

/** A reason that came from something on screen links back to it. */
function refLink(text, ref = '') {
  return /^(note|answer):/.test(ref)
    ? `<a href="#" class="eng-ref" data-ref="${escHtml(ref)}">${escHtml(text)}</a>`
    : escHtml(text);
}

function evidenceText(ev) {
  const parts = [['answer', 'answer'], ['note', 'note'], ['log', 'log']]
    .filter(([k]) => ev[k] > 0)
    .map(([k, w]) => `${ev[k]} ${w}${ev[k] !== 1 ? 's' : ''}`);
  return parts.length ? `based on ${parts.join(' \u00b7 ')}`
    : 'based only on what you rarely do';
}

/** The events that push this part each way, and which way the target
 *  wants it. Each can be logged from here. */
function leversHtml(r) {
  const m = r.moves || {};
  const t = onBrain(App.state?.target_engagement)
    .find(x => x.part === r.part);
  const gap = t ? t.score - r.score : 0;
  const want = Math.abs(gap) < 0.04 ? null : gap > 0 ? 'up' : 'down';
  const line = (key, word) => !m[key]?.length ? '' : `
    <div class="eng-lever${want === key ? ' aim' : ''}">
      <span>${word}${want === key ? ' <em>\u00b7 toward your target</em>' : ''}</span>
      <div>${m[key].map(eid => `<button class="eng-log" data-log="${
        escHtml(eid)}" title="Log one now">+ ${
        escHtml(App.eventById[eid]?.label || eid)}</button>`).join('')}</div>
    </div>`;
  return `
    <div class="eng-levers">
      <div class="cur-group">WHAT MOVES IT</div>
      ${line('up', 'Raises it')}${line('down', 'Lowers it')}
    </div>`;
}

/** One part's row. Current explains with "because you ..."; Target lists
 *  the aims behind it, which do not read as a sentence. */
function engRow(r, target = false) {
  const p = partOf(r.part);
  const w = Math.min(50, Math.abs(r.score) / ENG_FULL * 50);
  const reasons = r.why?.reasons || [];
  const links = reasons.map((x, i) => refLink(x, r.why?.refs?.[i]));
  const because = !reasons.length ? ''
    : target ? ` <span>From: ${links.join(' \u00b7 ')}.</span>`
    : ` <span>Mostly because you ${joinAnd(links)}.</span>`;
  const tr = target ? null : trendOf(r.part);
  const ev = !target && r.why?.evidence
    ? `<div class="eng-ev">${evidenceText(r.why.evidence)}${
      tr ? ` \u00b7 ${tr.text}` : ''}</div>` : '';
  const why = r.why?.meaning
    ? `<div class="eng-why">${escHtml(cap(r.why.meaning))}.${because}${ev}</div>`
    : '';
  const on = r.part === App.curFocus;
  return `
    <div class="eng-row ${r.level}" role="button" tabindex="0"
         data-part="${r.part}">
      <div class="eng-top">
        <span class="eng-sw" style="background:${p.color}"></span>
        <span class="eng-name">${nameHtml(r.part)}</span>
        <span class="eng-bar"><i style="width:${w.toFixed(1)}%"></i></span>
        ${tr ? tr.svg : ''}
        <span class="eng-arrow">${r.level === 'more' ? '\u25b2' : '\u25bc'}</span>
      </div>${why}${on && !target ? leversHtml(r) + disputeHtml(r) : ''}
    </div>`;
}

function engGroups(rows, target, unknownText) {
  const more = rows.filter(r => r.level === 'more');
  const less = rows.filter(r => r.level === 'less');
  const balanced = rows.filter(r => r.level === 'typical' && r.basis === 'balanced');
  const unknown = rows.filter(r => r.level === 'typical' && r.basis !== 'balanced');
  return `
    ${more.length ? `<div class="cur-group">MORE ENGAGED</div>${more.map(r => engRow(r, target)).join('')}` : ''}
    ${less.length ? `<div class="cur-group">LESS ENGAGED</div>${less.map(r => engRow(r, target)).join('')}` : ''}
    ${balanced.length ? `<div class="cur-group">BALANCED</div>
      <div class="eng-typical">${balanced.map(nameOf).join(', ')}
        <span>What you described pulls these both ways, about equally, so
          they come out near typical.</span></div>` : ''}
    ${unknown.length && rows.length ? `<div class="cur-group">${unknownText[0]}</div>
      <div class="eng-typical">${unknown.map(nameOf).join(', ')}
        <span>${unknownText[1]}</span></div>` : ''}`;
}

function notesHtml(notes, delAttr, emptyText) {
  if (!notes.length) return `<div class="eng-typical">${emptyText}</div>`;
  return notes.map((n, i) => {
    const read = Object.keys(n.events || {})
      .map(e => App.eventById[e]?.label.toLowerCase() || e).join(', ');
    const fx = noteTags(i);
    return `
      <div class="cur-note${i === App.curNote ? ' on' : ''}" data-note="${i}"
           role="button" tabindex="0" title="Show what this touches">
        <div>\u201c${escHtml(n.text)}\u201d
          <span>read as ${escHtml(read)}</span>
          ${fx.length ? `<div class="cur-fx">${fx.map(e => `
            <em class="${e.dir}" data-part="${e.part}"><i style="background:${partOf(e.part).color}"></i>${
              escHtml(fullName(e.part))} ${
              e.dir === 'more' ? '\u25b2' : '\u25bc'}</em>`).join('')}</div>`
            : ''}</div>
        <button class="cur-x" ${delAttr === 'data-del' ? 'data-edit'
          : 'data-edit-goal'}="${i}" title="Fix how this was read">\u270e</button>
        <button class="cur-x" ${delAttr}="${i}" title="Remove">\u2715</button>
      </div>`;
  }).join('');
}

function targetPanelHtml(rows) {
  const tp = App.state?.target_profile;
  const answered = Object.keys(tp?.answers || {}).length;
  return `
    <div class="vis-kicker">TARGET</div>
    <div class="vis-title">Who you want to become</div>
    <div class="cur-sub">how that brain would be engaged, compared with typical
      <span class="tag model">MODEL</span></div>
    ${tp ? '' : `
      <div class="cur-empty">No target yet, so progress is measured against
        the app's default aims. Describe who you want to be above - "I want to
        pause before I answer" - or answer the situations the way you would
        like to react.</div>`}
    ${engGroups(rows, true, ['NOT PART OF YOUR AIMS YET',
      'What you are aiming for moves these too little to tell, or not at '
      + 'all.'])}
    <div class="cur-group">YOUR AIMS</div>
    ${notesHtml(reactionsNotes(tp), 'data-del-goal',
      'Nothing described yet. Add aims above, as many as you like.')}
    <div class="vis-row">
      <button class="cur-ghost" data-quiz="target">${answered
        ? 'Edit situations' : 'Answer situations'}</button>
      ${tp ? '<button class="cur-ghost" data-clear-target>Clear</button>' : ''}
    </div>
    <div class="cur-foot">
      <span>Your target is what "gap closed" measures. Unwanted loops never go
        below the app's floor: a working alarm system is part of any healthy
        target.</span>
    </div>`;
}

const reactionsNotes = (p) => p?.notes || [];

function comparePanelHtml(cur, tgt) {
  const byPart = (rows) => Object.fromEntries(rows.map(r => [r.part, r]));
  const c = byPart(cur), t = byPart(tgt);
  const ids = [...new Set([...Object.keys(c), ...Object.keys(t)])];
  const mark = (r) => !r || r.level === 'typical' ? '\u2013'
    : r.level === 'more' ? '\u25b2 more' : '\u25bc less';
  const rows = ids.map(id => {
    const gap = (t[id]?.score || 0) - (c[id]?.score || 0);
    return { id, gap, c: c[id], t: t[id] };
  }).sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
  // Same band the engagement read-out uses for "about typical".
  const label = (g) => Math.abs(g) < 0.04 ? 'on track'
    : g > 0 ? 'needs more' : 'needs less';
  return `
    <div class="vis-kicker">COMPARE</div>
    <div class="vis-title">Where you are, and where you are aiming</div>
    <div class="cur-sub">left brain: now \u00b7 right brain: target
      <span class="tag model">MODEL</span></div>
    ${!App.state?.target_profile ? `<div class="cur-empty">No target yet.
      Describe who you want to be in the Target tab and the right-hand brain
      will show it.</div>` : ''}
    ${!App.state?.profile ? `<div class="cur-empty">Nothing described about
      how you react yet, so the left-hand brain is typical everywhere. Add
      that in the Current tab.</div>` : ''}
    ${rows.length ? `
      <div class="cmp-row head"><span>PART</span><span>NOW</span>
        <span>TARGET</span><span style="text-align:right">GAP</span></div>
      ${rows.map(r => `
        <div class="cmp-row" data-part="${r.id}">
          <span class="eng-top"><span class="eng-sw"
            style="background:${partOf(r.id).color}"></span>
            <span>${nameHtml(r.id)}</span></span>
          <span>${mark(r.c)}</span><span>${mark(r.t)}</span>
          <span class="gap ${label(r.gap) === 'on track' ? 'on' : ''}">${
            label(r.gap)}</span>
        </div>`).join('')}` : ''}`;
}

function overnightHtml() {
  const s = App.state;
  const o = s?.overnight;
  if (!o?.pathways?.length || App.overnightSeen === s.day) return '';
  const sleep = o.sleep >= 0.7 ? 'good' : o.sleep >= 0.45 ? 'middling' : 'poor';
  return `
    <div class="cur-night">
      <div class="cur-night-top">
        <span class="cur-group">WHILE YOU SLEPT</span>
        <span class="tag model">MODEL</span>
        <button class="cur-x" data-night-close title="Dismiss">\u00d7</button>
      </div>
      ${o.pathways.slice(0, 4).map(p => `
        <div class="cur-night-row">
          <span>${escHtml(p.label)}</span>
          <span>${Math.round(p.kept * 100)}% of yesterday's
            ${p.gain > 0 ? 'gain' : 'drop'} held</span>
        </div>`).join('')}
      <div class="cur-night-foot">The night was ${sleep} sleep. In this model
        change sets overnight: what consolidates resists fading, and better
        sleep keeps more. On the brain, the solid core of a connection is the
        part that has set.</div>
    </div>`;
}

function currentPanelHtml(rows, hasProfile) {
  return `
    <div class="vis-kicker">CURRENT</div>
    <div class="vis-title">How your brain is engaged</div>
    <div class="cur-sub">compared with a typical starting point
      <span class="tag model">MODEL</span></div>
    ${overnightHtml()}
    ${hasProfile ? '' : `
      <div class="cur-empty">Nothing to show yet: the app knows nothing about
        you, so every part is "typical". Describe how you tend to react above,
        and each thing you add shows which parts your habits lean on more or
        less than a typical person's.</div>`}
    ${engGroups(rows, false, ['NOT ENOUGH INFORMATION YET',
      'What you have said moves these too little to tell, or not at all. '
      + 'Describe more, or answer a few situations.'])}
    <div class="cur-group">YOUR PATTERNS</div>
    ${notesHtml(App.state?.profile?.notes || [], 'data-del',
      'Nothing described yet. Add as many as you like above, whenever you '
      + 'think of one - each refines the picture.')}
    <div class="vis-row">
      <button class="cur-ghost" data-quiz="current">${hasProfile && Object.keys(
        App.state.profile.answers || {}).length
          ? 'Edit situations' : 'Answer situations'}</button>
    </div>
    ${hasProfile ? habitsHtml(App.state?.habits || []) : ''}
    ${changeHtml()}
    ${hasProfile ? `
      <details class="cur-base">
        <summary>What counts as "typical"?</summary>
        <p>Everything you say is turned into how often you do each kind of
          thing - going over things, lashing out, pausing, reframing and so on
          - and six weeks of that are run through the app's fixed rules.
          "Typical" is the same six weeks for an invented person who answers
          every situation with an even mix of all its options. Anything you
          have not said counts as typical, so it moves nothing. It is a
          reference point inside this model, not a population average, and
          more engaged is not better or worse.</p>
      </details>
      <div class="cur-foot">
        <span>Built from what you have described and logged. Not a scan, not
          an assessment.</span>
        <button class="cur-ghost" data-clear>Clear</button>
      </div>` : ''}`;
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const perWeek = (x) => (Math.round(x * 10) / 10).toString();

function habitWhere(h) {
  if (h.progress == null) return '';
  if (h.progress >= 1) return 'at or past your aim';
  if (h.progress < -0.05) return 'moving away from your aim';
  if (h.progress < 0.05) return 'not moved yet';
  return `${Math.round(h.progress * 20) * 5}% of the way`;
}

function habitRow(h) {
  const p = h.progress == null ? null : Math.max(0, Math.min(1, h.progress));
  return `
    <div class="hab-row${h.progress != null && h.progress < -0.05 ? ' away' : ''}">
      <div class="hab-top">
        <span class="hab-name">${escHtml(h.label)}</span>
        <span class="hab-num">${perWeek(h.start)} \u2192 <b>${perWeek(h.now)}</b>
          <i>a week</i></span>
        <button class="eng-log" data-log="${escHtml(h.event)}"
          title="Log one now">+</button>
      </div>
      ${p == null ? '' : `<div class="hab-bar"><i style="width:${
        (p * 100).toFixed(0)}%"></i></div>`}
      <div class="hab-sub">${h.target != null
        ? `aim ${perWeek(h.target)} a week \u00b7 ${habitWhere(h)}` : ''}${
        h.logged ? `${h.target != null ? ' \u00b7 ' : ''}logged ${h.logged}\u00d7 `
          + 'in the last 4 weeks' : ''}</div>
    </div>`;
}

/** How often you do each thing: as described, now, and as aimed for. */
function habitsHtml(rows) {
  if (!rows.length) return '';
  const aimed = rows.filter(h => h.progress != null);
  const rest = rows.filter(h => h.progress == null);
  const hasTarget = rows.some(h => h.target != null);
  return `
    <div class="cur-group">YOUR HABITS</div>
    <div class="hab-intro">Times a week: what you described \u2192 now, updated
      by what you log.${hasTarget ? ''
        : ' Set a target to see how far each has come.'}</div>
    ${aimed.map(habitRow).join('')}
    ${rest.length ? `<details class="cur-base"${aimed.length ? '' : ' open'}>
      <summary>${aimed.length ? 'Habits your target does not change'
        : 'All habits'} (${rest.length})</summary>
      ${rest.map(habitRow).join('')}</details>` : ''}
    <div class="hab-foot">Assumes that on a day you log, you log what
      happened. Only the situations you log about are updated; the rest keep
      what you described.</div>`;
}

/* ---- Change over time ---------------------------------------------------
 * One log moves the model a point or two, too little to see. These views
 * show change at true scale by showing a lot of it: everything since day 0,
 * or today's habits run forward. */
const HORIZONS = [30, 90, 180];
const pts = (x) => `${x >= 0 ? '+' : '\u2212'}${Math.abs(x * 100).toFixed(1)}`;
const projKey = (s) => `${s.day}|${s.log_count}|${s.progress}|${s.alignment}`;
const progWords = (x) => `${progressBand(x * 100)} (~${Math.round(x * 20) * 5}%)`;
const projectionBand = (band) => band
  ? `${(band[0] * 100).toFixed(1)}–${(band[1] * 100).toFixed(1)}%` : '';

async function ensureProjection() {
  const k = projKey(App.state);
  if (App.projection?.key === k) return App.projection;
  const p = await API.projection();
  p.key = k;
  if (App.state && projKey(App.state) === k) App.projection = p;
  return p;
}

function refreshProjection() {
  const k = projKey(App.state);
  if (App.projLoading || App.projFailed === k) return;
  App.projLoading = ensureProjection()
    .catch(() => { App.projFailed = k; toast('Could not run the projection'); })
    .finally(() => {
      App.projLoading = null;
      if (typeof App.pathView === 'number') renderReading();
    });
}

const projFresh = () => App.state && App.projection?.key === projKey(App.state);

function projAt(h, scen = App.projScenario) {
  if (!projFresh()) return null;
  const p = App.projection;
  return (scen === 'stop' ? p.stop : p.keep).find(x => x.days === h) || null;
}

function changeHtml() {
  const s = App.state;
  const rows = s?.practice || [];
  if (!rows.length && !s?.log_count && !s?.profile) return '';
  const v = App.pathView;
  const btn = (val, label) => `<button data-pview="${val}"${
    String(v) === String(val) ? ' class="active"' : ''}>${label}</button>`;
  const most = Math.max(1e-6, ...rows.map(r => Math.max(r.gained, r.lost)));
  return `
    <div class="cur-group">CHANGE OVER TIME <span class="tag model">MODEL</span></div>
    <div class="segmented pv-seg">${btn('since', 'Since day 0')}${
      HORIZONS.map(h => btn(h, `In ${h} days`)).join('')}</div>
    <div class="pv-out">${pviewHtml()}</div>
    ${rows.length ? `
      <div class="hab-intro">Each pathway since day 0. A log counts
        <b>toward</b> when its rule pushes the pathway toward your aim and
        <b>away</b> when it pushes back. Points are the model's 0-1 scale
        \u00d7100. Click one to show only it.</div>
      ${rows.map(r => practiceRow(r, most)).join('')}` : ''}`;
}

function practiceRow(r, most) {
  const w = (x) => (Math.min(1, x / most) * 50).toFixed(1);
  const week = r.toward_week || r.away_week
    ? ` \u00b7 this week ${r.toward_week} / ${r.away_week}` : '';
  return `
    <div class="pr-row${App.pathFocus === r.pathway ? ' on' : ''}"
         data-pathway="${r.pathway}" role="button" tabindex="0">
      <div class="hab-top">
        <span class="eng-sw" style="background:${r.color}"></span>
        <span class="hab-name">${escHtml(r.label)}</span>
        <span class="hab-num"><b>${r.toward}</b> <i>toward</i> \u00b7
          <b>${r.away}</b> <i>away</i></span>
      </div>
      <div class="pr-bar" title="left: lost to logs that pushed back \u00b7 right: gained from logs">
        <i class="lost" style="width:${w(r.lost)}%"></i>
        <i class="gain" style="width:${w(r.gained)}%"></i>
      </div>
      <div class="hab-sub">logs ${pts(r.gained)} / ${pts(-r.lost)} \u00b7
        nights ${pts(r.nights)} \u00b7 net <b>${pts(r.net)}</b> toward your aim
        \u00b7 ${Math.round(r.consolidated * 100)}% set${week}</div>
    </div>`;
}

function pviewHtml() {
  const s = App.state, v = App.pathView;
  if (v == null) {
    return `<div class="hab-intro">Pick one to draw it on the brain at true
      scale - no enlarging.</div>`;
  }
  if (v === 'since') {
    return `<div class="hab-intro">Day 0 \u2192 day ${s.day}: every log and
      every night added up. <i class="pv-up"></i> stronger
      <i class="pv-down"></i> weaker, thickness = size of the change. One log
      is a point or two; this is all of them.</div>`;
  }
  if (!projFresh()) return '<div class="hab-intro">Running the model forward\u2026</div>';
  const p = App.projection;
  if (!p.keep.length) {
    return `<div class="hab-intro">Nothing to run forward yet. Log a few
      things, or describe how you react, and this shows where that leads.</div>`;
  }
  const keep = projAt(v, 'keep'), stop = projAt(v, 'stop');
  const at = projAt(v);
  const seg = (k, label) => `<button data-scen="${k}"${
    App.projScenario === k ? ' class="active"' : ''}>${label}</button>`;
  const moved = Object.entries(at.pathways)
    .map(([id, then]) => ({ id, then, d: then - (s.pathways[id] ?? then) }))
    .filter(x => Math.abs(x.d) >= 0.005)
    .sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 4);
  const label = (id) => App.model.pathways.find(x => x.id === id)?.label || id;
  const toward = (x) => x.d * ((s.pathway_targets[x.id] ?? 0) - (s.pathways[x.id] ?? 0)) > 0;
  const progressValue = (x) => `${progWords(x.progress)}${
    x.progress_band ? `<small class="pv-range">10th–90th percentile ${
      projectionBand(x.progress_band)}</small>` : ''}`;
  const rates = Object.entries(p.rates).sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([id, r]) => `${escHtml(App.eventById[id]?.label || id)} ${perWeek(r)}`)
    .join(' \u00b7 ');
  return `
    <div class="segmented pv-seg sub">${seg('keep', 'Keep this up')}${
      seg('stop', 'Stop practising')}</div>
    <div class="cur-night-row"><span>Now</span><span>${progWords(s.progress)}</span></div>
    <div class="cur-night-row"><span>In ${v} days, keep this up</span>
      <span>${progressValue(keep)}</span></div>
    <div class="cur-night-row"><span>In ${v} days, practice stopped</span>
      <span>${progressValue(stop)}</span></div>
    ${moved.map(x => `
      <div class="cur-night-row"><span>${escHtml(label(x.id))}</span>
        <span class="${toward(x) ? 'pv-t' : 'pv-a'}">${x.d > 0 ? 'stronger'
          : 'weaker'} by ${Math.abs(x.d * 100).toFixed(1)} \u00b7 ${
          toward(x) ? 'toward' : 'away from'} your aim${
          at.pathway_bands?.[x.id] ? `<small class="pv-range">projected level ${
            projectionBand(at.pathway_bands[x.id])}</small>` : ''}</span></div>`).join('')}
    <div class="hab-intro">Runs your habits a week (${rates}) through the same
      rules, on a copy - nothing is recorded. "Stop practising" keeps
      everything else and takes the practice out. Based on what was
      ${p.basis}. The ranges are the 10th–90th percentiles across 24 seeded
      model runs. They show model variability, not a confidence interval or
      personal prediction.</div>`;
}

function pathViewEdges(moved) {
  const part = App.curFocus;
  const nodes = part ? App.model.nodes
    .filter(n => n.structure === part || n.cortex === part).map(n => n.id) : null;
  return App.model.edges
    .filter(e => (!App.pathFocus || e.pathway === App.pathFocus)
      && (!nodes || nodes.includes(e.src) || nodes.includes(e.dst))
      && moved(e.id))
    .map(e => e.id);
}

/** Held until switched off: Since day 0, or a projected day. */
function drawPathView() {
  const s = App.state, C = App.circuits;
  if (App.pathView === 'since') {
    const a = s.start_weights, b = s.weights;
    const ids = pathViewEdges(id => Math.abs((b[id] ?? 0) - (a[id] ?? 0)) > 0.002);
    C.spotlight(ids.length ? ids : null);
    C.setCores(null);
    C.showDelta(a, b);
    return;
  }
  if (C.deltaMode && !App.compareOn) C.showDelta(null, null);
  const p = projAt(App.pathView);
  if (!p) {
    if (!projFresh()) refreshProjection();
    C.spotlight(null);
    return;
  }
  const now = s.weights;
  const ids = pathViewEdges(id => Math.abs((p.weights[id] ?? 0) - (now[id] ?? 0)) > 0.005);
  if (!ids.length) { C.spotlight(null); return; }
  C.spotlight(ids);
  const key = `${App.pathView}|${App.projScenario}|${App.projection.key}|${ids}`;
  if (App.pathDrawn !== key) {
    App.pathDrawn = key;
    C.animate(now, p.weights, 1.8);
  } else C.update(p.weights);
  C.setCores(p.consolidation, p.weights);
}

function leavePathView() {
  App.pathView = null;
  App.pathFocus = null;
  App.pathDrawn = null;
  if (!App.compareOn) App.circuits.showDelta(null, null);
}

function setPathView(v) {
  const val = v === 'since' ? v : Number(v);
  if (App.pathView === val) leavePathView();
  else { App.pathView = val; App.pathDrawn = null; }
  if (App.pathView == null) App.circuits.spotlight(null);
  renderReading();
}

/** Under a log's result: its place in the tally, and where the habit leads. */
async function fillAhead(ev, deltas, s) {
  const seq = ++App.logSeq;
  const out = [];
  const acts = ev.activations || {};
  const touched = (s.practice || []).filter(r => acts[r.pathway])
    .sort((a, b) => Math.abs(acts[b.pathway]) - Math.abs(acts[a.pathway]))
    .slice(0, 3);
  for (const r of touched) {
    const name = escHtml(r.label);
    if (acts[r.pathway] * r.sign > 0) {
      out.push(`<div class="cur-night-row"><span>${name}</span>
        <span>log #${r.toward} toward your aim</span></div>`);
      continue;
    }
    // Set-back measured in the currency of practice, so a slip reads as
    // ground lost rather than as failure.
    const d = Math.abs(deltas.find(x => x.pathway === r.pathway)?.delta || 0);
    const per = r.toward >= 3 ? r.gained / r.toward : 0;
    const n = per > 0 ? Math.round(d / per) : 0;
    out.push(`<div class="cur-night-row"><span>${name}</span>
      <span class="pv-a">pushed back${n >= 1 ? ` about ${n} ${
        n > 1 ? "logs'" : "log's"} worth` : ''} (${r.away} so far)</span></div>`);
  }
  const paint = (ahead) => {
    const el = $('#logAhead');
    if (!el || seq !== App.logSeq) return;
    el.innerHTML = out.length || ahead ? `
      <div class="p-sec">
        <div class="p-h" style="color:var(--txt-faint)">WHERE THIS LEADS
          <span class="tag model">MODEL</span></div>
        ${out.join('')}${ahead}
      </div>` : '';
    const b = el.querySelector('[data-ahead]');
    if (b) b.onclick = () => {
      LogDock.close();
      if (App.mode !== 'simulation') setMode('simulation');
      if (App.viewState !== 'current') selectState('current');
      setPathView(90);
      $('#curPanel .pv-seg')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    };
  };
  paint('');
  let p;
  try { p = await ensureProjection(); } catch { return; }
  if (!p.keep.length) return;
  const h = 90;
  const keep = p.keep.find(x => x.days === h), stop = p.stop.find(x => x.days === h);
  const rate = p.rates[ev.id];
  paint(`
    ${rate ? `<div class="hab-intro">At about ${perWeek(rate)} a week, the
      next ${h} days hold roughly ${Math.round(rate * h / 7)} more of these.
      This one is a single step; the path is all of them.</div>` : ''}
    <div class="cur-night-row"><span>Now</span><span>${progWords(s.progress)}</span></div>
    <div class="cur-night-row"><span>In ${h} days, keep this up</span>
      <span>${progWords(keep.progress)}</span></div>
    <div class="cur-night-row"><span>In ${h} days, practice stopped</span>
      <span>${progWords(stop.progress)}</span></div>
    <button class="cur-ghost" data-ahead style="margin-top:8px">Show on the
      brain</button>`);
}
const joinAnd = (xs) => xs.length < 2 ? xs.join('')
  : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;

async function openQuiz(kind = 'current', showQ = null) {
  const target = kind === 'target';
  let qs;
  try { qs = (await API.profileQuestions()).questions; }
  catch { toast('Could not load the questions'); return; }
  const prev = (target ? App.state?.target_profile : App.state?.profile)
    ?.answers || {};
  const had = (qid, oid) => [].concat(prev[qid] || []).includes(oid);
  Modal.open(`
    <div class="m-title">${target ? 'How you would like to react'
                                  : 'How you tend to react'}</div>
    <div class="m-sub">Everyday situations - all optional. ${target
      ? 'Tick what you would like to do instead - the person you are aiming to be.'
      : 'Tick everything you usually do, not what you wish you did.'} Skip any
      that do not fit, and they count as typical. Written for this app: not a
      validated questionnaire, and nothing here is an assessment.</div>
    <form id="quiz">
      ${qs.map(q => `
        <fieldset class="qz-q" data-q="${q.id}">
          <legend>${escHtml(q.prompt)}</legend>
          <div class="qz-opts">${q.options.map(o => `
            <label class="qz-opt">
              <input type="checkbox" name="${q.id}" value="${o.id}"
                ${had(q.id, o.id) ? 'checked' : ''}>
              <span>${escHtml(o.label)}</span>
            </label>`).join('')}
          </div>
        </fieldset>`).join('')}
      <fieldset class="qz-q">
        <legend>${target ? 'Anything else you are aiming for?'
          : 'Anything else about how you handle things?'} <i>optional - you
          can add more later from the box above the panel</i></legend>
        <textarea id="quizText" rows="3" maxlength="1200" data-grow
          placeholder="${target
            ? 'e.g. I want to stay calm when someone criticises me'
            : 'e.g. I bottle things up at work and then snap at home'}"
          ></textarea>
      </fieldset>
      <div class="qz-foot">
        <span id="quizMsg">Read by the same classifier as your log entries -
          the language model when one is running. Anything you have logged
          is kept.</span>
        <button type="submit" class="primary">${target
          ? 'Update my target' : 'Update my current brain'}</button>
      </div>
    </form>`);

  const shown = showQ && $(`#quiz [data-q="${CSS.escape(showQ)}"]`);
  if (shown) {
    shown.scrollIntoView({ block: 'center' });
    shown.classList.add('flash');
  }

  $('#quiz').onsubmit = async (e) => {
    e.preventDefault();
    const form = e.target;
    const answers = {};
    for (const q of qs) {
      const ids = $$(`#quiz input[name="${q.id}"]:checked`).map(i => i.value);
      if (ids.length) answers[q.id] = ids;
    }
    const btn = form.querySelector('button[type=submit]');
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Building\u2026';
    try {
      const r = await (target ? API.setTarget : API.setProfile)(
        answers, $('#quizText').value);
      Modal.close();
      App.replayDay = null;
      App.curFocus = null;
      App.curNote = null;
      applyState(r.state);
      const read = (r.read_as || []).map(p => p.label.toLowerCase());
      toast(target ? 'Target updated' : 'Current brain updated from your answers',
        read.length ? `Your description read as: ${read.join(', ')}` : '');
    } catch (err) {
      btn.disabled = false;
      btn.textContent = label;
      toast('Could not build it', err.message);
    }
  };
}

async function runVisualise(text) {
  // Pressing Explain twice can put two replies in the air. Only the newest
  // is allowed to touch the brain.
  const seq = ++App.visSeq;
  const out = $('#visOut'), go = $('#composerGo');
  go.disabled = true;
  out.innerHTML = '';

  explainVisualise(text, [], seq).finally(() => {
    if (seq === App.visSeq) go.disabled = false;
  });
}

/** Prose from the model, anatomy from the rules. */
async function explainVisualise(text, eventIds, seq) {
  const out = $('#visOut');
  out.innerHTML = '<p class="vis-prose">working out what that engages…</p>';
  let a;
  try {
    a = (await API.analyse(text, eventIds, true)).analysis;
  } catch {
    if (seq === App.visSeq) out.innerHTML = '';
    return;
  }
  if (seq !== App.visSeq) return;

  const parts = [...(a.structures || []), ...(a.regions || [])]
    .filter(id => App.structById[id] || App.cortexById[id]);

  App.visLevels = a.levels || {};
  App.visMoves = (a.moves || []).filter(m => partOf(m.part));
  App.visAlts = a.alternatives || [];
  App.visEvents = (a.events || []).map(e => e.id);
  App.visPreview = null;
  for (const m of App.visMoves) if (!parts.includes(m.part)) parts.push(m.part);
  const moved = new Set(App.visMoves.map(m => m.part));
  const others = parts.filter(id => !moved.has(id));
  visualiseParts(parts);

  out.innerHTML = `
    <p class="vis-prose">${linkifyParts(a.text, parts)}</p>
    ${App.visMoves.length ? `
      <div class="vis-head">
        <span class="cur-group">IN THIS MOMENT \u00b7 IN THIS MODEL</span>
        <button class="vis-replay" title="Play the circuit again">\u21bb Replay</button>
      </div>
      ${App.visMoves.map(momentRow).join('')}` : ''}
    ${others.length ? `
      <div class="cur-group">${App.visMoves.length ? 'ALSO INVOLVED' : 'INVOLVED'}</div>
      <div class="vis-parts">${others.map(id =>
        `<button class="vis-part" data-part="${id}">${partLabel(id)}</button>`
      ).join('')}</div>` : ''}
    ${App.visAlts.length ? `
      <div class="cur-group">TRY A DIFFERENT RESPONSE</div>
      <div class="vis-alts">${App.visAlts.map((x, i) =>
        `<button class="vis-alt" data-alt="${i}">${escHtml(tryLabel(x))}</button>`
      ).join('')}</div>
      <div class="vis-alt-note" id="visAltNote">Point at one to see what it
        would push instead. Nothing is recorded.</div>` : ''}`;

  // Every part stays lit; clicking one brings it forward and fades the rest.
  // Clicking it again, or empty space, puts them all back on equal terms.
  $$('#visOut .vis-part, #visOut .eng-row').forEach(b => {
    const go = () => {
      const id = b.dataset.part;
      focusVisualisePart(App.visFocus === id ? null : id);
    };
    if (b.classList.contains('eng-row')) pressable(b, go); else b.onclick = go;
  });
  const replay = $('#visOut .vis-replay');
  if (replay) replay.onclick = playMoment;
  $$('#visOut .vis-alt').forEach(b => {
    const alt = App.visAlts[Number(b.dataset.alt)];
    const on = () => previewResponse(alt, b);
    const off = () => { if (!b.classList.contains('pinned')) previewResponse(null); };
    b.onmouseenter = on;
    b.onfocus = on;
    b.onmouseleave = off;
    b.onblur = off;
    // Touch has no hover: a tap keeps the preview until tapped again.
    b.onclick = () => {
      const pin = !b.classList.contains('pinned');
      $$('#visOut .vis-alt').forEach(x => x.classList.remove('pinned'));
      b.classList.toggle('pinned', pin);
      previewResponse(pin ? alt : null, pin ? b : null);
    };
  });
  playMoment();
}

const TRY_LABELS = {
  name_emotion: 'Name the feeling', pause: 'Pause first',
  disengage: 'Let the thought go', reappraisal: 'Reframe it',
  mindful_practice: 'Slow breathing', regulated_success: 'Stay steady',
};
const tryLabel = (x) => TRY_LABELS[x.event] || x.label;

// Bars use a square root: the amygdala's push dwarfs everything else.
const momentBar = (m) => Math.sqrt(m.strength) * 50;

function momentRow(m) {
  const p = partOf(m.part);
  return `
    <div class="eng-row ${m.dir}" role="button" tabindex="0"
         data-part="${m.part}">
      <div class="eng-top">
        <span class="eng-sw" style="background:${p.color}"></span>
        <span class="eng-name">${nameHtml(m.part)}</span>
        <span class="eng-bar"><i style="width:${momentBar(m).toFixed(1)}%"></i></span>
        <span class="eng-arrow">${m.dir === 'more' ? '\u25b2' : '\u25bc'}</span>
      </div>
      ${m.meaning ? `<div class="eng-why">${escHtml(cap(m.meaning))}.</div>` : ''}
    </div>`;
}

/** The connections a set of events strengthens and weakens, by edge id. */
function momentEdges(eventIds) {
  const amt = {};
  for (const id of eventIds) {
    for (const [pid, a] of Object.entries(App.eventById[id]?.activations || {})) {
      amt[pid] = (amt[pid] || 0) + a;
    }
  }
  const up = [], down = [];
  for (const e of App.model.edges) {
    const a = amt[e.pathway];
    if (a && Math.abs(a) >= 0.05) (a > 0 ? up : down).push(e.id);
  }
  return { up, down };
}

/** Draw the moment's circuit: strengthened routes bright with pulses running
 *  down them, weakened ones thin and dim. `hold` 0 keeps it up. */
function showCircuit(eventIds, hold = 5200) {
  const { up, down } = momentEdges(eventIds);
  clearTimeout(App.pathTimer);
  App.pathTimer = null;
  if (!up.length && !down.length) { App.circuits.spotlight(null); return 0; }
  const w = {};
  for (const id of up) w[id] = 0.8;
  for (const id of down) w[id] = 0.15;
  App.circuits.spotlight([...up, ...down]);
  App.circuits.update(w);
  App.circuits.cascade(up, { stagger: 0.13, speed: 1.35 });
  if (hold) App.pathTimer = setTimeout(endPaths, hold);
  return up.length + down.length;
}

function stopMoment() {
  App.visTimers.forEach(clearTimeout);
  App.visTimers = [];
  App.visReveal = Infinity;
}

/** The circuit first, then each part takes its arrow, biggest push first. */
function playMoment() {
  stopMoment();
  if (!App.visMoves.length) return;
  App.visPreview = null;
  $$('#visOut .vis-alt').forEach(x => x.classList.remove('pinned'));
  const t0 = showCircuit(App.visEvents) ? 900 : 0;
  App.visReveal = 0;
  applyVisualise(App.visFocus);
  App.visMoves.forEach((m, i) => App.visTimers.push(setTimeout(() => {
    App.visReveal = i + 1;
    applyVisualise(App.visFocus);
    $(`#visOut .eng-row[data-part="${m.part}"]`)?.classList.add('lit');
  }, t0 + i * 170)));
  App.visTimers.push(setTimeout(() => {
    App.visReveal = Infinity;
    $$('#visOut .eng-row.lit').forEach(r => r.classList.remove('lit'));
  }, t0 + App.visMoves.length * 170 + 900));
}

/** Show what a different response would push instead, or null to go back. */
function previewResponse(alt, btn = null) {
  stopMoment();
  App.visPreview = alt;
  $$('#visOut .vis-alt').forEach(x => x.classList.toggle('on', x === btn));
  const note = $('#visAltNote');
  if (alt) {
    for (const m of alt.moves) {
      if (partOf(m.part) && !App.visParts.includes(m.part)) App.visParts.push(m.part);
    }
    showCircuit([alt.event], 0);
    if (note) note.textContent = `${tryLabel(alt)}: ` + alt.moves
      .filter(m => partOf(m.part)).slice(0, 4)
      .map(m => `${partLabel(m.part)} ${m.dir === 'more' ? '\u25b2' : '\u25bc'}`)
      .join(' \u00b7 ') + ' (in this model)';
  } else {
    endPaths();
    if (note) note.textContent = 'Point at one to see what it would push '
      + 'instead. Nothing is recorded.';
  }
  applyVisualise(App.visFocus);
}

// The camera stays put: re-aiming it at the part moved the whole brain.
function focusVisualisePart(id) {
  $$('#visOut .vis-part, #visOut .eng-row').forEach(x =>
    x.classList.toggle('on', !!id && x.dataset.part === id));
  applyVisualise(id);
}

const escHtml = (s) => s.replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/**
 * Turn every brain part the prose names into a control that finds it.
 *
 * Done here rather than by asking the model for markers: small local models
 * emit them unreliably, and this has to work identically for the rule-based
 * text produced when no model is connected.
 */
function linkifyParts(text, ids) {
  const byName = new Map();
  for (const id of ids) {
    const p = App.structById[id] || App.cortexById[id];
    for (const n of [id, p?.short, p?.name]) {
      if (n) byName.set(String(n).toLowerCase(), id);
    }
  }
  const safe = escHtml(text);
  if (!byName.size) return safe;

  // Longest first, so "sgACC" is claimed before "ACC" can take its tail.
  const names = [...byName.keys()]
    .sort((a, b) => b.length - a.length)
    .map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const rx = new RegExp(`\\b(${names.join('|')})\\b`, 'gi');
  return safe.replace(rx, (m) => {
    const id = byName.get(m.toLowerCase());
    return id
      ? `<button class="vis-part inline" data-part="${id}">${m}</button>` : m;
  });
}

function partLabel(id) {
  const p = App.structById[id] || App.cortexById[id];
  return p?.short || p?.name || p?.label || id;
}

/** Light the engaged parts and label only them. They stay until Reset. */
function visualiseParts(parts) {
  App.visParts = parts || [];
  App.visFocus = null;
  if (!App.visParts.length) {
    clearSelection();
    App.viewer.clearGlow();
  }
  applyVisualise();
}

/** (Re)apply the Visualise highlight. Parts the moment pushes up come
 *  forward and glow by how hard; parts it pushes down go faint; `focus`
 *  brings one part forward alone. A response preview swaps in its pushes. */
function applyVisualise(focus = null) {
  const parts = App.visParts;
  App.visFocus = focus;
  App.viewer.setPinFilter(parts);
  if (!parts.length) return;

  const moves = (App.visPreview?.moves || App.visMoves)
    .slice(0, App.visReveal);
  const dir = Object.fromEntries(moves.map(m => [m.part, m]));
  const labels = new Map(), structs = [];
  for (const id of parts) {
    if (App.structById[id]) structs.push(id);
    else if (App.cortexById[id]) labels.set(id, App.cortexById[id].label_index);
  }
  const forward = (id) => focus ? id === focus : dir[id]?.dir !== 'less';
  // Focus only isolates a part; its glow still says how hard it is pushed.
  const glow = (id) => {
    if (dir[id]) return dir[id].dir === 'more'
      ? VIS_GLOW_MIN + VIS_GLOW_SPAN * Math.sqrt(dir[id].strength) : 0;
    return visGlow(id, null);
  };

  // Rules can name parts outside the cast (thalamus, NAcc, VTA); a hidden
  // mesh would take the highlight and show nothing.
  App.viewer.setStructureCast([...new Set([...VIS_CAST, ...structs])]);
  App.viewer.selectMany([...labels.values()], structs, null);
  App.viewer.setEmphasis(
    [...labels].filter(([id]) => forward(id)).map(([, li]) => li),
    structs.filter(forward), true, !!focus);
  App.viewer.clearGlow();
  App.viewer.setPinBadges(Object.fromEntries(moves.map(m =>
    [m.part, m.dir === 'more' ? '\u25b2' : '\u25bc'])));

  for (const id of structs) {
    if (forward(id)) App.viewer.setStructureGlow(id, glow(id));
  }
  for (const [id, label] of labels) {
    if (forward(id)) App.viewer.setRegionGlow(label, glow(id));
  }
}

/**
 * Light the regions, structures and pathways the rules derived for this entry.
 *
 * Establishes the whole lit state in one call so it can be replayed after a
 * hover preview ends, without depending on what ran before it.
 */
function highlightAnalysis(a) {
  if (!a) return;
  const pids = (a.pathways || []).map(p => p.id);

  // Strongest engagement wins per node, so a part carrying two pathways
  // glows for the bigger of them rather than the last one read.
  const best = new Map();
  for (const p of a.pathways || []) {
    const amt = Math.min(1, Math.abs(p.amount));
    for (const e of App.model.edges) {
      if (e.pathway !== p.id) continue;
      for (const nid of [e.src, e.dst]) {
        if (!(best.get(nid)?.amt >= amt)) best.set(nid, { amt });
      }
    }
  }

  const labels = [];
  for (const rid of a.regions || []) {
    const r = App.cortexById[rid];
    if (r) labels.push(r.label_index);
  }
  const structs = (a.structures || []).filter(s => App.structById?.[s]);
  if (!labels.length && !structs.length && !pids.length) return;

  App.viewer.selectMany(labels, structs, null);
  App.viewer.clearGlow();
  for (const [nid, { amt }] of best) {
    const n = App.nodeById[nid];
    if (!n) continue;
    const g = amt * 0.9;
    if (n.structure) App.viewer.setStructureGlow(n.structure, g);
    if (n.cortex) {
      const r = App.cortexById[n.cortex];
      if (r) App.viewer.setRegionGlow(r.label_index, g);
    }
  }
  App.circuits.focus(pids.length ? pids : null, FOCUS_ON, FOCUS_OFF);
}

async function doLog(eventId, intensity = 1.0, note = '') {
  const ev = App.eventById[eventId];
  endPreview();
  // The fork is drawn on live weights; a held view would hide it.
  if (App.pathView != null) leavePathView();
  const before = { ...(App.state?.weights || {}) };
  const habitBefore = App.state?.habits?.find(h => h.event === eventId);
  const res = await API.log(eventId, { intensity, note });
  applyState(res.state);
  const habitAfter = res.state.habits?.find(h => h.event === eventId);

  animateResponse(ev);
  const scale = playFork(ev, before, res.state.weights);
  // Rendered in place: the cascade above is playing on the brain right now,
  // and a dialog would be sitting on top of it.
  LogDock.set(resultHtml(res.applied, res.deltas, res.state));
  fillAhead(ev, res.deltas, res.state);
  $('#againBtn').onclick = openTrigger;
  $('#nextDayBtn').onclick = async () => { await advance(1); };

  const fmt = (d) => `${d.label} ${d.delta > 0 ? '+' : '\u2212'}`
    + `${Math.abs(d.percent).toFixed(2)}%`;
  const ups = res.deltas.filter(d => d.delta > 0).slice(0, 2);
  const downs = res.deltas.filter(d => d.delta < 0).slice(0, 1);
  const moved = [...ups, ...downs];
  const habit = habitAfter ? ` \u00b7 habit ${perWeek(habitBefore?.now ?? 0)}`
    + ` \u2192 ${perWeek(habitAfter.now)} a week`
    + (habitAfter.target != null ? ` (aim ${perWeek(habitAfter.target)})` : '')
    : '';
  if (moved.length) {
    toast(`${ev.icon} ${ev.label}`,
      `${moved.map(fmt).join(' \u00b7 ')} (simulated)`
      + (scale > 1 ? ` \u00b7 thickness change shown \u00d7${scale} so it `
        + 'can be seen' : '') + habit);
  }
}

/* Connections are drawn only when they are the story: the moment an
 * experience is logged, the night after, or a part being inspected. */
const PATH_ENLARGE_TO = 0.12;

function endPaths() {
  clearTimeout(App.pathTimer);
  App.pathTimer = null;
  if (App.pathView != null) leavePathView();
  App.circuits.spotlight(null);
}

/** Hand the tubes back to whatever the view shows once a replay ends. */
function pathsDone() {
  App.pathTimer = null;
  App.circuits.spotlight(null);
  showPartPaths();
}

/**
 * The fork: every connection this experience moved, at its old strength,
 * then easing to the new one - the route taken thickens, the ones it
 * competes with thin. Real per-log changes are around 1%, invisible as
 * thickness, so the change is enlarged by a stated factor, and the tubes
 * leave afterwards so the enlarged state is never left on screen.
 * Returns the factor used (1 = true scale, 0 = nothing shown).
 */
function playFork(ev, before, after) {
  if (App.mode !== 'simulation' || App.circuits.deltaMode) return 0;
  const acts = ev?.activations || {};
  const moved = App.model.edges
    .map(e => ({ id: e.id, d: (after[e.id] ?? 0) - (before[e.id] ?? 0),
                 own: !!acts[e.pathway] }))
    .filter(x => x.own || Math.abs(x.d) > 5e-4)
    .sort((a, b) => (b.own - a.own) || Math.abs(b.d) - Math.abs(a.d))
    .slice(0, 6);
  if (!moved.length) return 0;
  const big = Math.max(...moved.map(x => Math.abs(x.d)), 1e-6);
  const scale = Math.max(1, Math.min(40, Math.round(PATH_ENLARGE_TO / big)));
  const shown = {};
  for (const x of moved) shown[x.id] = (before[x.id] ?? 0) + x.d * scale;

  clearTimeout(App.pathTimer);
  App.circuits.spotlight(moved.map(x => x.id));
  App.circuits.animate(before, before, 0.01);
  App.pathTimer = setTimeout(() => {
    App.circuits.animate(before, shown, 1.8);
    App.pathTimer = setTimeout(pathsDone, 4200);
  }, 700);
  return scale;
}

/** The night after: what of yesterday's change set, and what faded. */
function playNight(before, s) {
  const o = s.overnight;
  if (App.mode !== 'simulation' || !o?.pathways?.length) return;
  const top = o.pathways.slice(0, 3).map(p => p.pathway);
  const ids = App.model.edges.filter(e => top.includes(e.pathway))
    .map(e => e.id);
  clearTimeout(App.pathTimer);
  App.circuits.spotlight(ids);
  App.circuits.animate(before, s.weights, 2.2);
  App.circuits.setCores(s.consolidation, s.weights);
  App.pathTimer = setTimeout(pathsDone, 5000);
}

/** Worn paths: a part in focus shows its own connections - glow for how
 *  strong each is, solid core for the share that has consolidated. */
function showPartPaths() {
  if (App.pathTimer) return;
  if (App.pathView != null && App.mode === 'simulation'
      && App.viewState === 'current') {
    drawPathView();
    return;
  }
  const s = App.viewState;
  const part = App.mode === 'simulation' && READING.has(s)
    ? App.curFocus : null;
  const nodes = part ? App.model.nodes
    .filter(n => n.structure === part || n.cortex === part)
    .map(n => n.id) : [];
  const ids = App.model.edges
    .filter(e => nodes.includes(e.src) || nodes.includes(e.dst))
    .map(e => e.id);
  if (!ids.length) { App.circuits.spotlight(null); return; }
  const tgt = s === 'target';
  const w = (tgt ? App.state?.targets : App.state?.weights) || {};
  App.circuits.spotlight(ids);
  App.circuits.update(w);
  App.circuits.setCores(tgt ? null : App.state?.consolidation, w);
}

/** Light the structures and fire pulses down the pathways the event used. */
function animateResponse(ev) {
  App.viewer.clearGlow();
  const { edges, lit } = eventTargets(ev);
  for (const [n, a] of lit) {
    if (n.structure) App.viewer.setStructureGlow(n.structure, a);
    if (n.cortex) {
      const r = App.cortexById[n.cortex];
      if (r) App.viewer.setRegionGlow(r.label_index, a);
    }
  }
  App.circuits.cascade(edges, { stagger: 0.13, speed: 1.35 });
  setTimeout(() => {
    App.viewer.clearGlow();
    if (App.mode === 'simulation' && App.viewState === 'current') {
      for (const [p, g] of Object.entries(App.engGlow || {}))
        setPartGlow(App.viewer, p, g);
    }
  }, 3200);
}

/* ======================================================================= */
/*  TIME                                                                    */
/* ======================================================================= */

async function advance(days) {
  App.replayDay = null;
  if (App.pathView != null) leavePathView();
  const before = { ...(App.state?.weights || {}) };
  const s = await API.advance(days);
  applyState(s);
  $('#tlLive').classList.add('active');
  if (days === 1) playNight(before, s);
  if (days > 1) toast(`${days} days passed`,
    'Unrehearsed pathways decayed; consolidated ones held.');
}

async function scrubTo(day) {
  App.playing = false;
  $('#tlPlay').textContent = '▶ Replay';
  App.replayDay = day;
  $('#tlLive').classList.remove('active');
  const s = await API.replay(day);
  App.state = { ...App.state, weights: s.weights, engagement: s.engagement };
  $('#dayNum').textContent = day;
  $('#tlDay').textContent = `day ${day}`;
  App.timeline.set(App.state, day);
  setViewState(App.viewState);
}

function goLive() {
  App.playing = false;
  App.replayDay = null;
  $('#tlPlay').textContent = '▶ Replay';
  $('#tlLive').classList.add('active');
  API.state().then(applyState);
}

function togglePlay() {
  if (!App.state || App.state.day < 2) {
    toast('Nothing to replay yet',
      'Log some practice and let days pass, or run the demo from ⋯');
    return;
  }
  App.playing = !App.playing;
  $('#tlPlay').textContent = App.playing ? '❚❚ Pause' : '▶ Replay';
  if (App.playing) {
    if (App.mode !== 'simulation') setMode('simulation');
    App.replayDay = App.replayDay ?? 0;
    App.playT = App.replayDay;
    $('#tlLive').classList.remove('active');
  }
}

/* ======================================================================= */
/*  SESSION MENU                                                            */
/* ======================================================================= */

function openMenu() {
  Modal.open(`
    <div class="m-title">Session</div>
    <div class="m-sub">The simulation lives in the Python process.</div>
    <div class="choice-grid" style="grid-template-columns:1fr 1fr">
      <button class="choice" id="mDemo">
        <span class="cico">🎲</span><span>
          <span class="clab">Run a 12-week demo</span>
          <span class="crule">Scripted practice history so you can see the
            evolution immediately</span></span></button>
      <button class="choice" id="mReset">
        <span class="cico">↺</span><span>
          <span class="clab">Reset to day 0</span>
          <span class="crule">Start a fresh simulation</span></span></button>
      <button class="choice" id="mSave">
        <span class="cico">💾</span><span>
          <span class="clab">Save session</span>
          <span class="crule">To ~/.neuroforge/session.json</span></span></button>
      <button class="choice" id="mLoad">
        <span class="cico">📂</span><span>
          <span class="clab">Load session</span>
          <span class="crule">From ~/.neuroforge/session.json</span></span></button>
    </div>
    <div class="p-note" style="margin-top:20px">
      <b>Keyboard:</b> 1/2/3 modes · R reset camera · L log · Space replay ·
      Esc close
    </div>`);

  $('#mDemo').onclick = async () => {
    Modal.close();
    toast('Simulating 12 weeks of practice…');
    applyState(await API.demo(12, 0.72));
    setMode('simulation');
    toast('12 weeks simulated', 'Press ▶ Replay to watch it evolve.');
  };
  $('#mReset').onclick = async () => {
    Modal.close(); App.replayDay = null;
    applyState(await API.reset());
    toast('Reset to day 0');
  };
  $('#mSave').onclick = async () => {
    const r = await API.save('session');
    Modal.close(); toast('Saved', r.path);
  };
  $('#mLoad').onclick = async () => {
    const r = await API.load('session');
    Modal.close();
    if (r.error) return toast('No saved session found');
    applyState(r); toast('Session loaded');
  };
}

/* ======================================================================= */
/*  LOOP                                                                    */
/* ======================================================================= */

function startLoop() {
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = now / 1000;

    if (App.playing && App.state) {
      App.playT += dt * 6;                       // ~6 simulated days a second
      const d = Math.floor(App.playT);
      if (d >= App.state.day) {
        App.playing = false;
        $('#tlPlay').textContent = '▶ Replay';
        goLive();
      } else if (d !== App.replayDay) {
        scrubToLocal(d);
      }
    }

    App.viewer.render(dt, t);
    if (App.viewer2 && document.body.classList.contains('split')) {
      syncCameras();
      App.viewer2.render(dt, t);
    }
    App.circuits.render(dt, t);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/** Replay without a round-trip: interpolate from the history we already have. */
function scrubToLocal(day) {
  App.replayDay = day;
  $('#dayNum').textContent = day;
  $('#tlDay').textContent = `day ${day}`;
  App.timeline.set(App.state, day);
  API.replay(day).then(s => {
    if (!App.playing && App.replayDay !== day) return;
    App.circuits.update(s.weights,
      App.viewState === 'compare' ? App.state.targets : null);
  });
}

boot();
