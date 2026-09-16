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
  Modal, Panel, Timeline, choicesHtml, edgePanel, groupPanel, honestyHtml,
  networkPanel, proposalsHtml, regionPanel, resultHtml, toast,
  triggerModalHtml,
} from './ui.js';

const $  = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const App = {
  scene: null, model: null, state: null,
  mode: 'anatomy', viewState: 'current', replayDay: null,
  playing: false, playT: 0,
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
  App.circuits.build(App.model);
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

  /* ---- regions that no external view can show ---- */
  // The insula is folded inside the lateral sulcus. Selecting it on an opaque
  // brain highlights nothing visible, which reads as a broken feature, so the
  // viewer switches to x-ray and says why.
  for (const r of App.scene.cortical_regions) {
    if (r.buried) V.buriedLabels.add(r.label_index);
  }
  V._onReveal = (msg) => toast('Hidden region', msg);

  /* ---- picking ---- */
  V.onPick((hit) => {
    if (!hit) { V.select(null); Panel.close(); return; }
    if (hit.kind === 'edge') { showEdge(hit.id); return; }
    V.select(hit);
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
  $('#chkXray').onchange    = (e) => { V.forceSolid = !e.target.checked;
                                       V.setXray(e.target.checked, true); };
  $('#chkSubcort').onchange = (e) => V.setSubcortical(e.target.checked);
  $('#chkLabels').onchange  = (e) => V.setPins(e.target.checked);
  $('#chkSpin').onchange    = (e) => V.setAutoRotate(e.target.checked);

  /* ---- structure list ---- */
  buildStructureList();
  buildGroupList();
  buildNetworkList();

  /* ---- state toggle ---- */
  $$('#stateSeg button').forEach(b => b.onclick = () => {
    $$('#stateSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    setViewState(b.dataset.state);
  });

  /* ---- actions ---- */
  $('#btnTrigger').onclick  = openTrigger;
  $('#btnHonesty').onclick  = () =>
    Modal.open(honestyHtml(App.model, App.scene.notes));
  $('#btnTimeline').onclick = () =>
    $('#timeline').classList.toggle('hidden');
  $('#btnMenu').onclick     = openMenu;

  $('#panelClose').onclick  = () => { Panel.close(); App.viewer.select(null); };
  $('#btnChat').onclick     = () => App.chat?.toggle();
  $('#chatClose').onclick   = () => App.chat?.close();
  $('#modalClose').onclick  = Modal.close;
  $('#modalWrap').onclick   = (e) => {
    if (e.target.id === 'modalWrap') Modal.close();
  };
  $('#stripHide').onclick   = () => $('#strip').classList.add('hidden');

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
      else if (Panel.isOpen()) { Panel.close(); V.select(null); }
    }
    if (e.target.tagName === 'INPUT') return;
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

/* Composite anatomical terms. "Prefrontal cortex" and "dorsal striatum" are
 * collections of regions, not parcels, so they are offered as selections over
 * things that already exist rather than drawn as invented regions. */
function buildGroupList() {
  const groups = App.scene.groups || [];
  const host = $('#groupList');
  if (!groups.length) { host.classList.add('hidden'); return; }
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
      host.querySelectorAll('.structrow').forEach(x => x.classList.remove('on'));
      if (already) { App.viewer.selectMany([], []); Panel.close(); return; }
      row.classList.add('on');
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

function buildStructureList() {
  const host = $('#structList');
  const items = App.scene.structures
    .slice().sort((a, b) => a.order - b.order);
  host.innerHTML = '<div class="docktitle">STRUCTURES</div>' +
    items.map(s => `
      <div class="structrow" data-id="${s.id}">
        <span class="sw" style="background:${s.color};color:${s.color}"></span>
        <span class="nm">${s.short}</span>
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
      App.viewer.select({ kind: 'structure', id });
      showStructure(id);
      focusStructure(id);
    };
  });
}

function focusStructure(id) {
  const s = App.structById[id];
  if (!s) return;
  const c = s.centroids[0];
  // RAS (x,y,z) -> world (x, z, -y), matching viewer.root's rotation
  App.viewer.controls.target.set(c[0], c[2], -c[1]);
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

  App.circuits.setVisible(circ);
  $('#statebar').classList.toggle('hidden', !sim);
  $('#btnTimeline').classList.toggle('hidden', !sim);
  $('#timeline').classList.toggle('hidden', !sim);
  $('#metricWrap').style.opacity = sim ? '1' : '.32';

  if (mode === 'anatomy') {
    $('#chkXray').checked = false;
    App.viewer.forceSolid = true;
    Panel.close();
    App.viewer.select(null);
    App.circuits.clearPulses();
  } else {
    $('#chkXray').checked = true;
    App.viewer.forceSolid = false;
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
  App.viewState = s;
  const hint = $('#stateHint');
  if (s === 'current') {
    App.circuits.update(App.state.weights);
    App.circuits.setGhost(false);
    hint.textContent = 'simulated strengths built by your logged practice';
  } else if (s === 'target') {
    App.circuits.update(App.state.targets);
    App.circuits.setGhost(false);
    hint.textContent = 'the behavioural direction you have chosen to practise';
  } else {
    App.circuits.update(App.state.weights, App.state.targets);
    App.circuits.setGhost(true);
    hint.textContent = 'solid = current · dashed = target';
  }
}

/* ======================================================================= */
/*  STATE APPLICATION                                                       */
/* ======================================================================= */

function applyState(s = App.state) {
  App.state = s;
  $('#dayNum').textContent = App.replayDay ?? s.day;
  $('#practiceNum').textContent = s.total_practices;
  $('#streakNum').textContent = s.streak;
  const pct = (s.alignment * 100);
  $('#alignBar').style.width = pct + '%';
  $('#alignVal').textContent = pct.toFixed(1) + '%';
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
  for (const id of ids) {
    if (App.cortexById[id]) labels.push(App.cortexById[id].label_index);
    else if (App.structById[id]) structs.push(id);
  }
  if (!labels.length && !structs.length) return;

  // selectMany() already handles buried cortex: it flips to x-ray when a
  // hidden region is selected and restores the previous mode afterwards.
  V.selectMany(labels, structs);
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
  Modal.open(triggerModalHtml());
  $('#choiceHost').innerHTML =
    choicesHtml(App.model.events, App.model.categories);
  $$('#choiceHost .choice').forEach(b =>
    b.onclick = () => doLog(b.dataset.event));

  wireInterpreter();

  // the trigger itself: light salience, then fire the cascade
  App.viewer.setStructureGlow('amygdala', 1.0);
  App.circuits.cascade(
    ['BLA->CeA', 'BLA->dACC'], { stagger: 0.18, speed: 1.8 });
  setTimeout(() => App.viewer.setStructureGlow('amygdala', 0.35), 1600);
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
    go.textContent = 'Readingâ€¦';
    out.classList.remove('hidden');
    out.innerHTML = '<div class="nl-empty">Interpretingâ€¦</div>';
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
    } catch (e) {
      out.innerHTML =
        `<div class="nl-empty">Could not interpret that (${e.message}).
         Pick an event below instead.</div>`;
    } finally {
      go.disabled = false;
      go.textContent = 'Interpret';
    }
  };

  $('#nlText').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) go.click();
  });
}

async function doLog(eventId, intensity = 1.0, note = '') {
  const ev = App.eventById[eventId];
  const res = await API.log(eventId, { intensity, note });
  applyState(res.state);

  animateResponse(ev);
  Modal.open(resultHtml(res.applied, res.deltas, res.state));
  $('#againBtn').onclick = openTrigger;
  $('#nextDayBtn').onclick = async () => { Modal.close(); await advance(1); };

  const top = res.deltas[0];
  if (top) {
    toast(`${ev.icon} ${ev.label}`,
      `${top.label} ${top.delta > 0 ? '+' : ''}${top.percent.toFixed(2)}% `
      + `(simulated)`);
  }
}

/** Light the structures and fire pulses down the pathways the event used. */
function animateResponse(ev) {
  App.viewer.clearGlow();
  const acts = ev.activations || {};
  const edges = [];
  for (const e of App.model.edges) {
    const a = acts[e.pathway];
    if (!a || a <= 0) continue;
    edges.push(e.id);
    for (const nid of [e.src, e.dst]) {
      const n = App.nodeById[nid];
      if (!n) continue;
      if (n.structure) App.viewer.setStructureGlow(n.structure, Math.abs(a));
      if (n.cortex) {
        const r = App.cortexById[n.cortex];
        if (r) App.viewer.setRegionGlow(r.label_index, Math.abs(a));
      }
    }
  }
  App.circuits.cascade(edges, { stagger: 0.13, speed: 1.35 });
  setTimeout(() => App.viewer.clearGlow(), 3200);
}

/* ======================================================================= */
/*  TIME                                                                    */
/* ======================================================================= */

async function advance(days) {
  App.replayDay = null;
  applyState(await API.advance(days));
  $('#tlLive').classList.add('active');
  if (days > 1) toast(`${days} days passed`,
    'Unrehearsed pathways decayed; consolidated ones held.');
}

async function scrubTo(day) {
  App.playing = false;
  $('#tlPlay').textContent = '▶ Replay';
  App.replayDay = day;
  $('#tlLive').classList.remove('active');
  const s = await API.replay(day);
  App.state = { ...App.state, weights: s.weights };
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
