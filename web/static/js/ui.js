/* ============================================================================
 * ui.js — DOM panels, modals, timeline and toasts.
 *
 * Every knowledge panel is deliberately split into two visually distinct
 * kinds of claim:
 *    REAL   (green)  — what the neuroscience literature actually supports
 *    MODEL  (amber)  — what this program is doing with it
 * That separation is a design requirement, not decoration.
 * ========================================================================= */

const $ = (s) => document.querySelector(s);

/* ------------------------------------------------------------------ toast */
export function toast(msg, sub = '', ms = 3400) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<div>${msg}</div>${sub ? `<div class="sub">${sub}</div>` : ''}`;
  $('#toasts').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .4s, transform .4s';
    el.style.opacity = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => el.remove(), 420);
  }, ms);
}

/* ------------------------------------------------------------------ panel */
export const Panel = {
  open(html) {
    LogDock.close();
    $('#panelBody').innerHTML = html;
    $('#panel').classList.remove('hidden');
    document.body.classList.add('panel-open');
  },
  close() {
    $('#panel').classList.add('hidden');
    document.body.classList.remove('panel-open');
  },
  isOpen() { return !$('#panel').classList.contains('hidden'); },
};

/* --------------------------------------------------------------- logdock */
/* Logging is docked, not modal. A dialog over the brain meant the one thing
 * worth watching - the model responding - happened behind the dialog. */
export const LogDock = {
  open(html) {
    Panel.close();
    this.set(html);
    $('#logdock').classList.remove('hidden');
    document.body.classList.add('log-open');
  },
  set(html) {
    const body = $('#logBody');
    body.innerHTML = html;
    body.parentElement.scrollTop = 0;
  },
  close() {
    $('#logdock').classList.add('hidden');
    document.body.classList.remove('log-open');
  },
  isOpen() { return !$('#logdock').classList.contains('hidden'); },
};

function section(title, tag, body, cls = '') {
  if (!body) return '';
  const tagHtml = tag
    ? `<span class="tag ${tag}">${tag === 'real' ? 'REAL' : 'MODEL'}</span>`
    : '';
  return `<div class="p-sec">
      <div class="p-h" style="color:var(--txt-faint)">${title}${tagHtml}</div>
      <div class="p-body ${cls}">${body}</div>
    </div>`;
}

/** Knowledge panel for a subcortical structure or a cortical parcel. */
export function regionPanel(meta, kind, connections = [], networkNote = '',
                            fineName = null, network = null) {
  const k = meta.knowledge;
  const conns = connections.length ? `
    <div class="p-divider"></div>
    <div class="p-h" style="color:var(--txt-faint)">
      SIMULATED CONNECTIONS<span class="tag model">MODEL</span>
    </div>
    ${connections.map(c => `
      <div class="conn">
        <span class="cname">${c.label}</span>
        <span class="cbar"><i style="width:${(c.w * 100).toFixed(0)}%;
              background:${c.color}"></i></span>
        <span class="cnum">${c.w.toFixed(2)}</span>
      </div>`).join('')}
    <div style="font-size:10.5px;color:var(--txt-faint);margin-top:9px;
         line-height:1.5">
      These values are variables in this program, on a 0–1 scale. They are
      not synapse counts, not gray-matter volume and not measured activity.
    </div>` : '';

  return `
    <div class="p-kicker">${kind === 'cortex' ? 'CORTICAL REGION'
                                              : 'BRAIN STRUCTURE'}</div>
    <div class="p-title">${meta.name}</div>
    <div class="p-sub">${meta.system || meta.short || ''}</div>
    ${fineName ? `<div class="p-sec">
      <div class="p-h">YOU CLICKED<span class="tag real">REAL</span></div>
      <div class="p-body"><b>${fineName}</b><br>
      <span style="font-size:10.5px;color:var(--txt-faint)">
      Named from the Destrieux atlas. The coloured region above is a coarser
      functional grouping this app defines; this is the specific anatomical
      fold your cursor landed on.</span></div>
    </div>` : ''}
    ${network ? `<div class="p-sec">
      <div class="p-h">FUNCTIONAL NETWORK<span class="tag real">REAL</span></div>
      <div class="p-body real">
        <span class="sw" style="background:${network.color};
              color:${network.color}"></span> <b>${network.name}</b><br>
        <span style="font-size:10.5px;color:var(--txt-faint)">
        This vertex was assigned to that network by resting-state fMRI in 1000
        adults (Yeo 2011), not by its anatomy. Turn on Functional networks in
        the sidebar to see the whole of it.</span></div>
    </div>` : ''}
    ${section('WHAT IT IS', 'real', k.what_it_is, 'real')}
    ${section('WHERE IT IS', 'real', k.where_it_is, 'real')}
    ${section('WHAT IT CONTRIBUTES TO', 'real', k.contributes_to, 'real')}
    ${section('IN THIS SIMULATION', 'model', k.in_the_model, 'model')}
    ${k.caveat ? `<div class="p-note"><b>Careful:</b> ${k.caveat}</div>` : ''}
    ${conns}
    ${networkNote ? `<div class="p-divider"></div>
       <div class="p-note" style="border-color:rgba(56,224,255,.4);
            background:rgba(56,224,255,.05)">${networkNote}</div>` : ''}
  `;
}

/** Knowledge panel for a composite term spanning several regions. */
/**
 * A small inline SVG line chart of one pathway's simulated strength.
 *
 * The y-axis is fixed to 0..1 rather than fitted to the data. An
 * auto-fitted axis makes a 0.02 wobble look like a dramatic climb, which
 * would be a visual overstatement of what the model actually did.
 */
export function sparkline(series, color = '#38e0ff', w = 268, h = 54) {
  if (!series || series.length < 2) {
    return `<div class="spark-empty">No history yet &mdash; log a few days
            and this becomes a curve.</div>`;
  }
  const n = series.length;
  const x = (i) => (i / (n - 1)) * (w - 2) + 1;
  const y = (v) => h - 3 - Math.max(0, Math.min(1, v)) * (h - 6);
  const pts = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const area = `M ${x(0).toFixed(1)},${(h - 3).toFixed(1)} L ${pts.join(' L ')}
                L ${x(n - 1).toFixed(1)},${(h - 3).toFixed(1)} Z`;
  const first = series[0], last = series[n - 1];
  const delta = last - first;
  const sign = delta >= 0 ? '+' : '';
  return `
    <svg class="spark" viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
      <line x1="1" y1="${y(0.5)}" x2="${w - 1}" y2="${y(0.5)}"
            stroke="var(--line)" stroke-width="1" stroke-dasharray="3 3"/>
      <path d="${area}" fill="${color}" opacity="0.13"/>
      <polyline points="${pts.join(' ')}" fill="none" stroke="${color}"
                stroke-width="1.8" stroke-linejoin="round"
                stroke-linecap="round"/>
      <circle cx="${x(n - 1)}" cy="${y(last)}" r="3" fill="${color}"/>
    </svg>
    <div class="spark-foot">
      <span>day ${0}</span>
      <span class="spark-delta ${delta >= 0 ? 'up' : 'down'}">
        ${sign}${(delta * 100).toFixed(1)}%
      </span>
      <span>now</span>
    </div>`;
}

/**
 * Panel for a single simulated connection between two anatomical anchors.
 *
 * This is the one panel that is almost entirely MODEL. A line between two
 * regions in this app is a variable in a program, not a tract anyone has
 * imaged in the user, and the panel has to keep saying so.
 */
export function edgePanel(edge, pathway, srcName, dstName, series, weight) {
  const col = pathway?.color || '#38e0ff';
  return `
    <div class="p-kicker">SIMULATED PATHWAY</div>
    <div class="p-title">${srcName} &rarr; ${dstName}</div>
    <div class="p-sub">
      <span class="sw" style="background:${col};color:${col}"></span>
      ${pathway?.label || 'pathway'} &middot; ${edge.label || ''}
    </div>

    <div class="p-sec">
      <div class="p-h">STRENGTH OVER TIME<span class="tag model">MODEL</span></div>
      <div class="p-body model">
        <div class="bigval" style="color:${col}">
          ${(weight * 100).toFixed(0)}<i>%</i>
        </div>
        ${sparkline(series, col)}
        Simulated pathway strength, 0&ndash;100%. This is a number this
        program keeps and updates from what you log. It is not a measurement
        of your neurons, synapses, gray matter or brain activity.
      </div>
    </div>

    ${section('WHAT THE EVIDENCE SAYS', 'real', edge.science, 'real')}
    ${pathway?.science && pathway.science !== edge.science
      ? section('ABOUT THIS ROUTE', 'real', pathway.science, 'real') : ''}
  `;
}

/** Knowledge panel for a group of regions shown as one system. */
export function groupPanel(g, parts = []) {
  return `
    <div class="p-kicker">SYSTEM</div>
    <div class="p-title">${g.name}</div>
    <div class="p-sub">${parts.length} regions shown together</div>
    <div class="p-sec">
      <div class="p-h">NOT ONE REGION<span class="tag real">REAL</span></div>
      <div class="p-body real">
        This is a collection, not a parcel. There is no boundary in the
        tissue where it begins and ends - only a convention about where to
        draw the line. What is highlighted is every part it contains:
        <div style="margin-top:8px">
        ${parts.map(p => `<div class="conn">
            <span class="sw" style="background:${p.color};
                  color:${p.color}"></span>
            <span class="cname">${p.name}</span>
          </div>`).join('')}
        </div>
      </div>
    </div>
    ${section('WHAT IT IS', 'real', g.what, 'real')}
    ${section('WHY THE TERM IS USED', 'real', g.why, 'real')}
    ${g.caveat ? `<div class="p-note"><b>Careful:</b> ${g.caveat}</div>` : ''}
  `;
}

/** Knowledge panel for one of the seven Yeo resting-state networks. */
export function networkPanel(net, caveat = '') {
  return `
    <div class="p-kicker">FUNCTIONAL NETWORK</div>
    <div class="p-title">${net.name}</div>
    <div class="p-sub">Yeo et al. 2011 &middot; seven-network parcellation</div>
    <div class="p-sec">
      <div class="p-h">HOW THIS WAS DEFINED<span class="tag real">REAL</span></div>
      <div class="p-body real">
        Not by anatomy. These boundaries come from resting-state fMRI in 1000
        adults: vertices whose spontaneous activity rose and fell together were
        grouped. That is why a network can be scattered across lobes.
      </div>
    </div>
    ${section('WHAT IT IS', 'real', net.what, 'real')}
    ${section('AND THE SENSE OF SELF', 'real', net.self, 'real')}
    ${net.caveat ? `<div class="p-note"><b>Careful:</b> ${net.caveat}</div>` : ''}
    ${caveat ? `<div class="p-divider"></div>
      <div class="p-note" style="border-color:rgba(56,224,255,.4);
           background:rgba(56,224,255,.05)">${caveat}</div>` : ''}
  `;
}

/* ------------------------------------------------------------------ modal */
export const Modal = {
  open(html) {
    $('#modalBody').innerHTML = html;
    $('#modalWrap').classList.remove('hidden');
  },
  close() { $('#modalWrap').classList.add('hidden'); },
  isOpen() { return !$('#modalWrap').classList.contains('hidden'); },
};

/** Step 1 of logging: the trigger. Rendered into the docked surface. */
export function triggerModalHtml() {
  return `
    <div class="p-kicker">SOMETHING HAPPENED</div>
    <div class="m-title">What did you actually do?</div>
    <div class="m-sub">
      A trigger raises emotional salience and opens two competing routes &mdash;
      the old automatic one and the deliberate regulated one.
    </div>
    <div class="preview-hint">Hover a response to see what it engages.</div>

    <div class="nl-box">
      <label class="nl-label" for="nlText">
        Describe it in your own words
        <span class="nl-hint">optional &mdash; you can just pick below</span>
      </label>
      <textarea id="nlText" rows="3" placeholder="e.g. I kept replaying an argument in my head all evening, then went for a walk and felt steadier."></textarea>
      <div class="nl-row">
        <button id="nlGo">Interpret</button>
        <span class="nl-kbd">Ctrl + Enter</span>
        <span id="nlSource" class="nl-source"></span>
      </div>
    </div>

    <div id="nlOut" class="nl-out hidden"></div>

    <details id="pickWrap" class="nl-pick" open>
      <summary><span>or choose directly</span></summary>
      <div id="choiceHost"></div>
    </details>`;
}

/**
 * Render suggested events for confirmation.
 *
 * Every proposal is a button the user must press. Nothing here applies
 * itself, and the confidence is shown rather than hidden, because a
 * classifier that quietly logged a "regulated success" the user did not have
 * would be corrupting the only record they have.
 */
export function proposalsHtml(interp) {
  if (!interp.proposals.length) {
    return `<div class="nl-empty">${interp.note}</div>`;
  }
  return `
    <div class="nl-note">${interp.note}</div>
    ${interp.proposals.map(p => {
      const pct = Math.round(p.confidence * 100);
      const band = p.confidence >= 0.66 ? 'hi'
                 : p.confidence >= 0.4 ? 'mid' : 'lo';
      return `
      <button class="nl-prop" data-event="${p.event}"
              data-intensity="${p.intensity}">
        <span class="nl-prop-main">
          <b>${p.label}</b>
          <i>${p.why}</i>
        </span>
        <span class="nl-conf ${band}" title="how sure the classifier is">
          ${pct}%
        </span>
      </button>`;
    }).join('')}
    <div class="nl-foot">
      These are guesses about <b>which of the 16 known events</b> you
      described. The effect on the model is decided by fixed rules, not by
      the interpreter.
    </div>`;
}

/** One boxed placeholder while the classifier runs. */
export function interpretingHtml(stage = 'Reading your entry') {
  return `
    <div class="an-wrap pending">
      <div class="an-h">INTERPRETING</div>
      <div class="an-load"><i></i><i></i><i></i></div>
      <div class="an-stage">${stage}</div>
    </div>`;
}

/**
 * The brain reading. Pathway rows come from the rules; the prose may come
 * from a model, which is why the two are labelled separately.
 */
export function analysisHtml(a) {
  if (!a || !a.events || !a.events.length) return '';
  const rows = (a.pathways || []).map(p => {
    const up = p.amount > 0;
    const pct = Math.min(100, Math.abs(p.amount) * 100);
    // Colour is "toward or away from your target", not the raw sign: a
    // stress habit gaining +0.70 is not good news.
    return `
      <div class="an-row ${p.beneficial ? 'good' : 'bad'}"
           title="${p.science.replace(/"/g, '&quot;')}">
        <span class="an-name">${p.label}</span>
        <span class="an-bar"><i style="width:${pct}%"></i></span>
        <span class="an-amt">${up ? '+' : '&minus;'}${Math.abs(p.amount).toFixed(2)}</span>
      </div>`;
  }).join('');

  // State events move sleep and stress instead of any one connection, so
  // they get a row of their own rather than an empty panel.
  const mods = (a.modulators || []).map(m => `
      <div class="an-mod">
        <span class="an-name">${m.label}</span>
        <span class="an-amt">${m.absolute ? '' : (m.amount > 0 ? '+' : '&minus;')}${Math.abs(m.amount).toFixed(2)}</span>
      </div>`).join('');
  const modWrap = mods
    ? `<div class="an-mods"><div class="an-sub">CONDITIONS FOR LEARNING</div>${mods}</div>`
    : '';

  const src = a.source === 'model'
    ? `<span class="tag model">MODEL</span> written by ${a.model}`
    : `<span class="tag real">RULES</span> composed from the rule table`;

  return `
    <div class="an-wrap">
      <div class="an-h">WHAT THIS ENGAGES</div>
      <div class="an-text">${a.text}</div>
      ${rows ? `<div class="an-rows">${rows}</div>` : ''}
      ${rows ? `<div class="an-legend">
        <span class="up"><i></i>toward your target</span>
        <span class="dn"><i></i>away from it</span>
      </div>` : ''}      ${modWrap}
      <div class="an-src">${src}${a.note ? ' &middot; ' + a.note : ''}</div>
    </div>`;
}

export function choicesHtml(events, categories) {
  const CAT_COPY = {
    Practice: ['REGULATED ROUTE', 'effortful now, more available later'],
    Slip:     ['OLD ROUTE', 'low effort now, more available later too'],
    State:    ['CONTEXT', 'these change how well anything else sticks'],
  };
  return categories.map(cat => {
    const items = events.filter(e => e.category === cat);
    if (!items.length) return '';
    const [title, sub] = CAT_COPY[cat] || [cat, ''];
    return `
      <div class="cat-h">${title}
        <span style="letter-spacing:0;text-transform:none;margin-left:8px;
              opacity:.75">${sub}</span></div>
      <div class="choice-grid">
        ${items.map(e => `
          <button class="choice ${cat.toLowerCase()}" data-event="${e.id}">
            <span class="cico">${e.icon}</span>
            <span>
              <span class="clab">${e.label}</span>
              <span class="crule">${e.rule}</span>
            </span>
          </button>`).join('')}
      </div>`;
  }).join('');
}

/** Step 2: what the model did about it. */
export function resultHtml(applied, deltas, state) {
  const rows = deltas.map(d => {
    const up = d.delta > 0;
    const w = Math.min(100, Math.abs(d.percent) * 14 + 6);
    return `
      <div class="delta-row">
        <span class="dname">${d.label}</span>
        <span class="dtrack">
          <i style="left:${up ? 50 : 50 - w / 2}%;width:${w / 2}%;
             background:${d.color}"></i>
        </span>
        <span class="dval ${up ? 'up' : 'down'}">
          ${up ? '+' : ''}${d.percent.toFixed(2)}%
        </span>
      </div>`;
  }).join('');

  return `
    <div class="p-kicker">PRACTICE RECORDED</div>
    <div class="m-title">${applied.icon} ${applied.label}</div>
    <div class="m-sub">Day ${state.day} · simulated pathway strengths
      updated by the rule below.</div>

    <div class="p-sec">
      <div class="p-h" style="color:var(--txt-faint)">
        CHANGE IN SIMULATED PATHWAY STRENGTH<span class="tag model">MODEL</span>
      </div>
      ${rows || '<div class="p-body">No pathway change — this event only '
                + 'altered sleep or stress, which affects how well '
                + 'tomorrow&rsquo;s practice sticks.</div>'}
    </div>

    <div class="p-note" style="margin-bottom:18px">
      These are percentage-point changes in a <b>model variable</b> on a 0–1
      scale. Nothing was measured. No neuron was pruned, no gray matter was
      gained. One repetition should move the model only slightly — that is
      the point.
    </div>

    ${section('THE RULE APPLIED', 'model', applied.rule, 'model')}
    ${section('WHAT THE SCIENCE ACTUALLY SAYS', 'real', applied.science,
              'real')}
    ${applied.caveat
      ? `<div class="p-note"><b>Careful:</b> ${applied.caveat}</div>` : ''}

    <div style="display:flex;gap:9px;margin-top:22px">
      <button class="primary" id="againBtn" style="flex:1;justify-content:center">
        Log another
      </button>
      <button class="secondary" id="nextDayBtn"
              style="flex:1;justify-content:center">
        End the day →
      </button>
    </div>`;
}

/* ---------------------------------------------------------------- honesty */
export function honestyHtml(model, notes) {
  return `
    <div class="m-title">What this is, and what it is not</div>
    <div class="m-sub">Scientific integrity statement</div>

    ${section('WHAT NEUROPLASTICITY ACTUALLY MEANS', 'real', `
      Learning changes the nervous system, but the dominant mechanism is not
      "grow new neurons". It is change in the <b>strength, reliability and
      timing of existing connections</b> — long-term potentiation and
      depression, receptor trafficking, changes in intrinsic excitability —
      together with shifts in which network dominates a behaviour.
      <br><br>
      Structural change is real too: dendritic spine turnover, axonal
      remodelling, and changes in myelination. But these typically require
      weeks to months of substantial practice, group-level imaging effects
      are small, and popular reporting of them is routinely inflated.
      <br><br>
      Different processes operate on different timescales, and this app
      distinguishes them: fast <b>synaptic strengthening</b>, gradual
      <b>weakening of unreinforced routes</b>, <b>network-level
      reconfiguration</b>, sleep-dependent <b>consolidation</b>, slow
      <b>habit formation</b>, and slower <b>structural</b> change.`, 'real')}

    ${section('WHAT "PRUNING" ACTUALLY MEANS', 'real', `
      Synaptic pruning is a real, activity-dependent, partly
      microglia-mediated process. It is most dramatic during development and
      adolescence.
      <br><br>
      <b>Deciding not to ruminate today does not prune a synapse.</b>
      Conscious thought does not delete individual neurons. In this
      simulation, "weakening" means only this: a route stops being
      reinforced and loses a competition for a limited input budget at its
      target. Nothing is deleted — which is exactly why old patterns can
      return under stress, poor sleep, or after a long gap. The model
      reproduces that on purpose.`, 'real')}

    ${section('WHAT CANNOT BE INFERRED FROM BEHAVIOUR', 'real', `
      You cannot read brain state off behaviour. Logging "I reappraised"
      tells this program that you reported doing something; it tells nobody
      anything about your prefrontal cortex. The same behaviour can be
      produced by different underlying processes in different people, and
      the same brain state can produce different behaviour depending on
      context, sleep, arousal and a hundred other things. Reverse
      inference — "region X lit up, therefore the person felt Y" — is a
      well-known fallacy in neuroimaging.`, 'real')}

    ${section('THE BRAIN IS NOT A SET OF MODULES', 'real',
              notes.network, 'real')}

    ${section('ABOUT THIS GEOMETRY', 'model', notes.geometry, 'model')}

    ${section('THE TARGET STATE IS NOT A "GOOD BRAIN"', 'model', `
      The target is a <b>chosen behavioural direction</b> written as numbers:
      less rumination, faster recovery, a stronger pause. It is not a picture
      of a healthy brain and there is no such picture.
      <br><br>
      Note that the threat pathway's target is deliberately <b>not zero</b>.
      A person with no threat response is not regulated; they are in danger.
      The app will not let you aim at silence.`, 'model')}

    <div class="p-note" style="border-color:rgba(255,92,110,.5);
         background:rgba(255,92,110,.05)">
      <b>Not a clinical tool.</b> If you are dealing with trauma, panic or
      persistent low mood, exposure-style practice is best done with a
      clinician. This program cannot tell whether a practice is helping you.
    </div>

    <div class="p-divider"></div>
    <div class="p-h" style="color:var(--txt-faint)">
      DELIBERATELY LEFT OUT<span class="tag model">MODEL</span></div>
    <div class="p-body">
      ${model.exclusions.map(e =>
        `<div style="margin-bottom:7px"><b>${e.name}</b> — ${e.why}</div>`
      ).join('')}
    </div>`;
}

/* --------------------------------------------------------------- timeline */
export class Timeline {
  constructor(canvas, onScrub) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.onScrub = onScrub;
    this.state = null;
    this.cursor = null;
    this.hoverX = null;

    const pick = (e) => {
      const r = this.cv.getBoundingClientRect();
      const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
      return Math.round(f * Math.max(1, this.state?.day || 1));
    };
    let dragging = false;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true; canvas.setPointerCapture(e.pointerId);
      this.onScrub(pick(e));
    });
    canvas.addEventListener('pointermove', (e) => {
      const r = this.cv.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      if (dragging) this.onScrub(pick(e));
      this.draw();
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointerleave', () => {
      this.hoverX = null; this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(state, cursorDay = null) {
    this.state = state;
    this.cursor = cursorDay;
    this.draw();
    const days = state.day || 1;
    const marks = [0, 7, 30, 90, 180].filter(d => d <= Math.max(days, 7));
    if (!marks.includes(days)) marks.push(days);
    $('#tlScale').innerHTML = marks.map(d => `<span>day ${d}</span>`).join('');
  }

  draw() {
    const s = this.state;
    if (!s) return;
    const dpr = Math.min(devicePixelRatio, 2);
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    this.cv.width = W * dpr; this.cv.height = H * dpr;
    const c = this.ctx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);

    const days = Math.max(1, s.day);
    const X = (d) => 4 + (W - 8) * (d / days);
    const Y = (v) => H - 8 - (H - 20) * Math.max(0, Math.min(1, v));

    // horizon grid
    c.strokeStyle = 'rgba(120,170,220,.07)'; c.lineWidth = 1;
    for (const v of [0.25, 0.5, 0.75, 1]) {
      c.beginPath(); c.moveTo(4, Y(v)); c.lineTo(W - 4, Y(v)); c.stroke();
    }

    // per-pathway faint traces
    const P = s.history;
    if (P.length > 1 && P[0].pathways) {
      const keys = Object.keys(P[0].pathways);
      const COL = {
        regulation: '#4cc9f0', context: '#80ffdb', goal_directed: '#a0c4ff',
        new_habit: '#b8f2a6', rumination: '#ff6b6b', threat: '#ffb703',
        stress_habit: '#c77dff',
      };
      for (const k of keys) {
        c.beginPath();
        P.forEach((h, i) => {
          const x = X(h.day), y = Y(h.pathways[k]);
          i ? c.lineTo(x, y) : c.moveTo(x, y);
        });
        c.strokeStyle = (COL[k] || '#456') + '55';
        c.lineWidth = 1.2; c.stroke();
      }
    }

    // alignment curve — the headline
    c.beginPath();
    P.forEach((h, i) => {
      const x = X(h.day), y = Y(h.alignment);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    const grad = c.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#1c7f9c'); grad.addColorStop(1, '#38e0ff');
    c.strokeStyle = grad; c.lineWidth = 2.1;
    c.shadowColor = 'rgba(56,224,255,.55)'; c.shadowBlur = 9;
    c.stroke(); c.shadowBlur = 0;

    // event ticks
    for (const e of s.log) {
      const x = X(e.day);
      const cat = this._cat(e.event);
      c.fillStyle = cat === 'Practice' ? 'rgba(94,234,212,.75)'
                  : cat === 'Slip' ? 'rgba(255,92,110,.7)'
                  : 'rgba(120,150,190,.35)';
      c.fillRect(x - 0.6, H - 6, 1.4, 4);
    }

    // playhead
    const d = this.cursor ?? s.day;
    const px = X(d);
    c.strokeStyle = '#ffb454'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(px, 2); c.lineTo(px, H - 2); c.stroke();
    c.fillStyle = '#ffb454';
    c.beginPath(); c.arc(px, Y(this._alignAt(d)), 3, 0, 7); c.fill();
  }

  _alignAt(day) {
    const h = this.state.history;
    let best = h[0];
    for (const x of h) { if (x.day <= day) best = x; else break; }
    return best ? best.alignment : 0;
  }

  _cat(id) {
    return (window.__NF_EVENTCAT || {})[id] || 'State';
  }
}
