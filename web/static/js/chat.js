/* ==========================================================================
   chat.js — conversational panel
   --------------------------------------------------------------------------
   Two jobs share one input box, because they are the same gesture from the
   user's point of view: say something about your brain.

     Ask   → streamed explanation, grounded in the app's own vetted text
     Log   → free text becomes a *proposal* you confirm before anything moves

   The distinction matters and is never guessed. Silently deciding that a
   sentence was a log entry and mutating the record would be the single worst
   thing this panel could do, so the mode is an explicit control the user sets.

   Streaming is read manually rather than with EventSource, because
   EventSource cannot issue a POST and the question does not belong in a URL.
   ========================================================================== */

const $ = (s) => document.querySelector(s);

export class Chat {
  constructor({ onShow, onLog }) {
    this.onShow = onShow;      // (ids)   -> highlight regions in the 3-D view
    this.onLog = onLog;        // (id, intensity, note) -> apply an event
    this.history = [];         // verbatim turns only; never a summary
    this.busy = false;
    this.mode = 'ask';
    this.el = {
      root: $('#chat'),
      body: $('#chatBody'),
      input: $('#chatInput'),
      send: $('#chatSend'),
      status: $('#chatStatus'),
      seg: $('#chatSeg'),
    };
  }

  async init() {
    try {
      const cfg = await fetch('/api/chat/config').then((r) => r.json());
      this.cfg = cfg;
      if (cfg.enabled) {
        this.el.status.textContent = cfg.local
          ? `local · ${cfg.model}` : cfg.model;
        this.el.status.className = 'chat-status ok';
      } else {
        this.el.status.textContent = 'no model connected';
        this.el.status.className = 'chat-status off';
      }
    } catch {
      this.el.status.textContent = 'no model connected';
      this.el.status.className = 'chat-status off';
    }
    this.wire();
  }

  wire() {
    this.el.send.onclick = () => this.submit();
    this.el.input.onkeydown = (e) => {
      // Enter sends; Shift+Enter is a newline. Standard, and worth keeping
      // because the log flow often wants two or three sentences.
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.submit(); }
    };
    this.el.input.oninput = () => {
      const t = this.el.input;
      t.style.height = 'auto';
      t.style.height = Math.min(140, t.scrollHeight) + 'px';
    };
    this.el.seg.querySelectorAll('button').forEach((b) => {
      b.onclick = () => {
        this.mode = b.dataset.mode;
        this.el.seg.querySelectorAll('button')
          .forEach((x) => x.classList.toggle('active', x === b));
        this.el.input.placeholder = this.mode === 'ask'
          ? 'Ask about a region, a pathway, or what changed…'
          : 'What happened? e.g. "got annoyed but paused before replying"';
      };
    });
  }

  open() {
    this.el.root.classList.remove('hidden');
    document.body.classList.add('chat-open');
    if (!this.greeted) { this.greet(); this.greeted = true; }
    setTimeout(() => this.el.input.focus(), 60);
  }

  close() {
    this.el.root.classList.add('hidden');
    document.body.classList.remove('chat-open');
  }

  isOpen() { return !this.el.root.classList.contains('hidden'); }

  toggle() { this.isOpen() ? this.close() : this.open(); }

  greet() {
    this.add('note',
      'Ask about anything you can see. Switch to <b>Log</b> to record '
      + 'something that happened — you will always confirm before the model '
      + 'changes.');
  }

  /* ---- message rendering ------------------------------------------- */

  add(kind, html) {
    const d = document.createElement('div');
    d.className = `msg ${kind}`;
    d.innerHTML = html;
    this.el.body.appendChild(d);
    this.scroll();
    return d;
  }

  scroll() { this.el.body.scrollTop = this.el.body.scrollHeight; }

  setBusy(on) {
    this.busy = on;
    this.el.send.disabled = on;
    this.el.send.textContent = on ? '…' : 'Send';
  }

  /* ---- submit ------------------------------------------------------- */

  submit() {
    const text = this.el.input.value.trim();
    if (!text || this.busy) return;
    this.el.input.value = '';
    this.el.input.style.height = 'auto';
    this.add('user', escape(text));
    if (this.mode === 'log') this.doInterpret(text);
    else this.doAsk(text);
  }

  /* ---- ask: streamed answer ----------------------------------------- */

  async doAsk(text) {
    if (!this.cfg?.enabled) {
      this.add('note',
        'No model is connected. Start a local server (llama.cpp, Ollama, '
        + 'LM Studio or vLLM), set <code>NEUROFORGE_LLM_BASE</code> to its '
        + 'address, and restart NeuroForge.');
      return;
    }
    this.setBusy(true);
    const bubble = this.add('bot', '<span class="cursor"></span>');
    let acc = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, history: this.history.slice(-8) }),
      });

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let event = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        // SSE frames are newline-delimited; a chunk can split one in half,
        // so keep the trailing partial line in the buffer.
        const lines = buf.split('\n');
        buf = lines.pop();

        for (const line of lines) {
          if (line.startsWith('event: ')) { event = line.slice(7).trim(); continue; }
          if (!line.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }

          if (event === 'token') {
            acc += data.text;
            bubble.innerHTML = render(acc) + '<span class="cursor"></span>';
            this.scroll();
          } else if (event === 'show') {
            this.onShow?.(data.ids);
          } else if (event === 'error') {
            bubble.innerHTML = `<span class="err">${escape(data.message)}</span>`;
          } else if (event === 'done') {
            if (data.text) acc = data.text;
            bubble.innerHTML = render(acc);
            if (data.ids?.length) {
              bubble.appendChild(chips(data.ids, this.onShow));
            }
          }
        }
      }
      bubble.querySelector('.cursor')?.remove();

      this.history.push({ role: 'user', content: text });
      this.history.push({ role: 'assistant', content: acc });
    } catch (err) {
      bubble.innerHTML = `<span class="err">${escape(String(err))}</span>`;
    } finally {
      this.setBusy(false);
    }
  }

  /* ---- log: propose, then confirm ------------------------------------ */

  async doInterpret(text) {
    this.setBusy(true);
    const bubble = this.add('bot', 'Reading that…');
    try {
      const r = await fetch('/api/sim/interpret', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      }).then((x) => x.json());

      const props = r.interpretation?.proposals || [];
      if (!props.length) {
        bubble.innerHTML =
          'I could not match that to anything I know how to record. '
          + 'Try naming the action — paused, avoided, reframed, exercised.';
        return;
      }

      bubble.innerHTML =
        '<div class="prop-intro">That looks like:</div>'
        + props.map((p, i) => `
          <button class="prop" data-i="${i}">
            <span class="prop-main">
              <b>${escape(p.label)}</b>
              <i>${escape(p.why || '')}</i>
            </span>
            <span class="prop-conf ${conf(p.confidence)}">
              ${Math.round(p.confidence * 100)}%
            </span>
          </button>`).join('')
        + '<div class="prop-foot">Nothing has changed yet. Pick one to record '
        + 'it, and your own words are kept as the note.</div>';

      bubble.querySelectorAll('.prop').forEach((b) => {
        b.onclick = async () => {
          const p = props[+b.dataset.i];
          bubble.querySelectorAll('.prop').forEach((x) => {
            x.disabled = true;
            x.classList.toggle('chosen', x === b);
          });
          await this.onLog?.(p.event, p.intensity, text);
          this.add('note', `Recorded <b>${escape(p.label)}</b>.`);
        };
      });
    } catch (err) {
      bubble.innerHTML = `<span class="err">${escape(String(err))}</span>`;
    } finally {
      this.setBusy(false);
    }
  }
}

/* ---- helpers -------------------------------------------------------- */

function escape(s) {
  return String(s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// Minimal formatting only. The model's output is escaped first, so this can
// never inject markup: it only re-introduces the two forms that actually help
// readability here.
function render(t) {
  let s = escape(t);
  s = s.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  s = s.replace(/\b(REAL|MODEL):/g,
    (_, w) => `<span class="tagi ${w.toLowerCase()}">${w}</span>`);
  return s.replace(/\n{2,}/g, '</p><p>').replace(/^/, '<p>') + '</p>';
}

function chips(ids, onShow) {
  const wrap = document.createElement('div');
  wrap.className = 'chips';
  ids.forEach((id) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = id;
    b.onclick = () => onShow?.([id]);
    wrap.appendChild(b);
  });
  return wrap;
}

function conf(c) { return c >= 0.7 ? 'hi' : c >= 0.45 ? 'mid' : 'lo'; }
