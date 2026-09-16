/* ============================================================================
 * api.js — the only channel between the browser and the Python simulation.
 * The renderer holds no simulation state of its own.
 * ========================================================================= */

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function jpost(url, body = {}) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

export const API = {
  scene:    () => jget('/api/scene'),
  model:    () => jget('/api/model'),
  state:    () => jget('/api/sim/state'),
  log:      (event, opts = {}) => jpost('/api/sim/log', { event, ...opts }),
  // Suggests events from free text. Does not change the simulation.
  interpret: (text) => jpost('/api/sim/interpret', { text }),
  advance:  (days = 1) => jpost('/api/sim/advance', { days }),
  replay:   (day) => jpost('/api/sim/replay', { day }),
  reset:    () => jpost('/api/sim/reset'),
  demo:     (weeks = 12, adherence = 0.72) =>
              jpost('/api/sim/demo', { weeks, adherence }),
  save:     (name) => jpost('/api/sim/save', { name }),
  load:     (name) => jpost('/api/sim/load', { name }),
};

/* ---- binary decoding of the mesh buffers -------------------------------- */

function b64bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function decodeF32(b64) {
  const b = b64bytes(b64);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

export function decodeU32(b64) {
  const b = b64bytes(b64);
  return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

export function decodeU8(b64) {
  return b64bytes(b64);
}

export function decodeU16(b64) {
  const b = b64bytes(b64);
  return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2);
}
