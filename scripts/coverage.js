/* Checks that every cortical region and every structure in the scene can
 * actually be selected and produces a panel. A region that exists in the
 * metadata but has no vertices would open a panel that highlights nothing,
 * which has happened before. */
const { chromium } = require('playwright-core');
// Overridable: this path is Windows-specific and not guaranteed even there.
const EDGE = process.env.NF_BROWSER ||
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL = process.argv[2] || 'http://127.0.0.1:8770/';

(async () => {
  const b = await chromium.launch({
    executablePath: EDGE,
    args: ['--use-gl=angle', '--use-angle=swiftshader',
           '--enable-unsafe-swiftshader'],
  });
  const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
  const errs = [];
  p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(URL, { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => window.NF && window.NF.scene, { timeout: 90000 });
  await p.waitForTimeout(3000);

  const counts = await p.evaluate(() => {
    const out = { labelCounts: {}, regions: [], structures: [] };
    for (const m of [window.NF.viewer.cortexL, window.NF.viewer.cortexR]) {
      const a = m.geometry.getAttribute('aLabel');
      for (let i = 0; i < a.count; i++) {
        const v = Math.round(a.getX(i));
        out.labelCounts[v] = (out.labelCounts[v] || 0) + 1;
      }
    }
    out.regions = window.NF.scene.cortical_regions
      .map(r => [r.label_index, r.short]);
    out.structures = window.NF.scene.structures.map(s => s.id);
    return out;
  });

  let bad = 0;
  for (const [idx, short] of counts.regions) {
    const n = counts.labelCounts[idx] || 0;
    await p.evaluate(i => window.NF.debug.showCortex(i), idx);
    await p.waitForTimeout(120);
    const title = await p.textContent('.p-title').catch(() => null);
    const ok = n > 0 && title;
    if (!ok) bad++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} region ${String(idx).padStart(2)} ` +
                `${short.padEnd(15)} ${String(n).padStart(6)} verts  ${title}`);
  }
  for (const id of counts.structures) {
    await p.evaluate(i => window.NF.debug.showStructure(i), id);
    await p.waitForTimeout(120);
    const title = await p.textContent('.p-title').catch(() => null);
    if (!title) bad++;
    console.log(`${title ? 'ok  ' : 'FAIL'} struct ${id.padEnd(14)} ${title}`);
  }

  // Composite terms. A group must highlight strictly less of the brain than
  // no selection at all, or it is not focusing on anything.
  const groups = await p.evaluate(() => (window.NF.scene.groups || [])
    .map(g => [g.id, g.regions.length + g.structures.length]));
  await p.evaluate(() => window.NF.viewer.selectMany([], []));
  await p.waitForTimeout(400);
  const litAll = await p.evaluate(() => window.NF.debug.litFraction());
  for (const [id, n] of groups) {
    const ok0 = await p.evaluate(i => window.NF.debug.showGroup(i), id);
    await p.waitForTimeout(400);
    const title = await p.textContent('.p-title').catch(() => null);
    const lit = await p.evaluate(() => window.NF.debug.litFraction());
    const ok = ok0 && title && lit < litAll;
    if (!ok) bad++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} group  ${id.padEnd(14)} ` +
                `${n} parts  lit ${lit.toFixed(3)}/${litAll.toFixed(3)}  ${title}`);
  }

  console.log('FAILURES:', bad);
  console.log('ERRORS:', errs.length);
  errs.slice(0, 5).forEach(e => console.log('  ', e));
  await b.close();
  process.exit(bad || errs.length ? 1 : 0);
})();
