/* Headless smoke test of the WebGL frontend.
 * Loads the app in Edge, waits for the loader to clear, exercises the three
 * modes and a log action, and reports any console/page errors. */

const { chromium } = require('playwright-core');

// Overridable: this path is Windows-specific and not guaranteed even there.
const EDGE = process.env.NF_BROWSER ||
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL = process.argv[2] || 'http://127.0.0.1:8770/';

(async () => {
  const browser = await chromium.launch({
    executablePath: EDGE,
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader',
           '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
  const errors = [];
  const logs = [];
  page.on('console', m => {
    const t = m.text();
    logs.push(`${m.type()}: ${t}`);
    if (m.type() === 'error') errors.push(t);
  });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

  await page.goto(URL, { waitUntil: 'domcontentloaded' });

  // wait for boot to finish
  try {
    await page.waitForFunction(
      () => document.querySelector('#loader')?.classList.contains('done'),
      { timeout: 90000 });
    console.log('BOOT: ok');
  } catch {
    const msg = await page.textContent('#loaderMsg').catch(() => '?');
    console.log('BOOT: FAILED — loader says: ' + msg);
  }

  const stat = await page.evaluate(() => {
    const V = window.NF?.viewer;
    return {
      hasViewer: !!V,
      structures: V ? V.structures.size : 0,
      cortexTris: V?.cortexL
        ? (V.cortexL.geometry.index.count + V.cortexR.geometry.index.count) / 3
        : 0,
      edges: window.NF?.circuits?.edges.size || 0,
      day: window.NF?.state?.day,
      align: window.NF?.state?.alignment,
      glLost: !V?.renderer.getContext() ||
              V.renderer.getContext().isContextLost(),
    };
  });
  console.log('SCENE:', JSON.stringify(stat));

  // exercise the modes
  for (const mode of ['circuits', 'simulation', 'anatomy']) {
    await page.click(`#modebar button[data-mode="${mode}"]`);
    await page.waitForTimeout(700);
    console.log(`MODE ${mode}: ok`);
  }

  // views
  for (const v of ['superior', 'medial', 'anterior', 'lateral']) {
    await page.click(`.viewbtn[data-view="${v}"]`);
    await page.waitForTimeout(400);
  }
  console.log('VIEWS: ok');

  // pick a structure from the list
  await page.click('.structrow[data-id="amygdala"] .nm');
  await page.waitForTimeout(500);
  const panelTitle = await page.textContent('.p-title').catch(() => null);
  console.log('PANEL:', panelTitle);

  // functional-network overlay. Isolating the default mode network should
  // leave clearly less of the surface lit than the whole cortex does; that is
  // measured rather than eyeballed, because a "highlight" that accidentally
  // brightens the brain has shipped here before.
  const netCount = await page.$$eval('#netList .structrow', els => els.length);
  console.log('NETWORKS:', netCount);
  if (netCount) {
    // Clear the structure selection first, otherwise the baseline is measured
    // while the cortex is already ghosted and the comparison means nothing.
    await page.evaluate(() => window.NF.viewer.select(null));
    await page.waitForTimeout(500);
    const litAll = await page.evaluate(() => window.NF.debug.litFraction());
    await page.check('#chkNetworks');
    await page.waitForTimeout(400);
    await page.click('.structrow[data-net="7"]');
    await page.waitForTimeout(700);
    const netTitle = await page.textContent('.p-title').catch(() => null);
    const litDmn = await page.evaluate(() => window.NF.debug.litFraction());
    console.log('DMN PANEL:', netTitle);
    console.log('LIT all/dmn:', litAll.toFixed(3), litDmn.toFixed(3),
                litDmn < litAll ? 'ok' : 'SUSPECT');
    await page.uncheck('#chkNetworks');
    await page.waitForTimeout(300);
  }

  // logging flow
  await page.click('#btnTrigger');
  await page.waitForTimeout(600);
  const choices = await page.$$eval('#choiceHost .choice', els => els.length);
  console.log('CHOICES:', choices);
  await page.click('.choice[data-event="name_emotion"]');
  await page.waitForTimeout(900);
  const delta = await page.textContent('.delta-row .dval').catch(() => null);
  console.log('DELTA:', delta && delta.trim());
  await page.click('#modalClose');

  // demo + replay
  await page.evaluate(async () => {
    const r = await fetch('/api/sim/demo', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weeks: 8 }) });
    return r.json();
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => document.querySelector('#loader')?.classList.contains('done'),
    { timeout: 90000 });
  await page.click('#modebar button[data-mode="simulation"]');
  await page.waitForTimeout(800);
  const after = await page.evaluate(() => ({
    day: window.NF?.state?.day,
    align: window.NF?.state?.alignment,
  }));
  console.log('AFTER DEMO:', JSON.stringify(after));

  await page.screenshot({ path: 'nf_anatomy.png' });
  await page.click('#modebar button[data-mode="circuits"]');
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'nf_circuits.png' });

  // measure average frame time
  const fps = await page.evaluate(() => new Promise(res => {
    let n = 0; const t0 = performance.now();
    const tick = () => { if (++n < 60) requestAnimationFrame(tick);
                         else res(Math.round(60000 / (performance.now() - t0))); };
    requestAnimationFrame(tick);
  }));
  console.log('FPS (software rasteriser):', fps);

  console.log('\nERRORS:', errors.length);
  errors.slice(0, 12).forEach(e => console.log('  ! ' + e));

  await browser.close();
})();
