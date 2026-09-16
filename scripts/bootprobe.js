const { chromium } = require('playwright-core');
// Overridable: this path is Windows-specific and not guaranteed even there.
const EDGE = process.env.NF_BROWSER ||
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL = process.argv[2] || 'http://127.0.0.1:8770/';

(async () => {
  const browser = await chromium.launch({
    executablePath: EDGE, headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader',
           '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on('console', m => console.log(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', e => console.log('PAGEERROR: ' + (e.stack || e.message)));
  page.on('requestfailed', r =>
    console.log('REQFAIL: ' + r.url() + ' :: ' + r.failure()?.errorText));
  page.on('response', r => {
    if (r.status() >= 400) console.log(`HTTP ${r.status()} ${r.url()}`);
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(40000);

  console.log('--- probe ---');
  const probe = await page.evaluate(() => ({
    loader: document.querySelector('#loaderMsg')?.textContent,
    done: document.querySelector('#loader')?.classList.contains('done'),
    hasNF: !!window.NF,
    keys: window.NF ? Object.keys(window.NF) : [],
    viewer: !!window.NF?.viewer,
    webgl2: (() => { try { return !!document.createElement('canvas')
      .getContext('webgl2'); } catch (e) { return 'err ' + e.message; } })(),
  }));
  console.log(JSON.stringify(probe, null, 2));
  await page.screenshot({ path: 'nf_boot.png' });
  await browser.close();
})();
