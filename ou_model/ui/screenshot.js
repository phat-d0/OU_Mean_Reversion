// Capture dashboard screenshots with Playwright (node). Start the app first:
//   .venv/bin/python -m streamlit run ou_model/ui/app.py --server.headless true --server.port 8599
// then: node ou_model/ui/screenshot.js [baseUrl]
const { chromium } = require('playwright');
const path = require('path');

const BASE = process.argv[2] || 'http://localhost:8599';
const OUT = path.join(__dirname, 'screenshots');
const SHOTS = [
  { file: 'pair_commodities_crack_321.png', query: '?asset_class=commodities&pair=crack_321', tab: null },
  { file: 'pair_fx_eurchf.png', query: '?asset_class=fx&pair=eurchf', tab: null },
  { file: 'cross_asset.png', query: '', tab: 'Cross-asset summary' },
  { file: 'simulator.png', query: '', tab: 'Simulator' },
];

async function settle(page, minPlots) {
  await page.waitForSelector('[data-testid="stApp"]');
  await page.waitForFunction(
    (n) => document.querySelectorAll('.js-plotly-plot .main-svg').length >= n &&
      !document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"]'),
    minPlots, { timeout: 60000 });
  await page.waitForTimeout(2500);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  for (const s of SHOTS) {
    await page.goto(BASE + '/' + s.query);
    await settle(page, 1);
    if (s.tab) {
      await page.getByRole('tab', { name: s.tab }).click();
      await page.waitForTimeout(2500);
    }
    await page.screenshot({ path: path.join(OUT, s.file), fullPage: false });
    console.log('saved', s.file);
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
