// 프로덕션 빌드(vite preview) + 실제 백엔드를 대상으로 하는 스모크 테스트.
// 사용: FAKE_VIDEO=/path/face.y4m node e2e/smoke.mjs  (playwright 패키지 필요)
import { chromium } from 'playwright';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const FAKE_VIDEO = process.env.FAKE_VIDEO;

const args = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'];
if (FAKE_VIDEO) args.push(`--use-file-for-fake-video-capture=${FAKE_VIDEO}`);

const browser = await chromium.launch({ args });
const context = await browser.newContext({ permissions: ['camera'] });
const page = await context.newPage();

const pageErrors = [];
page.on('pageerror', (err) => pageErrors.push(err.message));

const step = (name) => console.log(`\n▶ ${name}`);
const fail = async (msg) => {
  await page.screenshot({ path: 'e2e-failure.png', fullPage: true });
  throw new Error(msg);
};

try {
  step('/ redirects to /measurement');
  await page.goto(BASE_URL);
  await page.waitForURL('**/measurement');

  step('FaceMesh loads in the production bundle');
  await page.waitForFunction(() => typeof window.FaceMesh === 'function', null, { timeout: 30000 });

  if (FAKE_VIDEO) {
    step('face detected from fake camera -> start button enabled');
    const startButton = page.locator('button.fab-measure');
    await startButton.waitFor();
    await page.waitForFunction(
      () => !document.querySelector('button.fab-measure')?.disabled,
      null,
      { timeout: 60000 }
    ).catch(async () => fail(`start button stayed disabled: "${await startButton.textContent()}"`));

    step('run a 10s measurement against the backend');
    await startButton.click();
    const outcome = await Promise.race([
      page.locator('.result-number').first().waitFor({ timeout: 90000 }).then(() => 'result'),
      page.getByText(/측정 품질 기준 미달|Quality below threshold/).waitFor({ timeout: 90000 }).then(() => 'quality-fail'),
    ]).catch(() => 'timeout');
    console.log(`  outcome: ${outcome}`);
    if (outcome === 'result') {
      console.log(`  result: ${(await page.locator('.result-number').first().textContent()).trim()}`);
    } else if (outcome === 'timeout') {
      const text = (await page.locator('body').innerText()).slice(0, 1500);
      await fail(`measurement produced neither a result nor a quality failure. Page text:\n${text}`);
    }
  }

  step('summary page renders charts (recharts)');
  await page.goto(`${BASE_URL}/summary`);
  await page.locator('.recharts-surface').first().waitFor({ timeout: 20000 })
    .catch(async () => fail('no recharts chart rendered on /summary'));
  console.log(`  charts: ${await page.locator('.recharts-surface').count()}`);

  step('lifestyle page loads data');
  await page.goto(`${BASE_URL}/lifestyle`);
  await page.getByText(/불러오지 못했습니다|could not load/i).waitFor({ timeout: 5000 })
    .then(async () => fail('lifestyle page shows a load error'))
    .catch((err) => { if (err.message.includes('lifestyle page')) throw err; });

  if (pageErrors.length) await fail(`uncaught page errors:\n${pageErrors.join('\n')}`);
  console.log('\n✅ smoke test passed');
} finally {
  await browser.close();
}
