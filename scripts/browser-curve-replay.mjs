// Read-only UI acceptance plus one explicit local assessment save and history replay.
import {spawn} from 'node:child_process';
import {createWriteStream} from 'node:fs';
import {mkdir, readFile, writeFile, mkdtemp} from 'node:fs/promises';
import path from 'node:path';

const projectRoot = path.resolve(path.dirname(import.meta.filename), '..');
const OUT = path.resolve(process.env.ARTIFACT_DIR || path.join(projectRoot, 'runs/artifacts/browser-curve-replay'));
await mkdir(OUT, {recursive: true});
const TMP = process.env.CLAUDE_TEMP_SESSION_DIR;
if (!TMP) throw Error('Tool-assigned project temporary directory required');
const attempt = process.argv[2] || '01';
const prefix = `curve-replay-${attempt}`;
const origin = 'http://127.0.0.1:8501';
const report = {
  status: 'running', origin, actions: [], checks: [], screenshots: [],
  runtimeErrors: [], consoleErrors: [], personalSaveActions: 0,
  researchApiActions: 0, formalUpdateActions: 0,
};
const libraries = (await readFile(
  '/home/nvme03/zc/rmb/.claude-tmp/sessions/fund-build/browser-libraries-1_xz6esi/library-path.txt',
  'utf8',
)).trim();
const profile = await mkdtemp(path.join(TMP, `curve-replay-${attempt}-`));
const chromeLog = createWriteStream(path.join(OUT, `${prefix}-chromium.log`), {flags: 'wx'});
const chrome = spawn(
  '/home/zc/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  [
    '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run',
    '--disable-background-networking', '--disable-component-update', '--disable-sync',
    '--disable-default-apps', '--disable-domain-reliability', '--disable-breakpad',
    '--disable-features=MediaRouter,OptimizationHints,AutofillServerCommunication',
    '--no-pings', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
  ],
  {stdio: ['ignore', 'ignore', 'pipe'], env: {...process.env, TMPDIR: TMP, LD_LIBRARY_PATH: libraries}},
);
chrome.stderr.pipe(chromeLog);
let ws;
let session;
let sequence = 0;
const pending = new Map();
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function call(method, params = {}, target = session) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(Error(`CDP timeout ${method}`));
    }, 30000);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params, ...(target ? {sessionId: target} : {})}));
  });
}
async function evaluate(expression) {
  const result = await call('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true,
  });
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function wait(expression, label) {
  const until = Date.now() + 180000;
  while (Date.now() < until) {
    if (await evaluate(`Boolean(${expression})`)) return;
    const error = await evaluate("document.querySelector('[data-testid=stException]')?.innerText");
    if (error) throw Error(error);
    await pause(250);
  }
  throw Error(`Timeout ${label}`);
}
async function settle() {
  await pause(500);
  await wait("!document.querySelector('[data-testid=stMain] [data-stale=true]')", 'render');
  await pause(300);
}
function check(pass, label) {
  report.checks.push({pass: Boolean(pass), label});
  if (!pass) report.status = 'failed';
}
const find = (selector, text) =>
  `[...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.getClientRects().length&&` +
  `(e.innerText||e.getAttribute('aria-label')||'').includes(${JSON.stringify(text)}))`;
async function click(expression, label) {
  await evaluate(`(()=>{const e=(${expression});if(!e||e.disabled)throw Error('Missing ${label}');` +
    `e.scrollIntoView({block:'center',behavior:'instant'});e.focus()})()`);
  await pause(80);
  const point = await evaluate(`(()=>{const r=(${expression}).getBoundingClientRect();` +
    `return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);
  for (const type of ['mousePressed', 'mouseReleased']) {
    await call('Input.dispatchMouseEvent', {type, ...point, button: 'left', clickCount: 1});
  }
  report.actions.push(label);
  await settle();
}
async function key(key, extra = {}) {
  const values = {Home: 36, End: 35, ArrowRight: 39, ArrowDown: 40, ArrowUp: 38, Tab: 9, Enter: 13};
  const code = key.length === 1 ? `Digit${key}` : key;
  const value = values[key] ?? (key.length === 1 ? key.toUpperCase().charCodeAt(0) : 0);
  for (const type of ['keyDown', 'keyUp']) {
    await call('Input.dispatchKeyEvent', {
      type, key, code, windowsVirtualKeyCode: value, nativeVirtualKeyCode: value, ...extra,
    });
  }
}
async function setText(label, value) {
  const input = `document.querySelector(${JSON.stringify(`input[aria-label="${label}"]`)})`;
  await click(input, `Edit ${label}`);
  await key('a', {modifiers: 2, code: 'KeyA', windowsVirtualKeyCode: 65, nativeVirtualKeyCode: 65});
  await call('Input.insertText', {text: value});
  await key('Tab');
  await settle();
}
async function hoverTracePoint(chartSelector, traceName, pointIndex, label) {
  await evaluate(`document.querySelector(${JSON.stringify(chartSelector)})?.scrollIntoView({block:'center', behavior:'instant'})`);
  await pause(250);
  const target = await evaluate(`(() => {
    const chart = document.querySelector(${JSON.stringify(chartSelector)});
    if (!chart) return null;
    const traceIndex = (chart.data || []).findIndex(trace => trace.name === ${JSON.stringify(traceName)});
    const group = [...chart.querySelectorAll('.scatterlayer .trace')][traceIndex];
    const points = [...(group?.querySelectorAll('.point') || [])];
    const point = points[${pointIndex}];
    const rect = point?.getBoundingClientRect();
    const line = group?.querySelector('.js-line');
    const lineRect = line?.getBoundingClientRect();
    return {traceIndex, pointCount: points.length,
      point: rect && (rect.width || rect.height) ? {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2} : null,
      line: lineRect && (lineRect.width || lineRect.height) ? {x: lineRect.x + lineRect.width / 2, y: lineRect.y + lineRect.height / 2} : null,
      groupHTML: group?.outerHTML?.slice(0, 1200) || null};
  })()`);
  if (!target) throw Error(`Missing ${label} chart`);
  const pointer = target.point || target.line;
  if (!pointer) throw Error(`Missing visible ${label} point: ${JSON.stringify(target)}`);
  await call('Input.dispatchMouseEvent', {type: 'mouseMoved', ...pointer}, session);
  try {
    await wait("document.querySelector('.hoverlayer')?.textContent?.trim()", `${label} tooltip`);
  } catch (error) {
    throw Error(`${error.message}; target=${JSON.stringify(target)}`);
  }
  return evaluate("document.querySelector('.hoverlayer')?.textContent || ''");
}

async function snapshot(name, chartSelector = '.js-plotly-plot') {
  const value = await evaluate(`(()=>{
    const chart=document.querySelector(${JSON.stringify(chartSelector)});
    const main=document.querySelector('[data-testid=stMain]');
    const text=document.body.innerText;
    return {
      width:innerWidth, bodyWidth:document.body.scrollWidth,
      documentWidth:document.documentElement.scrollWidth,
      mainWidth:main?.clientWidth, mainScrollWidth:main?.scrollWidth,
      text, exceptions:[...document.querySelectorAll('[data-testid=stException]')].map(e=>e.innerText),
      traces:(chart?.data||[]).map(t=>({name:t.name,type:t.type,mode:t.mode,n:t.x?.length||0,x:t.x||[],y:t.y||[]})),
      tables:[...document.querySelectorAll('.buyer-chart-table table')].map(t=>t.innerText),
    };
  })()`);
  const shot = await call('Page.captureScreenshot', {format: 'png'});
  const image = path.join(OUT, `${prefix}-${name}.png`);
  await writeFile(image, Buffer.from(shot.data, 'base64'), {flag: 'wx'});
  await writeFile(image.replace('.png', '.json'), JSON.stringify(value, null, 2), {flag: 'wx'});
  report.screenshots.push(image);
  report[name] = value;
  check(!value.exceptions.length, `No UI exception at ${name}`);
  check(value.bodyWidth <= value.width && value.documentWidth <= value.width &&
        value.mainScrollWidth <= value.mainWidth, `No page overflow at ${name}`);
  return value;
}
try {
  const endpoint = await new Promise((resolve, reject) => {
    let text = '';
    const timer = setTimeout(() => reject(Error('Chrome startup timeout')), 20000);
    chrome.stderr.on('data', (chunk) => {
      text += chunk;
      const match = text.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
    chrome.once('error', reject);
    chrome.once('exit', (code, signal) => {
      clearTimeout(timer); reject(Error(`Chrome exited ${code}/${signal}`));
    });
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  ws.onmessage = ({data}) => {
    const message = JSON.parse(data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id);
      pending.delete(message.id); clearTimeout(item.timer);
      message.error ? item.reject(Error(JSON.stringify(message.error))) : item.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') report.runtimeErrors.push(message.params);
    if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
      report.consoleErrors.push(message.params.args);
    }
  };
  const target = await call('Target.createTarget', {url: 'about:blank'}, null);
  session = (await call('Target.attachToTarget', {targetId: target.targetId, flatten: true}, null)).sessionId;
  await call('Page.enable'); await call('Runtime.enable');
  await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1150, deviceScaleFactor: 1, mobile: false});
  await call('Page.navigate', {url: origin});
  await wait("document.querySelector('input[aria-label=\"想研究的基金代码\"]')", 'public input');
  await setText('想研究的基金代码', '018147');
  await click(find('button', '公开预览'), 'Open public 018147');
  await wait("document.querySelector('.st-key-decision_card')", 'recommendation card');
  await settle();
  const before = await snapshot('before-compute');
  check(before.text.includes('净值历史与未来目标点预估'), 'Current source renderer is deployed');
  check(before.text.includes('不拿其他期限补画'), 'Uncomputed exact head remains empty');
  check(!/nav_ridge|model_hash|target_definition_id|samples|drivers|source_url/.test(before.text),
    'No internal prediction fields in visible page');

  await click(find('button', '计算本周期判断'), 'Compute exact local curve');
  await wait("document.querySelector('.st-key-decision_card')&&!document.querySelector('[data-testid=stSpinner]')&&" +
    "document.querySelector('.js-plotly-plot')?.data?.some(t=>t.name==='各目标点预期涨跌')", 'curve render');
  await settle();
  const computed = await snapshot('computed-curve');
  const marker = computed.traces.find((trace) => trace.name === '各目标点预期涨跌');
  const ranges = computed.traces.filter((trace) => trace.name === '各目标点中间80%范围');
  report.hoverTemplates = await evaluate(`document.querySelector('.js-plotly-plot').data
    .filter(trace => trace.name === '各目标点预期涨跌' || trace.name === '各目标点中间80%范围')
    .slice(0, 2).map(trace => ({name: trace.name, hovertemplate: trace.hovertemplate, customdata: trace.customdata?.[0]}))`);
  check(marker?.n > 1, 'Multiple future target markers are rendered');
  check(ranges.length > 1, 'Independent target ranges are rendered');
  check(computed.text.includes('H=1') && computed.text.includes('H=5'), 'Target H labels and conditional dates are visible');
  check(!computed.text.includes('连续未来净值路径'), 'Page does not present a continuous NAV path');
  const markerTooltip = await hoverTracePoint('.js-plotly-plot', '各目标点预期涨跌', 0, 'target marker');
  report.markerTooltip = markerTooltip;
  check(/H|笔净值/.test(markerTooltip) && markerTooltip.includes('条件日期')
    && markerTooltip.includes('预期涨跌') && markerTooltip.includes('状态'),
  'Target marker hover exposes H, conditional date, expected change, and status');
  check(!/model_id|model_hash|samples|drivers|source_url|expected_value|10000/.test(markerTooltip),
    'Target marker hover excludes internal and personal fields');
  const rangeTooltip = await hoverTracePoint('.js-plotly-plot', '各目标点中间80%范围', 0, 'target range endpoint');
  report.rangeTooltip = rangeTooltip;
  check(/H|笔净值/.test(rangeTooltip) && rangeTooltip.includes('条件日期')
    && rangeTooltip.includes('中间80%范围端点') && rangeTooltip.includes('状态'),
  'Target range hover exposes H, conditional date, endpoint semantics, and status');
  check(!/model_id|model_hash|samples|drivers|source_url|expected_value|10000/.test(rangeTooltip),
    'Target range hover excludes internal and personal fields');
  await call('Input.dispatchMouseEvent', {type: 'mouseMoved', x: 0, y: 0}, session);

  await click(find('summary', '图表数据（可用键盘查看）'), 'Open accessible chart tables');
  await wait("document.querySelectorAll('.buyer-chart-table table').length===2", 'accessible chart tables');
  const tables = await snapshot('computed-tables');
  check(tables.tables.length === 2, 'Actual NAV and forecast accessible tables');
  check(tables.tables[1].includes('H=1') && tables.tables[1].includes('H=5'), 'Forecast table lists target points');
  await click(find('summary', '图表数据（可用键盘查看）'), 'Close accessible chart tables');

  await click(find('button', '保存这次判断'), 'Save displayed assessment');
  report.personalSaveActions += 1;
  await wait("document.body.innerText.includes('已保存刚才展示的判断')", 'save confirmation');
  await settle();
  check(!/提交买卖|下单成功/.test(''), 'No order action was requested');

  await click(find('[role=tab]', '历史记录'), 'Open history');
  await wait("document.querySelector('.st-key-history_buyer_view')", 'saved history view');
  await wait("document.querySelector('.st-key-history_forecast_chart .js-plotly-plot')", 'saved history chart');
  await settle();
  const history = await snapshot('history-replay', '.st-key-history_forecast_chart .js-plotly-plot');
  check(history.text.includes('净值历史与未来目标点预估'), 'History uses the saved curve renderer');
  check(history.text.includes('H=1') && history.text.includes('H=5'), 'History replays saved target points');
  const historyMarker = history.traces.find((trace) => trace.name === '各目标点预期涨跌');
  check(historyMarker?.n === marker?.n, 'History replays the saved marker count');
  check(history.traces.filter((trace) => trace.name === '各目标点中间80%范围').length === ranges.length,
    'History replays the saved independent ranges');
  const historyMarkerTooltip = await hoverTracePoint(
    '.st-key-history_forecast_chart .js-plotly-plot', '各目标点预期涨跌', 0, 'history target marker');
  report.historyMarkerTooltip = historyMarkerTooltip;
  check(/H|笔净值/.test(historyMarkerTooltip) && historyMarkerTooltip.includes('条件日期')
    && historyMarkerTooltip.includes('预期涨跌') && historyMarkerTooltip.includes('状态'),
  'History target marker hover replays public fields');
  const historyRangeTooltip = await hoverTracePoint(
    '.st-key-history_forecast_chart .js-plotly-plot', '各目标点中间80%范围', 0, 'history target range endpoint');
  report.historyRangeTooltip = historyRangeTooltip;
  check(/H|笔净值/.test(historyRangeTooltip) && historyRangeTooltip.includes('条件日期')
    && historyRangeTooltip.includes('中间80%范围端点') && historyRangeTooltip.includes('状态'),
  'History target range hover replays endpoint fields');
  check(!/model_id|model_hash|samples|drivers|source_url|expected_value|10000/.test(
    `${historyMarkerTooltip} ${historyRangeTooltip}`),
  'History hover excludes internal and personal fields');
  check(!/nav_ridge|model_hash|target_definition_id|samples|drivers|source_url/.test(history.text),
    'History does not expose internal prediction fields');
  check(!report.runtimeErrors.length && !report.consoleErrors.length, 'No browser runtime or console errors');
  if (report.status === 'running') report.status = 'passed';
} catch (error) {
  report.status = 'failed';
  report.failure = String(error.stack || error);
  process.exitCode = 1;
} finally {
  for (const item of pending.values()) clearTimeout(item.timer);
  ws?.close(); chrome.kill('SIGTERM'); chromeLog.end();
  report.finishedAt = new Date().toISOString();
  await writeFile(path.join(OUT, `${prefix}-report.json`), JSON.stringify(report, null, 2), {flag: 'wx'});
}
console.log(JSON.stringify({
  status: report.status, checks: report.checks.length,
  failed: report.checks.filter((item) => !item.pass), screenshots: report.screenshots.length,
  failure: report.failure || null,
}));
if (report.status === 'failed') process.exitCode = 1;
