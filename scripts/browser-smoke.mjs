// Real Chromium smoke with Node 22 built-ins; no paid services or npm dependencies.
import { execFileSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const url = process.env.APP_URL || 'http://127.0.0.1:8501';
const adviceMode = process.env.ADVICE_SMOKE === '1';
const researchMode = process.env.RESEARCH_SMOKE === '1';
const publicMode = process.env.PUBLIC_SMOKE === '1' || researchMode || adviceMode;
const projectRoot = path.resolve(import.meta.dirname, '..');
const artifactsRoot = path.join(projectRoot, 'runs/artifacts');
const storeRoot = process.env.SMOKE_ROOT ? path.resolve(process.env.SMOKE_ROOT) : null;
if (adviceMode && researchMode) throw Error('Select ADVICE_SMOKE or RESEARCH_SMOKE, not both');
if (adviceMode) {
  const relativeRoot = storeRoot && path.relative(artifactsRoot, storeRoot);
  if (!relativeRoot || relativeRoot.startsWith('..') || path.isAbsolute(relativeRoot)) {
    throw Error('ADVICE_SMOKE requires an isolated SMOKE_ROOT under runs/artifacts');
  }
  const endpoint = new URL(url);
  if (endpoint.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname)
      || endpoint.port !== '8501') {
    throw Error('ADVICE_SMOKE only uses the parent-started local app on port 8501');
  }
}
const output = path.resolve(process.env.ARTIFACT_DIR || 'runs/artifacts/ui-smoke');
const temporary = path.resolve(process.env.TMPDIR || path.join(projectRoot, '.claude-tmp/sessions/fund-build'));
const chrome = process.env.CHROMIUM || '/home/zc/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome';
// Read-only auditing must not initialize Workspace or import frozen research.
// In particular, Workspace's schema migration belongs before the UI baseline,
// not to an audit call that would mask an implicit write.
const adviceAudit = (withReport = false, allowMissing = false) => JSON.parse(execFileSync(
  path.join(projectRoot, '.venv/bin/python'), ['-I', '-c', `
import hashlib,json,sqlite3,sys
from datetime import datetime
from decimal import Decimal,InvalidOperation,localcontext
from pathlib import Path
from zoneinfo import ZoneInfo
root,work=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert work.is_relative_to((root/'runs/artifacts').resolve()) and work!=(root/'runs/artifacts').resolve(), 'SMOKE_ROOT escapes isolated artifacts'
source=root/'data/public.sqlite3'
db=work/'public.sqlite3'
assert db.resolve()!=source.resolve(), 'Smoke database aliases original public data'
if db.exists():
    assert not db.samefile(source), 'Smoke database is a hard link to original public data'
def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str)
def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()
def connect(file):
    c=sqlite3.connect(file.resolve().as_uri()+'?mode=ro',uri=True,timeout=15)
    c.row_factory=sqlite3.Row
    c.execute('PRAGMA query_only=ON')
    c.execute('BEGIN')
    assert c.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()[0]=='public', 'Expected PUBLIC database'
    return c
def tables(c):
    result={}
    for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        name=row[0]
        quoted='"'+name.replace('"','""')+'"'
        rows=sorted((list(item) for item in c.execute('SELECT * FROM '+quoted)),key=encoded)
        result[name]={'rows':len(rows),'hash':digest(rows)}
    return result
catalog_path=root/'configs/research-catalog.json'
catalog=json.loads(catalog_path.read_text())
pins={}
for study in catalog['studies']:
    references=[study['source'],*study.get('evidence',[])]
    references += [window[key] for window in study['windows'] for key in ('inputs','report','manifest')]
    for pin in references:
        file=(root/pin['path']).resolve()
        assert file.is_relative_to(root), 'Pinned evidence escapes project'
        value=hashlib.sha256(file.read_bytes()).hexdigest()
        assert value==pin['sha256'], 'Pinned evidence already differs: '+pin['path']
        pins[pin['path']]=value
with connect(source) as c:
    original=tables(c)
result={'public_hash':digest(original),'public_file_hash':hashlib.sha256(source.read_bytes()).hexdigest(),
        'pins':pins,'catalog_hash':hashlib.sha256(catalog_path.read_bytes()).hexdigest()}
if not db.is_file():
    assert sys.argv[4]=='1', 'Parent app has not initialized SMOKE_ROOT/public.sqlite3'
    result['store']=None
else:
    with connect(db) as c:
        inventory=tables(c)
        names=set(inventory)
        products=[json.loads(row['payload']) for row in c.execute('SELECT * FROM products ORDER BY id')]
        facts=[json.loads(row['payload']) for row in c.execute("SELECT * FROM facts WHERE kind='nav' ORDER BY event_date,observed_at,id")]
        snapshots=[]
        raw_matches={fact['id']:[] for fact in facts}
        for row in c.execute('SELECT * FROM snapshots ORDER BY observed_at,id'):
            item=dict(row)
            payload=json.loads(item.pop('payload'))
            raw=payload if isinstance(payload,list) else []
            points=set()
            for entry in raw:
                if not isinstance(entry,dict):
                    continue
                record=entry.get('record',entry)
                if not isinstance(record,dict):
                    continue
                try:
                    day=record.get('净值日期',record.get('FSRQ',record.get('date')))
                    value=record.get('单位净值',record.get('DWJZ',record.get('unit_nav')))
                    if 'x' in record and 'y' in record:
                        day=datetime.fromtimestamp(float(record['x'])/1000,ZoneInfo('Asia/Shanghai')).date().isoformat()
                        value=record['y']
                    if day is not None and value is not None:
                        number=Decimal(str(value))
                        if number.is_finite() and number>0:
                            points.add((str(day)[:10],number))
                except (ValueError,TypeError,InvalidOperation,OverflowError,OSError):
                    continue
            for fact in facts:
                if (fact['event_date'],Decimal(str(fact['payload']['unit_nav']))) in points:
                    raw_matches[fact['id']].append(item['id'])
            item['warnings']=json.loads(item['warnings'])
            item['raw_records']=len(raw)
            item['raw_nav_points']=len(points)
            item['payload_hash']=digest(payload)
            snapshots.append(item)
        checks=[]
        if 'advice_checks' in names:
            for row in c.execute('SELECT * FROM advice_checks'):
                item=dict(row)
                if isinstance(item.get('payload'),str):
                    item['payload']=json.loads(item['payload'])
                checks.append(item)
        effective={}
        for fact in facts:
            previous=effective.get(fact['event_date'])
            version=lambda item:(datetime.fromisoformat(item['available_at']),datetime.fromisoformat(item['observed_at']),item['id'])
            if previous is None or version(fact)>version(previous):
                effective[fact['event_date']]=fact
        window=[effective[day] for day in sorted(effective)][-61:]
        window_evidence=None
        if len(window)==61:
            base,latest=window[0],window[-1]
            base_value,latest_value=Decimal(str(base['payload']['unit_nav'])),Decimal(str(latest['payload']['unit_nav']))
            with localcontext() as context:
                context.prec=80
                displayed_percent=((latest_value/base_value-1)*100).quantize(Decimal('0.01'))
            window_evidence={'fact_ids':[fact['id'] for fact in window],
                             'base_fact_id':base['id'],'latest_fact_id':latest['id'],
                             'base_date':base['event_date'],'latest_date':latest['event_date'],
                             'base_nav_text':format(base_value,'f'),'latest_nav_text':format(latest_value,'f'),
                             'display_percent':format(displayed_percent,'+.2f')+'%',
                             'endpoint_comparison':'equal' if latest_value==base_value else 'higher' if latest_value>base_value else 'lower',
                             'rounded_nonzero':latest_value!=base_value and displayed_percent==0,
                             'expected_initial_target':'long' if latest_value>base_value else 'cash',
                             'momentum':str(latest_value/base_value-1)}
        local_report=None
        if sys.argv[3]=='1':
            # Bypass both constructors: the reader receives a transactionally
            # consistent, query-only Store; an accidental write fails loudly.
            from rmbfund.domain import now_iso
            from rmbfund.service import Workspace
            from rmbfund.storage import Store
            store=Store.__new__(Store)
            store.path,store.mode,store.conn,store._depth=db,'public',c,0
            workspace=Workspace.__new__(Workspace)
            workspace.root,workspace.mode,workspace.store=work,'public',store
            local_report=workspace.advice('open_fund:018147:C',now_iso(),holding=None)
        result['store']={'hash':digest(inventory),'tables':inventory,
                         'counts':{name:item['rows'] for name,item in inventory.items()},
                         'products':products,'nav_facts':facts,'snapshots':snapshots,
                         'nav_raw_matches':raw_matches,'actual_nav_window':window_evidence,
                         'advice_checks':checks,'report':local_report}
print(json.dumps(result,ensure_ascii=False,default=str))
`, projectRoot, storeRoot, withReport ? '1' : '0', allowMissing ? '1' : '0'],
  { encoding: 'utf8', cwd: projectRoot, maxBuffer: 16 * 1024 * 1024 },
));
const protectedBefore = adviceMode ? adviceAudit(false, true) : null;
await mkdir(output, { recursive: true });
await mkdir(temporary, { recursive: true });
const profile = await mkdtemp(path.join(temporary, 'chrome-smoke-'));
const browser = spawn(chrome, [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run',
  '--disable-background-networking', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--window-size=1440,1100', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let ws;
const pending = new Map();
let seq = 0;
const errors = [];
const browserRequests = [];
try {
  const endpoint = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Chromium did not become ready')), 20000);
    let stderr = '';
    browser.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
      const found = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (found) { clearTimeout(timer); resolve(found[1]); }
    });
    browser.on('exit', (code) => { clearTimeout(timer); reject(new Error(`Chromium exited ${code}: ${stderr}`)); });
    browser.on('error', reject);
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id); pending.delete(message.id); clearTimeout(item.timer);
      message.error ? item.reject(new Error(JSON.stringify(message.error))) : item.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    if (message.method === 'Network.requestWillBeSent') {
      const { request, type } = message.params;
      browserRequests.push({ url:request.url, method:request.method, type });
    }
  };
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 20000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const { targetId } = await call('Target.createTarget', { url });
  const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
  const expressionId = (expression) => createHash('sha256').update(String(expression)).digest('hex').slice(0, 12);
  const errorText = (error) => String(error?.message || error || 'unknown error')
    .replace(/\s+/g, ' ').slice(0, 320);
  const evaluate = async (expression, label = 'runtime evaluation') => {
    const marker = `${label} [expr-${expressionId(expression)}]`;
    try {
      const result = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
      if (result.exceptionDetails) {
        const detail = result.exceptionDetails.exception?.description
          || result.exceptionDetails.text || result.exceptionDetails.exception?.className || 'runtime exception';
        throw new Error(errorText(detail));
      }
      return result.result.value;
    } catch (error) {
      throw new Error(`Browser evaluation failed at ${marker}: ${errorText(error)}`, { cause: error });
    }
  };
  const dispatchClick = async (rectangle) => {
    await call('Input.dispatchMouseEvent', { type:'mouseMoved', ...rectangle }, sessionId);
    await call('Input.dispatchMouseEvent', { type:'mousePressed', button:'left', clickCount:1, ...rectangle }, sessionId);
    await call('Input.dispatchMouseEvent', { type:'mouseReleased', button:'left', clickCount:1, ...rectangle }, sessionId);
  };
  const clickElement = async (expression) => {
    const id = expressionId(expression);
    await evaluate(`(() => { const element=(${expression}); if(!element) throw Error('Missing click target'); element.scrollIntoView({block:'center',behavior:'instant'}); })()`, `scroll click target ${id}`);
    await new Promise(resolve=>setTimeout(resolve,150));
    const rectangle = await evaluate(`(() => { const element=(${expression}); if(!element) throw Error('Missing click target'); const r=element.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; })()`, `measure click target ${id}`);
    await dispatchClick(rectangle);
  };
  const clickVisibleElement = async (expression) => {
    const id = expressionId(expression);
    // BaseWeb menus are portaled overlays. Scrolling the option can close the
    // menu before the second CDP round-trip measures it, so measure and click
    // the currently visible rectangle without a scroll or delay.
    const rectangle = await evaluate(`(() => {
      const element=(${expression});
      if(!element) throw Error('Missing visible click target');
      const r=element.getBoundingClientRect();
      if(!r.width || !r.height) throw Error('Click target is not visible');
      return {x:r.x+r.width/2,y:r.y+r.height/2};
    })()`, `measure visible click target ${id}`);
    await dispatchClick(rectangle);
  };
  await call('Runtime.enable', {}, sessionId);
  await call('Page.enable', {}, sessionId);
  if (adviceMode) await call('Network.enable', {}, sessionId);
  await call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false }, sessionId);
  const waitFor = async (expression, timeout = 40000, label = 'UI condition') => {
    const marker = `${label} [predicate-${expressionId(expression)}]`;
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      // Predicates may end in querySelector or another object. Never ask CDP to
      // serialize live DOM/React graphs just to test whether they are truthy.
      if (await evaluate(`Boolean((${expression}))`, marker)) return;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    await writeFile(path.join(output, 'failure-body.txt'), await evaluate('document.body.innerText', 'capture timeout body'));
    const failed = await call('Page.captureScreenshot', { format:'png' }, sessionId);
    await writeFile(path.join(output, 'failure.png'), Buffer.from(failed.data, 'base64'));
    throw new Error(`UI condition timed out at ${marker}`);
  };
  await waitFor('document.body.innerText.includes("知基")');
  await waitFor('[...document.querySelectorAll("button")].some(x=>x.innerText.includes("明确初始化")) || document.body.innerText.includes("研究日期") || document.querySelectorAll("[role=tab]").length>=4');
  const initialized = await evaluate(`!![...document.querySelectorAll('button')].find(x=>x.innerText.includes('明确初始化'))`);
  if (adviceMode && initialized) throw Error('ADVICE_SMOKE refuses to initialize a DEMO app');
  if (initialized) await clickElement(`[...document.querySelectorAll('button')].find(x=>x.innerText.includes('明确初始化'))`);
  if (initialized) await waitFor('document.querySelectorAll("[role=tab]").length === 7');
  await waitFor('document.querySelectorAll("[role=tab]").length >= 4');
  const settled = async (timeout = 40000) => {
    await new Promise(resolve=>setTimeout(resolve,350));
    await waitFor(`!document.body.innerText.includes('\\nStop\\n') && !document.querySelector('[data-stale="true"]')`, timeout);
  };
  const capture = async (name) => {
    if (adviceMode) await call('Input.dispatchMouseEvent', { type:'mouseMoved', x:0, y:0 }, sessionId);
    await settled();
    const { data } = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false }, sessionId);
    await writeFile(path.join(output, name), Buffer.from(data, 'base64'));
  };
  await waitFor('document.querySelectorAll("[role=tab]").length === 7', 40000, 'all seven tabs rendered');
  await settled();
  if (!adviceMode) await capture('overview.png');
  const labels = await evaluate('[...document.querySelectorAll("[role=tab]")].map(x=>x.innerText.trim())');
  const check = (condition, message) => { if (!condition) throw Error(message); };
  const activeTab = () => evaluate(`document.querySelector('[role=tab][aria-selected=true]')?.innerText.trim()`);
  const clockLabels = ['按今天查看 / 回看过去', '高级：观察时钟与本地目录'];
  const clockButtons = ['切回今天查看', '更新观察时刻为上海当前时间'];
  const select = async (label, option) => {
    await settled();
    const active = await activeTab();
    const candidates = Array.isArray(label) ? label : [label];
    const scope = `[...document.querySelectorAll('[data-testid=stSelectbox]')].find(x=>${JSON.stringify(candidates)}.some(label=>x.innerText.trim().includes(label)))`;
    const selectedExpression = `(() => {
      const scope=(${scope});
      if(!scope) return false;
      const combobox=scope.querySelector('[role=combobox]');
      const control=scope.querySelector('[data-baseweb=select]') || combobox;
      // Streamlit may render the selected display in an input.value, which
      // innerText deliberately omits. Older BaseWeb versions use a text div.
      const values=[combobox?.value,combobox?.getAttribute('aria-valuetext'),
        ...[...scope.querySelectorAll('input')].map(input=>input.value),control?.innerText];
      return values.some(value=>String(value || '').trim().includes(${JSON.stringify(option)}));
    })()`;
    const selected = await evaluate(selectedExpression, 'inspect current select option');
    // Re-selecting the current item can dismiss a BaseWeb menu during a rerun.
    // Only actual changes need an overlay click; later window/scenario changes
    // still exercise the real mouse-driven interaction.
    if (!selected) {
      await clickElement(`(${scope}).querySelector('[role=combobox]')`);
      const optionExpression = `[...document.querySelectorAll('[role=option]')].find(x=>{const r=x.getBoundingClientRect(); return x.innerText.trim().includes(${JSON.stringify(option)}) && r.width>0 && r.height>0})`;
      await waitFor(optionExpression, 40000, 'visible select option');
      await clickVisibleElement(optionExpression);
    }
    await settled();
    await waitFor(`document.querySelector('[role=tab][aria-selected=true]')?.innerText.trim()===${JSON.stringify(active)}`);
    await waitFor(selectedExpression, 40000, 'selected option applied');
  };
  const tab = async (label) => {
    await settled();
    await clickElement(`[...document.querySelectorAll('[role=tab]')].find(x=>x.innerText.trim()===${JSON.stringify(label)})`);
    await waitFor(`[...document.querySelectorAll('[role=tab]')].some(x=>x.innerText.trim()===${JSON.stringify(label)} && x.getAttribute('aria-selected')==='true')`);
    await settled();
  };
  const summaryExpression = (label) => `[...document.querySelectorAll('summary')].find(x=>${JSON.stringify(Array.isArray(label) ? label : [label])}.some(candidate=>x.innerText.trim().includes(candidate)))`;
  const expand = async (label) => { await clickElement(summaryExpression(label)); };
  const setExpanded = async (label, open) => {
    const expression = `${summaryExpression(label)}?.closest('details')?.open`;
    const opened = await evaluate(expression);
    check(typeof opened === 'boolean', `Expander missing: ${label}`);
    if (opened !== open) await expand(label);
    await waitFor(`${expression}===${open}`);
  };
  if (adviceMode) {
    // Keep this real-source acceptance path separate from frozen-study writes,
    // explicit research reproduction, and the legacy demo/browser smoke.
    try {
      const expectedTabs = ['买卖建议', '观察与数据状态', '产品详情', '历史回测', '信息来源与决策清单', '模拟账户 / 订单 / 账本', '日志'];
      check(JSON.stringify(labels) === JSON.stringify(expectedTabs), `Expected seven ordered tabs: ${labels}`);
      check(!initialized, 'Advice smoke must use PUBLIC, not initialize DEMO');
      const protectedState = (audit) => ({public_hash:audit.public_hash, public_file_hash:audit.public_file_hash,
        catalog_hash:audit.catalog_hash, pins:audit.pins});
      const checkProtected = (audit) => {
        check(Object.keys(audit.pins).length === 10, 'Expected all ten pinned research files');
        check(JSON.stringify(protectedState(audit)) === JSON.stringify(protectedState(protectedBefore)),
          'Original data/public.sqlite3, catalog, or pinned evidence changed');
      };
      const checkNoTrading = (audit) => {
        for (const name of ['accounts', 'orders', 'ledger_events', 'journal', 'profiles']) {
          check(audit.store.counts[name] === 0, `Advice acceptance wrote ${name}`);
        }
      };
      const learningTitles = ['这段涨跌为什么不是我的赚亏？', '60 期为什么不是 60 天？', '30 天从哪一天算？',
        '为什么不每天刷新后换一个决定？', '有净值就代表所有信息齐了吗？'];
      const diagnosticLabel = '查看技术诊断（原始数据与错误）';
      const diagnosticInput = `(document.querySelector('.st-key-advice_diagnostics input[type=checkbox]') || [...document.querySelectorAll('[data-testid=stCheckbox]')].find(x=>x.innerText.trim()===${JSON.stringify(diagnosticLabel)})?.querySelector('input'))`;
      const compact = (text) => String(text || '').replace(/\s+/g, ' ');
      const forbiddenDefault = /\b(?:PUBLIC|DEMO|partial|hash|JSON)\b|现金目标|策略版本|回执|哈希|\b[a-f0-9]{64}\b|\.sqlite3|\/home\/|runs\/artifacts\//i;
      const assertAdvicePage = async ({diagnostics = false} = {}) => {
        await waitFor(`document.querySelector('.st-key-advice_card_flat h3') && document.querySelector('.st-key-advice_card_held h3')`, 40000, 'advice card readiness');
        check(await activeTab() === '买卖建议', 'Default/current advice tab not selected');
        const state = await evaluate(`(() => {
          const panel=document.querySelector('[role=tabpanel]:not([hidden])');
          const buttons=[...document.querySelectorAll('button')].filter(x=>x.innerText.trim()==='更新净值并生成买卖建议' && x.getBoundingClientRect().height>0);
          const cards=Object.fromEntries(['flat','held'].map(name=>{
            const card=panel?.querySelector('.st-key-advice_card_'+name);
            return [name,{label:card?.querySelector('h3')?.innerText.trim(),text:card?.innerText,headings:card?.querySelectorAll('h3').length}];
          }));
          const sections=Object.fromEntries(['why','meaning','next','timing','freshness','monitoring','risks'].map(name=>[name,panel?.querySelector('.st-key-advice_'+name)?.innerText || '']));
          const json=[...document.querySelectorAll('[data-testid=stJson]')];
          return {title:document.querySelector('h1')?.innerText.trim(),body:document.body.innerText,allText:document.body.textContent,panel:panel?.innerText,
            root:[...document.querySelectorAll('details')].find(x=>${JSON.stringify(clockLabels)}.some(label=>x.querySelector('summary')?.innerText.trim().includes(label)))?.textContent,
            enabled:buttons.length===1 && !buttons[0].disabled && buttons[0].getAttribute('aria-disabled')!=='true',
            cards,cardLabels:['flat','held'].map(name=>cards[name].label),sections,
            diagnostics:${diagnosticInput}?.checked,jsonNodes:json.length,jsonText:json.map(x=>x.textContent || x.innerText).join('\\n'),
            learning:[...(panel?.querySelectorAll('details') || [])].map(x=>({title:x.querySelector('summary')?.innerText.trim(),open:x.open,text:x.innerText})),
            plots:[...document.querySelectorAll('.js-plotly-plot')].filter(x=>x.getBoundingClientRect().height>0).length,
            frozenControls:[...document.querySelectorAll('[data-testid=stSelectbox]')].some(x=>/研究窗口|到账情景|查看交易 \\/ 信号的路径/.test(x.innerText) && x.getBoundingClientRect().height>0)};
        })()`, 'read keyed advice page snapshot');
        check(state.title === '知基 · 基金决策台', `Unexpected main title: ${state.title}`);
        check(/公开(?:基金)?数据/.test(state.body) && !state.body.includes('当前模式：PUBLIC'), 'Public mode is not identified in ordinary user language');
        check(state.panel?.includes('018147') && state.panel.includes('两种条件')
          && state.cards.flat.text?.includes('如果你还没买') && state.cards.held.text?.includes('如果你已经买了'), 'Advice identity or unknown-holding conditional scope missing');
        check(['flat','held'].every(name=>state.cards[name].headings === 1), 'Each stable advice card must contain one conclusion');
        check(state.enabled, 'The one explicit advice update button is missing or disabled');
        check(!state.plots && !state.frozenControls && !state.panel.includes('固定列出两窗口'), 'Default advice rendered frozen research instead of independent current advice');
        check(!/Traceback|ModuleNotFoundError|区块的本地读取未完成|这部分内容暂时无法显示|暂时读不到已保存/.test(state.body), 'Visible application exception or fallback boundary');
        check(state.diagnostics === diagnostics, 'Diagnostic checkbox is not in the explicitly requested state');
        if (!diagnostics) {
          check(state.jsonNodes === 0, 'Raw st.json was constructed before explicit diagnostics');
          check(!forbiddenDefault.test(state.body), 'Default advice exposes raw state, implementation vocabulary, or a local path');
          check(!state.allText.includes(storeRoot) && !state.root?.includes(storeRoot), 'Local directory was rendered before diagnostics was enabled');
        } else {
          check(state.jsonNodes > 0, 'Explicit diagnostics did not render inspectable raw JSON');
        }
        return state;
      };
      // This is a presentation contract for authoritative service actions, not a
      // second trading rule. A conditional held WAIT must never become SELL.
      const expectedCard = (report, state) => {
        if (report?.status !== 'ready' || report.receipt_state !== 'valid') return {action:'wait',label:'等待'};
        const action = report.branches?.[state]?.action;
        if (action === 'buy') return {action,label:'可考虑买入，先核对条件'};
        if (action === 'sell') return {action,label:'可考虑卖出，先核对渠道'};
        if (action === 'hold') return {action,label:state === 'held' ? '继续持有，先不追加' : '现在先别买'};
        if (action === 'wait') return {action,label:report.receipt?.target === 'cash'
          ? (state === 'flat' ? '现在先别买' : '先核对卖出条件') : '先等待，暂不操作'};
        return {action:'wait',label:'等待'};
      };
      const assertCardActions = (page, report) => {
        const expected = Object.fromEntries(['flat','held'].map(state=>[state,expectedCard(report,state)]));
        for (const state of ['flat','held']) {
          check(page.cards[state].label === expected[state].label,
            `${state} card contradicts authorized ${expected[state].action}: ${page.cards[state].label}`);
        }
        if (report?.status === 'ready' && report.receipt_state === 'valid' && expected.held.action === 'wait') {
          check(page.cards.held.text.includes('不是已获准卖出'), 'Unknown-held WAIT was presented as sell authorization');
        }
        return expected;
      };
      const shanghaiTime = (value) => {
        const parts = Object.fromEntries(new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Shanghai',
          year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'})
          .formatToParts(new Date(value)).map(part=>[part.type,part.value]));
        return {day:`${parts.year}-${parts.month}-${parts.day}`,time:`${parts.hour}:${parts.minute}`};
      };
      const assertComprehension = (page, report = null, actualWindow = null) => {
        const sections = Object.fromEntries(Object.entries(page.sections).map(([key,text])=>[key,compact(text)]));
        for (const name of ['why','meaning','next','timing','freshness','risks']) {
          check(sections[name].length > 20, `Novice explanation section missing or empty: ${name}`);
        }
        check(sections.meaning.includes('单位净值') && sections.meaning.includes('一份基金')
          && sections.meaning.includes('账面价值') && /不是.*账户余额.*个人盈亏/.test(sections.meaning)
          && /不等于.*分红.*总收益/.test(sections.meaning), 'NAV comparison is not distinguished from the user’s account balance, personal P&L, and total return');
        check(/支付宝.*(?:交易|份额确认).*确认日/.test(sections.next) && sections.next.includes('不是购买申请日')
          && /自确认日起满\s*30\s*个自然日/.test(sections.next) && /(?:模型|本方法).*条件/.test(sections.next)
          && /不是基金官方.*必须持有/.test(sections.next) && /必须卖出.*免手续费/.test(sections.next),
          'Next steps do not explain confirmation records and the model-only 30-day condition, distinct from official holding/fee rules');
        check(/限购.*暂停申赎.*费用/.test(sections.next) && /不下单.*不转账/.test(sections.next), 'Novice next steps omit channel checks or no-trading boundary');
        const unknownOpportunity = !report?.receipt && (report?.read_error === true
          || (report?.reasons || []).some(reason=>String(reason).includes('读不到')
            || reason === '这只基金有重复记录，需要先核对，不能给出买卖建议。'));
        const currentMonth = report?.evaluated_at ? shanghaiTime(report.evaluated_at).day.slice(0, 7) : null;
        if (unknownOpportunity) {
          check(/无法确认已保存的月度检查及本月检查机会/.test(sections.timing)
            && /不能假定本月机会尚未使用/.test(sections.timing), 'Unreadable or ambiguous records are incorrectly presented as an unused monthly opportunity');
        } else if (report?.receipt?.check_month && report.receipt.check_month === currentMonth) {
          check(/本月.*首次有效检查已经完成/.test(sections.timing)
            && /不会新建、重启月度检查或补发新的本月操作建议/.test(sections.timing)
            && sections.timing.includes('下月'), 'Saved monthly advice does not distinguish a reused original judgment from a new monthly check/action');
        } else {
          check(/本月.*(?:尚无有效检查|首次有效检查)/.test(sections.timing)
            && /可以现在尝试主动更新|稍后重新读取|等待/.test(sections.timing + sections.next),
            'Unavailable advice does not explain when a first valid monthly check can be attempted');
        }
        check(/净值更新、买入确认、卖出到账不是同一时间/.test(sections.timing) && sections.timing.includes('不是实时行情'),
          'QDII disclosure/confirmation/cash-arrival distinction missing');
        check(/净值所属日.*最近一次成功来源核查/.test(sections.freshness)
          && sections.freshness.includes('核查日期不是净值所属日') && /24\s*小时/.test(sections.freshness)
          && /7\s*个自然日/.test(sections.freshness)
          && /(?:不等于操作建议.*一直有效|建议是否过期另看检查日.*不能混为一谈)/.test(sections.freshness),
          'Data date, successful source check, freshness thresholds, and advice validity are conflated');
        check(/可能错过反弹/.test(sections.risks) && /买入后继续下跌/.test(sections.risks)
          && /不能.*保证/.test(sections.risks) && /历史.*有好有坏/.test(sections.risks)
          && /不能承诺未来收益.*准确率/.test(sections.risks), 'Limitations hide missed rebounds, further losses, mixed historical evidence, or lack of guarantees');
        const conclusions = assertCardActions(page, report);
        if (report?.status === 'ready' && report.receipt_state === 'valid') {
          const basis = report.receipt?.monitor;
          check(basis && actualWindow, 'Ready explanation lacks immutable monthly NAV evidence');
          check(sections.why.includes(basis.base_date) && sections.why.includes(basis.latest_date)
            && sections.why.includes(actualWindow.base_nav_text) && sections.why.includes(actualWindow.latest_nav_text),
            'Why section does not connect the actual monthly comparison dates and NAV values');
          if (actualWindow.rounded_nonzero) {
            check(sections.why.includes('四舍五入') && sections.why.includes('并非相等')
              && sections.why.includes(actualWindow.endpoint_comparison === 'higher' ? '略高' : '略低'), 'Rounded 0.00% hides the actual nonzero movement');
          } else if (actualWindow.endpoint_comparison === 'equal') {
            check(sections.why.includes('两端精确相等') && /沿用|首次检查.*默认/.test(sections.why), 'Exact equality invents a new directional signal');
          } else {
            check(sections.why.includes(actualWindow.display_percent)
              && sections.why.includes(actualWindow.endpoint_comparison === 'higher' ? '高于' : '低于')
              && /走强.*走弱/.test(sections.why), 'Why section omits the actual endpoint change and the model’s reasoning');
          }
          const expiry = shanghaiTime(report.receipt.expires_at);
          const checked = shanghaiTime(report.evidence.snapshot.observed_at);
          check(sections.timing.includes(expiry.day) && sections.timing.includes(expiry.time)
            && /仅限检查当日.*提前失效/.test(sections.timing) && /不会.*补发新的本月操作建议/.test(sections.timing)
            && sections.timing.includes('下月'), 'Current advice expiry and next-month-only action opportunity are unclear');
          check(sections.freshness.includes(report.monitor.latest_date) && sections.freshness.includes(checked.day)
            && sections.freshness.includes(checked.time), 'Actual NAV date and successful source-check time are not both visible');
          if (report.receipt.held_eligibility === false) {
            check(/首次检查时卖出条件未核实通过/.test(page.panel) && /后来补填日期不会补发本月卖出建议/.test(page.panel),
              'Filling dates later is misleadingly presented as unlocking this month’s sell');
          }
        } else {
          check(/有效|净值|核查|失效|更新|读取/.test(sections.why) && /核查|核对|等待|更新/.test(sections.next),
            'Unavailable advice does not explain a concrete blocker and next step');
          const evidence = report?.evidence || {};
          const freshness = evidence.freshness || {};
          if (Array.isArray(evidence.navs)) {
            const count = new Set(evidence.navs.filter(nav=>Number(nav.unit_nav) > 0).map(nav=>nav.event_date)).size;
            if (count < 61) check(sections.why.includes(`只有 ${count} 个不同日期`) && sections.why.includes('需要 61 个'), 'Insufficient-data explanation hides the actual missing observation count');
          }
          if (Number(freshness.snapshot_age_hours) > 24) check(/成功来源核查已超过\s*24\s*小时/.test(sections.why), 'Stale-source WAIT omits the actual source-check blocker');
          if (Number(freshness.economic_nav_age_days) > 7) check(/净值所属日.*超过\s*7\s*个自然日/.test(sections.why), 'Stale-NAV WAIT hides why checking today cannot make an old NAV current');
          if (evidence.nav_conflicts?.length || evidence.raw_nav_conflicts?.length) check(sections.why.includes('净值记录不一致'), 'Source-conflict WAIT hides the discrepancy');
          if (Number(evidence.unit_money?.nonempty) > 0 || evidence.corporate_events?.length || evidence.manual_unverified_actions?.length) {
            check(/分红.*拆分/.test(sections.why) && /核实|核对|复核/.test(sections.next), 'Corporate-event WAIT hides required verification');
          }
          if (report?.receipt_state === 'expired') {
            const saved = report.receipt || {};
            const evaluated = Date.parse(report.evaluated_at);
            const generated = Date.parse(saved.generated_at);
            const expiry = Date.parse(saved.expires_at);
            const haveDays = Number.isFinite(evaluated) && Number.isFinite(generated);
            const sameDay = haveDays && shanghaiTime(report.evaluated_at).day === shanghaiTime(saved.generated_at).day;
            const beforeDeadline = sameDay && generated <= evaluated && Number.isFinite(expiry) && evaluated < expiry;
            const sameBasis = Boolean(evidence.basis_hash && saved.basis_hash && evidence.basis_hash === saved.basis_hash);
            const changedBasis = Boolean(evidence.basis_hash && saved.basis_hash && evidence.basis_hash !== saved.basis_hash);
            if (beforeDeadline && sameBasis) {
              check(/原判断当前不满足有效条件/.test(sections.why)
                && /尚在原检查日截止前/.test(sections.timing)
                && /若仅来源核查超过\s*24\s*小时/.test(sections.timing)
                && /重新成功核查同一依据.*其他条件仍满足/.test(sections.timing)
                && /恢复同一原判断.*不是生成新建议/.test(sections.timing),
                'Same-day stale-source WAIT incorrectly forbids conditional restoration of the same original judgment or creates a new action');
            } else if (beforeDeadline && changedBasis) {
              check(/上次操作建议.*失效/.test(sections.why)
                && /比较依据已变.*不能将新依据套进原判断或补发本月建议/.test(sections.timing),
                'Changed-basis WAIT promises to restore an incompatible original judgment');
            } else if (haveDays && (!sameDay || Number.isFinite(expiry) && evaluated >= expiry)) {
              check(/上次操作建议.*失效/.test(sections.why)
                && /已经过了原检查日或有效截止/.test(sections.timing)
                && /即使重新核查到相同数据也不能恢复原建议/.test(sections.timing),
                'Cross-day or deadline-expired WAIT is misleadingly treated as recoverable through a data refresh');
            } else {
              check(/原判断当前不满足有效条件/.test(sections.why)
                && /检查时间或依据未完整提供.*不能承诺刷新后恢复/.test(sections.timing),
                'Incomplete expiry evidence promises restoration of an unverifiable old judgment');
            }
          }
        }
        return {conclusions,why:sections.why,not_personal_pnl:true,model_30day_not_official:true,
          data_date_distinct_from_source_check:true,monthly_and_expiry_explained:true,limitations_explained:true};
      };
      const assertNoExternalBrowserRequests = (start) => {
        const external = browserRequests.slice(start).filter(request=>{
          const target = new URL(request.url);
          return ['http:','https:','ws:','wss:'].includes(target.protocol) && target.host !== new URL(url).host;
        });
        check(external.length === 0, `Passive explanation/diagnostics requested an external browser URL: ${JSON.stringify(external)}`);
      };
      const assertPassive = (baseline, audit, start, context) => {
        checkProtected(audit);
        checkNoTrading(audit);
        check(audit.store.hash === baseline.store.hash, `${context} wrote rows, captured a source, or generated a monthly check`);
        assertNoExternalBrowserRequests(start);
      };
      const captureAdvice = async (name) => {
        await evaluate(`document.querySelector('[role=tabpanel]:not([hidden])').scrollIntoView({block:'start'})`);
        await capture(name);
      };
      const captureAdviceLower = async (name) => {
        await evaluate(`document.querySelector('.st-key-advice_timing').scrollIntoView({block:'start',behavior:'instant'})`, 'scroll advice timing and source evidence');
        await capture(name);
      };
      const exerciseLearning = async (baseline) => {
        const start = browserRequests.length;
        const initial = await assertAdvicePage();
        const lessons = initial.learning.filter(item=>learningTitles.some(title=>item.title.includes(title)));
        check(lessons.length === learningTitles.length && lessons.every(item=>!item.open), 'Further learning is absent or expanded without an explicit click');
        for (const title of learningTitles) {
          await setExpanded(title, true);
          const opened = await assertAdvicePage();
          const active = opened.learning.filter(item=>learningTitles.some(label=>item.title.includes(label)) && item.open);
          check(active.length === 1 && active[0].title.includes(title), 'Opening one lesson implicitly expanded another');
          if (title.startsWith('60 ')) {
            check(/61\s*个不同日期.*60\s*个观察间隔/.test(compact(active[0].text))
              && /不是\s*60\s*个自然日/.test(active[0].text) && /不固定等于三个月/.test(active[0].text), '60 observations are misleadingly explained as calendar days');
            await evaluate(`${summaryExpression(title)}.scrollIntoView({block:'start',behavior:'instant'})`);
            await capture('advice-learning.png');
          }
          if (title.startsWith('这段涨跌')) {
            check(/买入成本、份额、分红和费用/.test(active[0].text) && /没有读取这些账户记录/.test(active[0].text), 'Optional lesson does not explain why fund change is not the user’s P&L');
          }
          if (title.startsWith('30 ')) {
            check(/从份额确认日算/.test(active[0].text) && /不是官方禁售期/.test(active[0].text)
              && /不能靠后来补填日期在同月解锁/.test(active[0].text), '30-day lesson promises official restrictions or unlocks frozen sell eligibility');
          }
          if (title.startsWith('为什么不每天')) {
            const lesson = compact(active[0].text);
            check(/上海自然月/.test(lesson) && /不是涨一天就追买/.test(lesson)
              && /本月确实没有检查，可以现在尝试核查/.test(lesson)
              && /同一检查日截止前.*成功核查同一依据.*其他条件满足时可恢复原判断/.test(lesson)
              && /跨日、超过截止或依据改变，不能靠刷新恢复.*不会新发本月建议/.test(lesson)
              && /记录读不到或基金身份不确定.*不能假定本月机会未使用/.test(lesson),
              'Monthly lesson conflates a new check, conditional same-day restoration, irreversible expiry, or unknown opportunity');
          }
          if (title.startsWith('有净值')) {
            check(/不等于官方分红、拆分公告及渠道交易限制已全部核实/.test(active[0].text)
              && /没有提示事件.*不证明事件不存在/.test(compact(active[0].text)), 'Optional lesson incorrectly treats source completeness as official verification');
          }
          await setExpanded(title, false);
        }
        assertPassive(baseline, adviceAudit(), start, 'Explicit learning expansion/collapse');
      };
      const setDiagnostics = async (open) => {
        const checked = await evaluate(`${diagnosticInput}?.checked`);
        check(typeof checked === 'boolean', 'Explicit diagnostics checkbox missing');
        if (checked !== open) await clickElement(`(${diagnosticInput}).closest('[data-testid=stCheckbox]')?.querySelector('label') || ${diagnosticInput}`);
        await settled();
        await waitFor(`${diagnosticInput}?.checked===${open} && document.querySelectorAll('[data-testid=stJson]').length${open ? '>0' : '===0'}`);
      };
      const exerciseDiagnostics = async (baseline, report = null) => {
        const start = browserRequests.length;
        check((await assertAdvicePage()).jsonNodes === 0, 'Raw JSON exists before diagnostic opt-in');
        await setDiagnostics(true);
        await setExpanded(clockLabels, true);
        const diagnostic = await assertAdvicePage({diagnostics:true});
        check(diagnostic.root?.includes(storeRoot) && diagnostic.body.includes(storeRoot), 'Explicit diagnostics does not identify the parent’s isolated SMOKE_ROOT');
        check(diagnostic.jsonText.includes('status') && diagnostic.jsonText.includes('receipt_state'), 'Diagnostics does not expose original report state');
        if (report) {
          const values = [report.status,report.receipt_state];
          const receipt = report.receipt;
          const snapshot = report.evidence?.snapshot;
          const monitor = receipt?.monitor || report.monitor;
          if (receipt) values.push(receipt.id,receipt.policy_hash,receipt.basis_hash);
          if (snapshot) values.push(snapshot.id,snapshot.source_url,snapshot.content_hash);
          if (monitor) values.push(monitor.base_date,monitor.latest_date,String(monitor.base_nav),String(monitor.latest_nav));
          for (const value of values) {
            check(value && diagnostic.jsonText.includes(value), `Diagnostic JSON cannot substantiate actual evidence: ${value}`);
          }
        }
        await writeFile(path.join(output, 'advice-diagnostics-body.txt'), diagnostic.body);
        await evaluate(`document.querySelector('[data-testid=stJson]').scrollIntoView({block:'start'})`);
        await capture('advice-diagnostics.png');
        await setExpanded(clockLabels, false);
        await setDiagnostics(false);
        const returned = await assertAdvicePage();
        check(!returned.allText.includes(storeRoot), 'Turning diagnostics off leaves a local directory rendered');
        assertPassive(baseline, adviceAudit(), start, 'Explicit diagnostic opt-in/out');
        return {explicit_opt_in:true,raw_json_absent_before_and_after:true,isolated_root_verified:true,
          actual_evidence_inspectable:Boolean(report?.evidence?.snapshot && (report.receipt?.monitor || report.monitor))};
      };
      const rerunAdviceClock = async () => {
        await setExpanded(clockLabels, true);
        await clickElement(`[...document.querySelectorAll('button')].find(x=>${JSON.stringify(clockButtons)}.includes(x.innerText.trim()))`);
        await settled();
        await setExpanded(clockLabels, false);
        await assertAdvicePage();
      };
      await settled();
      const defaultPage = await assertAdvicePage();
      // Workspace has now initialized its local schema. A new empty monthly
      // table is legitimate; rendering an implicit monthly check is not.
      const before = adviceAudit(true);
      checkProtected(before);
      checkNoTrading(before);
      for (const name of ['products', 'facts', 'snapshots', 'advice_checks']) {
        check((before.store.counts[name] ?? 0) === 0, `ADVICE_SMOKE requires a fresh isolated Store: ${name}`);
      }
      const defaultComprehension = assertComprehension(defaultPage, before.store.report);
      check(JSON.stringify(defaultPage.cardLabels) === JSON.stringify(['等待','等待'])
        && !defaultPage.panel.includes('建议检查完成') && /还没有.*有效月度判断/.test(compact(defaultPage.sections.why)),
        'Unrefreshed default advice fabricated readiness or omitted why it cannot advise');
      check(defaultPage.learning.filter(item=>learningTitles.some(title=>item.title.includes(title))).every(item=>!item.open),
        'Default novice learning was expanded without an explicit request');
      const passiveRequestsStart = browserRequests.length;
      await writeFile(path.join(output, 'before.json'), JSON.stringify(before, null, 2));
      await writeFile(path.join(output, 'advice-default-body.txt'), defaultPage.body);
      const initialReplays = (await readdir(artifactsRoot)).filter(name=>name.startsWith('ui-replay-')).sort();
      await evaluate('document.querySelector("h1").scrollIntoView({block:"start"})');
      await capture('overview.png');
      await captureAdvice('advice-default-desktop.png');
      await call('Emulation.setDeviceMetricsOverride', {width:800,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
      assertComprehension(await assertAdvicePage(), before.store.report);
      await captureAdvice('advice-default-narrow.png');
      await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
      await exerciseLearning(before);
      const defaultDiagnostics = await exerciseDiagnostics(before, before.store.report);
      await rerunAdviceClock();
      const passive = adviceAudit();
      assertPassive(before, passive, passiveRequestsStart, 'Default display/resize/learning/diagnostics/clock rerun');
      await writeFile(path.join(output, 'passive-readonly.json'), JSON.stringify(passive, null, 2));

      // Exactly one click. The already-running app, not this script, fetches
      // actual sources. An invalid/stale source must remain WAIT with no check.
      const updateStarted = new Date().toISOString();
      await clickElement(`[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='更新净值并生成买卖建议')`);
      await waitFor(`['建议检查完成','本次检查已返回','本次更新未完成','操作未完成','操作被阻塞','暂时还不能给出新的建议','这次更新没有完成','更新未完成'].some(text=>document.body.innerText.includes(text))`, 180000);
      await settled(180000);
      const updatedPage = await assertAdvicePage();
      const afterUpdate = adviceAudit(true);
      checkProtected(afterUpdate);
      checkNoTrading(afterUpdate);
      const store = afterUpdate.store;
      const report = store.report;
      check(store.counts.products === 1 && store.products[0]?.id === 'open_fund:018147:C'
        && store.products[0].symbol === '018147' && !store.products[0].is_demo,
        'Explicit update did not register exactly the real 018147 C subject');
      check(report && report.product?.id === 'open_fund:018147:C', 'Read-only current advice report has wrong identity');
      check(store.snapshots.length > 0, 'Explicit update has no actual source-attempt snapshot');
      const started = Date.parse(updateStarted) - 2000;
      const finished = Date.now() + 2000;
      for (const snapshot of store.snapshots) {
        check(snapshot.product_id === 'open_fund:018147:C' && /^https?:\/\//.test(snapshot.source_url)
          && !/demo|fixture|mock|research/i.test(snapshot.source), 'Snapshot is not an actual public source attempt');
        const observed = Date.parse(snapshot.observed_at);
        check(observed >= started && observed <= finished, 'Snapshot was not observed during this explicit update');
        check(snapshot.content_hash === snapshot.payload_hash, 'Stored source payload hash does not match snapshot receipt');
      }
      for (const fact of store.nav_facts) {
        check(fact.product_id === 'open_fund:018147:C' && fact.kind === 'nav' && Number(fact.payload.unit_nav) > 0
          && /^https?:\/\//.test(fact.source_url) && !/demo|fixture|mock|research/i.test(`${fact.source} ${fact.quality}`),
          'Stored NAV is invalid or fabricated');
        check(Date.parse(fact.observed_at) >= started && Date.parse(fact.observed_at) <= finished
          && Date.parse(fact.available_at) >= started, 'NAV observation/availability was backdated');
        check(store.nav_raw_matches[fact.id]?.length > 0, `NAV has no matching actually fetched raw record: ${fact.id}`);
      }
      const ready = report.status === 'ready' && report.receipt_state === 'valid';
      const receipt = report.receipt;
      const checkCount = store.counts.advice_checks ?? 0;
      const navById = new Map(store.nav_facts.map(fact=>[fact.id, fact]));
      const actualWindow = store.actual_nav_window;
      if (actualWindow) {
        check(report.monitor?.latest_fact_id === actualWindow.latest_fact_id
          && report.monitor.base_fact_id === actualWindow.base_fact_id,
          'Current monitor does not use the newest 61 distinct actual NAV observations');
        check(Math.abs(Number(report.monitor.momentum) - Number(actualWindow.momentum)) < 1e-10,
          'Current monitor differs from independent exact NAV endpoint calculation');
      }
      check(!report.evidence?.usable || ready, 'Usable actual source failed to create its first monthly advice check');
      const updatedComprehension = assertComprehension(updatedPage, report, actualWindow);
      const expectedCards = ['flat','held'].map(state=>expectedCard(report,state).label);
      check(JSON.stringify(updatedPage.cardLabels) === JSON.stringify(expectedCards), 'Rendered novice advice cards differ from authoritative report actions');
      let navEvidence = null;
      if (ready) {
        check(checkCount === 1 && store.advice_checks.length === 1 && receipt?.id, 'Valid first update needs exactly one persisted monthly check');
        check(store.advice_checks[0].id === receipt.id, 'Displayed advice receipt is not the persisted check');
        const monitor = report.monitor;
        check(monitor?.observation_count === 61 && monitor.lookback_observations === 60, 'Ready report lacks 61 actual observations');
        const latest = navById.get(monitor.latest_fact_id);
        const base = navById.get(monitor.base_fact_id);
        check(latest && base && latest.event_date === monitor.latest_date && base.event_date === monitor.base_date
          && Number(latest.payload.unit_nav) === Number(monitor.latest_nav)
          && Number(base.payload.unit_nav) === Number(monitor.base_nav), 'Advice NAV endpoints are not actual fetched facts');
        check(Math.abs(Number(monitor.momentum) - (Number(monitor.latest_nav) / Number(monitor.base_nav) - 1)) < 1e-10,
          'Advice momentum differs from actual unit NAV endpoints');
        check(actualWindow && monitor.target === actualWindow.expected_initial_target
          && receipt.target === actualWindow.expected_initial_target,
          'First monthly target differs from the exact actual 60-observation rule');
        const persisted = store.advice_checks[0];
        check(JSON.stringify(persisted.payload) === JSON.stringify(receipt), 'Current receipt differs from the immutable stored payload');
        check(receipt.product_id === report.product.id && receipt.policy_hash === report.policy.hash
          && persisted.policy_hash === receipt.policy_hash && persisted.basis_hash === receipt.basis_hash
          && receipt.basis_hash === report.evidence?.basis_hash, 'Advice receipt identity/policy/input basis is not coherent');
        check(Date.parse(receipt.generated_at) >= started && Date.parse(receipt.generated_at) <= finished
          && Date.parse(receipt.expires_at) > Date.parse(receipt.generated_at), 'Advice receipt was backdated or already expired');
        check(!/"(?:holding|all_confirmed|latest_confirmation_date)"\s*:/.test(JSON.stringify(receipt)),
          'Default advice persisted raw session-only holding state');
        check(receipt.input_endpoint?.latest?.fact_id === latest.id && receipt.input_endpoint?.base?.fact_id === base.id,
          'Persisted receipt endpoints differ from actual fetched NAV');
        const monthlyMonitor = receipt.monitor;
        check(monthlyMonitor?.latest_fact_id === latest.id && monthlyMonitor.base_fact_id === base.id
          && monthlyMonitor.latest_date === latest.event_date && monthlyMonitor.base_date === base.event_date
          && Number(monthlyMonitor.latest_nav) === Number(latest.payload.unit_nav)
          && Number(monthlyMonitor.base_nav) === Number(base.payload.unit_nav),
          'Immutable monthly comparison does not substantiate the displayed novice explanation');
        const snapshot = store.snapshots.find(item=>item.id === report.evidence?.snapshot?.id);
        check(snapshot && snapshot.content_hash === report.evidence.snapshot.content_hash
          && store.nav_raw_matches[latest.id].includes(snapshot.id) && store.nav_raw_matches[base.id].includes(snapshot.id),
          'Advice evidence snapshot does not substantiate the actual NAV endpoints');
        check(receipt.evidence?.snapshot?.id === snapshot.id && receipt.evidence?.basis_hash === receipt.basis_hash,
          'Monthly receipt lost actual source evidence');
        check(report.evidence.navs?.length === 61 && report.evidence.navs.every((nav, index)=>nav.fact_id === actualWindow.fact_ids[index]),
          'Advice receipt evidence is not the independently selected 61-observation window');
        check(receipt.check_month === report.evaluated_at.slice(0, 7), 'Receipt month differs from current advice clock');
        check(updatedPage.panel.includes(monitor.latest_date) && updatedPage.panel.includes(monitor.base_date)
          && updatedPage.panel.includes(String(monitor.latest_nav)) && updatedPage.panel.includes(String(monitor.base_nav)),
          'Actual NAV endpoints are not rendered on advice page');
        check(updatedPage.panel.includes('建议检查完成'), 'Valid source update lacks explicit completion feedback');
        navEvidence = {latest, base, latest_snapshots:store.nav_raw_matches[latest.id], base_snapshots:store.nav_raw_matches[base.id]};
      } else {
        check(checkCount === 0 && store.advice_checks.length === 0, 'Invalid source created a monthly check');
        check(report.receipt_state !== 'valid' && report.status !== 'ready', 'Invalid source is presented as ready');
        const branches = Object.values(report.branches || {});
        const reasons = [...(report.reasons || []), ...branches.flatMap(branch=>branch.reasons || [])];
        check(reasons.some(reason=>String(reason).trim()), 'WAIT report has no explicit reason');
        check(branches.length === 2 && branches.every(branch=>branch.action === 'wait'), 'Invalid source offers a buy/sell instruction');
        check(updatedPage.cardLabels.every(label=>label === '等待') && !updatedPage.panel.includes('建议检查完成')
          && compact(updatedPage.sections.why).length > 20 && compact(updatedPage.sections.next).length > 20,
          'Invalid source WAIT does not show understandable reasons and next steps');
      }
      await writeFile(path.join(output, 'after-update.json'), JSON.stringify(afterUpdate, null, 2));
      await writeFile(path.join(output, 'advice-updated-body.txt'), updatedPage.body);
      await captureAdvice('advice-updated-desktop.png');
      await captureAdviceLower('advice-updated-desktop-lower.png');
      const updatedPassiveStart = browserRequests.length;
      await call('Emulation.setDeviceMetricsOverride', {width:800,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
      assertComprehension(await assertAdvicePage(), report, actualWindow);
      await captureAdvice('advice-updated-narrow.png');
      await captureAdviceLower('advice-updated-narrow-lower.png');
      await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
      await exerciseLearning(afterUpdate);
      const updatedDiagnostics = await exerciseDiagnostics(afterUpdate, report);
      await rerunAdviceClock();
      const rerun = adviceAudit(true);
      assertPassive(afterUpdate, rerun, updatedPassiveStart, 'Updated advice resize/learning/diagnostics/clock rerun');
      check((rerun.store.counts.advice_checks ?? 0) === checkCount, 'Passive rerun duplicated the monthly check');

      await tab('历史回测');
      await waitFor('document.body.innerText.includes("固定列出两窗口")');
      await select('研究窗口（不改变观察时钟）', '较早');
      await select('到账情景（不是已核交易规则）', '10');
      await select('查看交易 / 信号的路径', '月度动量');
      await capture('history-earlier-cash10.png');
      await select('研究窗口（不改变观察时钟）', '较近');
      await select('到账情景（不是已核交易规则）', '7');
      await select('查看交易 / 信号的路径', '买入持有');
      await capture('history-later-cash7.png');
      await tab('信息来源与决策清单');
      await waitFor('document.body.innerText.includes("研究来源 · 假设与缺口")');
      await capture('sources.png');
      await tab('模拟账户 / 订单 / 账本');
      await waitFor('document.body.innerText.includes("严格前向模拟未就绪")');
      await expand('新建模拟账户');
      await settled();
      const blocked = await evaluate(`(() => { const button=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='创建模拟账户'); return !!button?.disabled && ![...document.querySelectorAll('button')].some(x=>x.innerText.trim()==='确认提交模拟委托'); })()`);
      check(blocked, 'QDII forward trading unexpectedly enabled');
      await capture('forward-blocked.png');
      await tab('买卖建议');
      const returnedPage = await assertAdvicePage();
      assertComprehension(returnedPage, report, actualWindow);
      check(JSON.stringify(returnedPage.cardLabels) === JSON.stringify(expectedCards), 'Historical navigation changed the rendered current advice');
      const after = adviceAudit(true);
      checkProtected(after);
      checkNoTrading(after);
      check(after.store.hash === store.hash && (after.store.counts.advice_checks ?? 0) === checkCount,
        'Historical window/cash/path/source/paper navigation changed current advice data or duplicated the check');
      check(after.store.report.receipt?.id === report.receipt?.id && after.store.report.receipt_state === report.receipt_state,
        'Historical navigation changed current advice receipt');
      check(JSON.stringify((await readdir(artifactsRoot)).filter(name=>name.startsWith('ui-replay-')).sort()) === JSON.stringify(initialReplays),
        'Passive historical navigation created a replay');
      await captureAdvice('advice-after-navigation.png');
      await writeFile(path.join(output, 'after.json'), JSON.stringify(after, null, 2));
      const summary = {url, mode:'integrated-current-advice', smoke_root:storeRoot, tabs:labels,
        default_advice_independent:true, default_passive_readonly:true, explicit_update_clicks:1,
        comprehension:{default:defaultComprehension,updated:updatedComprehension,desktop_and_800px:true},
        learning_explicit_only:true, diagnostic_gate:{default:defaultDiagnostics,updated:updatedDiagnostics},
        passive_no_source_capture_or_check:true, passive_no_external_browser_requests:true,
        outcome:ready ? 'ready' : 'WAIT', current_advice:report, nav_evidence:navEvidence,
        source_snapshots:store.snapshots, advice_checks:store.advice_checks,
        before:{hash:before.store.hash,counts:before.store.counts}, after:{hash:after.store.hash,counts:after.store.counts},
        original_public_unchanged:true, pinned_evidence_unchanged:true, pinned_files:Object.keys(after.pins).length,
        rerun_navigation_readonly:true, no_trading_writes:true, forward_blocked:blocked, errors,
        screenshots:['overview.png','advice-default-desktop.png','advice-default-narrow.png','advice-updated-desktop.png',
          'advice-updated-desktop-lower.png','advice-updated-narrow.png','advice-updated-narrow-lower.png',
          'advice-learning.png','advice-diagnostics.png','history-earlier-cash10.png',
          'history-later-cash7.png','sources.png','forward-blocked.png','advice-after-navigation.png'],
        warning:'Real-source browser acceptance verifies honest current advice and audit evidence, not investment performance.'};
      await writeFile(path.join(output, 'report.json'), JSON.stringify(summary, null, 2));
      check(!errors.length, `Browser runtime errors: ${JSON.stringify(errors)}`);
      console.log(JSON.stringify(summary, null, 2));
    } catch (error) {
      await writeFile(path.join(output, 'failure-body.txt'), await evaluate('document.body.innerText'));
      const { data } = await call('Page.captureScreenshot', {format:'png'}, sessionId);
      await writeFile(path.join(output, 'failure.png'), Buffer.from(data, 'base64'));
      throw error;
    }
  } else if (researchMode) {
    try {
    const storeRoot = path.resolve(process.env.SMOKE_ROOT || '');
    const relativeRoot = path.relative(path.resolve('runs/artifacts'), storeRoot);
    if (!process.env.SMOKE_ROOT || !relativeRoot || relativeRoot.startsWith('..') || path.isAbsolute(relativeRoot)) {
      throw Error('RESEARCH_SMOKE requires an isolated SMOKE_ROOT under runs/artifacts');
    }
    const check = (condition, message) => { if (!condition) throw Error(message); };
    const audit = () => JSON.parse(execFileSync(path.resolve('.venv/bin/python'), ['-I', '-c', `
import hashlib,json,sqlite3,sys
from pathlib import Path
from rmbfund.research_replay import public_snapshot
root=Path.cwd()
catalog=json.loads((root/'configs/research-catalog.json').read_text())
files={}
for study in catalog['studies']:
    pins=[study['source'],*study.get('evidence',[])]
    pins += [w[k] for w in study['windows'] for k in ('inputs','report','manifest')]
    for pin in pins:
        p=root/pin['path']
        files[pin['path']]=hashlib.sha256(p.read_bytes()).hexdigest()
store=public_snapshot(Path(sys.argv[1])/'public.sqlite3')
with sqlite3.connect((Path(sys.argv[1])/'public.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as c:
    c.row_factory=sqlite3.Row
    notes=[dict(row) for row in c.execute('SELECT * FROM journal')]
source=public_snapshot(root/'data/public.sqlite3')
print(json.dumps({'store_hash':store['hash'],'counts':{k:v['rows'] for k,v in store['tables'].items()},
                  'notes':notes,'public_hash':source['hash'],'pins':files},ensure_ascii=False))
`, storeRoot], { encoding: 'utf8' }));
    const fill = async (label, value) => {
      await evaluate(`(() => {
        const marker=[...document.querySelectorAll('label')].find(x=>x.innerText.includes(${JSON.stringify(label)}));
        let scope=marker;
        let input;
        for(let i=0; i<5 && scope && !input; i++, scope=scope.parentElement) input=scope.querySelector('input,textarea');
        if(!input) throw Error('Missing text input: '+${JSON.stringify(label)});
        const proto=input.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;
        const setter=Object.getOwnPropertyDescriptor(proto,'value').set;
        setter.call(input,${JSON.stringify(value)}); input.dispatchEvent(new Event('input',{bubbles:true})); input.dispatchEvent(new Event('change',{bubbles:true})); input.blur();
      })()`);
    };
    check(labels.length === 7 && labels[0] === '买卖建议', `Expected advice plus six legacy tabs: ${labels}`);
    await tab('观察与数据状态');
    await waitFor('document.body.innerText.includes("018147") && document.body.innerText.includes("公开研究样本")');
    const before = audit();
    await writeFile(path.join(output,'before.json'),JSON.stringify(before,null,2));
    check(before.counts.products === 0 && before.counts.accounts === 0 && before.counts.journal === 0 && (before.counts.advice_checks ?? 0) === 0, 'Acceptance Store is not empty');
    const initialReplays = (await readdir('runs/artifacts')).filter(name=>name.startsWith('ui-replay-'));
    await capture('overview.png');
    await tab('产品详情');
    await waitFor('[...document.querySelectorAll(".js-plotly-plot")].some(x=>x.getBoundingClientRect().height>0 && x.data?.length===1)');
    await evaluate(`document.querySelector('.js-plotly-plot').scrollIntoView({block:'center'})`);
    await capture('product.png');
    await tab('历史回测');
    await waitFor('document.body.innerText.includes("固定列出两窗口")');
    const summaryDownload = await evaluate(`(() => { const scope=[...document.querySelectorAll('[data-testid=stDownloadButton]')].find(x=>x.innerText.includes('下载此表')); return !!scope?.querySelector('button,a'); })()`);
    check(summaryDownload, 'Research CSV entry missing');
    await select('研究窗口（不改变观察时钟）', '较早');
    await select('到账情景（不是已核交易规则）', '10');
    await select('查看交易 / 信号的路径', '月度动量');
    await select('研究图表主题', '深色');
    await waitFor(`[...document.querySelectorAll('.js-plotly-plot')].some(x=>x.getBoundingClientRect().height>0 && x.data?.length===4 && x.layout?.paper_bgcolor==='#1a1a19')`);
    const chartDetails = await evaluate(`(() => {const plots=[...document.querySelectorAll('.js-plotly-plot')].filter(x=>x.getBoundingClientRect().height>0); return plots.map(x=>({traces:x.data.length,names:x.data.map(t=>t.name),surface:x.layout.paper_bgcolor,yaxis2:!!x.layout.yaxis2,annotations:x.layout.annotations?.length}));})()`);
    check(chartDetails.length === 2 && chartDetails.every(x=>x.traces===4 && !x.yaxis2), 'Expected two single-axis four-path charts');
    await evaluate(`document.querySelector('[role=tabpanel]:not([hidden]) .js-plotly-plot').scrollIntoView({block:'center'})`);
    const hover = await evaluate(`(() => {const p=[...document.querySelectorAll('.js-plotly-plot')].find(x=>x.getBoundingClientRect().height>0); const d=p.querySelector('.nsewdrag').getBoundingClientRect(); return {x:d.x+d.width*0.6,y:d.y+d.height*0.5};})()`);
    await call('Input.dispatchMouseEvent', {type:'mouseMoved', ...hover}, sessionId);
    await waitFor('document.querySelectorAll(".hoverlayer .hovertext").length>0 || document.querySelectorAll(".hoverlayer .legend").length>0');
    await capture('history-dark-hover.png');
    await select('研究窗口（不改变观察时钟）', '较近');
    await select('到账情景（不是已核交易规则）', '7');
    await select('研究图表主题', '浅色');
    await waitFor(`[...document.querySelectorAll('.js-plotly-plot')].some(x=>x.getBoundingClientRect().height>0 && x.data?.length===4 && x.layout?.paper_bgcolor==='#fcfcfb')`);
    await evaluate(`document.querySelector('[role=tabpanel]:not([hidden]) .js-plotly-plot').scrollIntoView({block:'center'})`);
    await capture('history-light.png');
    await call('Browser.setDownloadBehavior', {behavior:'allow', downloadPath:output, eventsEnabled:true});
    await expand('曲线等价日级表与 CSV');
    await clickElement(`[...document.querySelectorAll('[role=tabpanel]')].find(x=>!x.hidden).querySelector('[data-testid=stDownloadButton] button')`);
    const deadline = Date.now()+20000;
    let downloads=[];
    while(Date.now()<deadline) {
      downloads=(await readdir(output)).filter(name=>name.endsWith('.csv'));
      if(downloads.length) break;
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    check(downloads.length>0, 'Research CSV did not download');
    const csv=await readFile(path.join(output,downloads[0]),'utf8');
    check(csv.includes('窗口') && csv.includes('月度动量'), 'Downloaded research summary missing expected fields');
    await tab('信息来源与决策清单');
    await waitFor('document.body.innerText.includes("研究来源 · 假设与缺口")');
    await capture('sources-unregistered.png');
    await tab('模拟账户 / 订单 / 账本');
    await waitFor('document.body.innerText.includes("严格前向模拟未就绪")');
    await expand('新建模拟账户');
    await settled();
    const blocked = await evaluate(`(() => {const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='创建模拟账户'); return !!b?.disabled && ![...document.querySelectorAll('button')].some(x=>x.innerText.trim()==='确认提交模拟委托');})()`);
    check(blocked, 'QDII forward interface unexpectedly enabled');
    await capture('forward-blocked.png');
    const passive = audit();
    check(JSON.stringify(passive)===JSON.stringify(before), 'Passive browse/filter/download changed data');
    await writeFile(path.join(output,'passive-readonly.json'),JSON.stringify(passive,null,2));
    check(JSON.stringify((await readdir('runs/artifacts')).filter(name=>name.startsWith('ui-replay-')))===JSON.stringify(initialReplays), 'Passive browsing created a replay');
    await tab('观察与数据状态');
    await clickElement(`[...document.querySelectorAll('button')].find(x=>x.innerText.includes('登记到公开观察清单'))`);
    await waitFor('document.body.innerText.includes("已登记公开观察项")');
    await tab('信息来源与决策清单');
    await fill('研究理由与待核验事项', '浏览器验收：仅记录事后历史研究，QDII规则仍待核验，不是投资指令。');
    await clickElement(`[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='保存研究记录')`);
    await waitFor('document.body.innerText.includes("研究记录已保存")');
    await capture('notes-registered.png');
    await tab('历史回测');
    await expand('显式复现 · 核对留存经济结果哈希');
    await clickElement(`[...document.querySelectorAll('[data-testid=stCheckbox]')].find(x=>x.innerText.includes('我确认运行本地研究复现')).querySelector('input')`);
    await clickElement(`[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='确认复现留存研究')`);
    await waitFor('document.body.innerText.includes("复现完成：两个窗口")');
    await capture('reproduction.png');
    const replaysAfter = (await readdir('runs/artifacts')).filter(name=>name.startsWith('ui-replay-') && !initialReplays.includes(name));
    check(replaysAfter.length===1, `Expected exactly one explicit replay root: ${replaysAfter}`);
    const receipt=JSON.parse(await readFile(path.join('runs/artifacts',replaysAfter[0],'receipt.json'),'utf8'));
    check(receipt.status.ok && receipt.status.windows.length===2 && receipt.status.windows.every(x=>x.hash_matched), 'Reproduction receipt is not successful');
    await select('研究窗口（不改变观察时钟）', '较早');
    await tab('日志');
    await capture('log.png');
    const after = audit();
    check(after.counts.products===1 && after.counts.journal===1 && after.counts.facts===0 && after.counts.accounts===0 && after.counts.orders===0 && after.counts.ledger_events===0 && (after.counts.advice_checks ?? 0)===0, `Unexpected Store writes: ${JSON.stringify(after.counts)}`);
    check(after.notes[0].product_id==='open_fund:018147:C' && after.notes[0].reason.includes('历史研究引用'), 'Research note lost its Store identity/context');
    check(after.public_hash===before.public_hash && JSON.stringify(after.pins)===JSON.stringify(before.pins), 'Original public data or pinned evidence changed');
    check((await readdir('runs/artifacts')).filter(name=>name.startsWith('ui-replay-') && !initialReplays.includes(name)).length===1, 'Filter rerun duplicated reproduction');
    await call('Emulation.setDeviceMetricsOverride', {width:800,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
    await tab('历史回测');
    await evaluate(`document.querySelector('[role=tabpanel]:not([hidden]) .js-plotly-plot').scrollIntoView({block:'center'})`);
    await capture('history-narrow.png');
    await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false}, sessionId);
    const body = await evaluate('document.body.innerText');
    check(!/Traceback|ModuleNotFoundError|区块的本地读取未完成|这部分内容暂时无法显示|暂时读不到已保存/.test(body), 'Visible application exception');
    const report = {url, mode:'integrated-public-research', tabs:labels, chartDetails, downloads,
      passive_readonly:true, original_public_unchanged:true, pinned_evidence_unchanged:true,
      forward_blocked:blocked, reproduction:receipt.status, before, after, errors,
      warning:'Browser acceptance verifies interaction and reproducibility, not investment performance.'};
    await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));
    check(!errors.length, `Browser runtime errors: ${JSON.stringify(errors)}`);
    console.log(JSON.stringify(report,null,2));
    } catch (error) {
      await writeFile(path.join(output,'failure-body.txt'),await evaluate('document.body.innerText'));
      const {data}=await call('Page.captureScreenshot',{format:'png'},sessionId);
      await writeFile(path.join(output,'failure.png'),Buffer.from(data,'base64'));
      throw error;
    }
  } else {
  check(labels.length === 7 && labels[0] === '买卖建议', `Expected advice plus six legacy tabs: ${labels}`);
  await tab('观察与数据状态');
  await capture('overview.png');
  await tab('产品详情');
  // Select an explicit demo clock; public mode stays at the true current time.
  if (!publicMode) {
    await setExpanded(clockLabels, true);
    await select(['查看时间', '观察时刻'], '22:00');
    await waitFor(`[...document.querySelectorAll('[data-testid=stSelectbox]')].some(x=>/查看时间|观察时刻/.test(x.innerText) && (x.innerText.trim().includes('22:00') || [...x.querySelectorAll('input')].some(input=>input.value.includes('22:00'))))`);
  }
  const symbol = publicMode ? '510300' : 'DEMO-ETF';
  await select(['选择基金', '标的筛选'], symbol);
  if (!publicMode) {
    await select(['查看日期（北京时间）', '研究日期'], '2026-01-31');
    await waitFor(`[...document.querySelectorAll('[data-testid=stSelectbox]')].some(x=>/查看日期|研究日期/.test(x.innerText) && (x.innerText.trim().includes('2026-01-31') || [...x.querySelectorAll('input')].some(input=>input.value.includes('2026-01-31'))))`);
  }
  await tab('产品详情');
  await waitFor('[...document.querySelectorAll(".js-plotly-plot")].some(x=>x.getBoundingClientRect().height>0)');
  await evaluate(`document.querySelector('.js-plotly-plot').scrollIntoView({block:'center'})`);
  await capture('product.png');
  const body = await evaluate('document.body.innerText');
  if (body.includes('Traceback') || body.includes('ModuleNotFoundError')) throw new Error('Visible Python exception');
  const report = { url, initialized, tabs: labels, errors, screenshots: ['overview.png', 'product.png'],
                   warning: 'Browser smoke covers rendering and one local interaction, not investment performance.' };
  await writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  if (errors.length) throw new Error(`Browser runtime errors: ${JSON.stringify(errors)}`);
  console.log(JSON.stringify(report, null, 2));
  }
} finally {
  if (ws) ws.close();
  browser.kill('SIGTERM');
  if (browser.exitCode === null) {
    await Promise.race([
      new Promise(resolve=>browser.once('exit',resolve)),
      new Promise(resolve=>setTimeout(()=>{browser.kill('SIGKILL');resolve();},3000)),
    ]);
  }
}
