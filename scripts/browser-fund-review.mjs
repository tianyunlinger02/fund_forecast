// Independent real-browser acceptance for the new daily multi-factor review.
// Keep browser-smoke.mjs's frozen monthly contract unchanged.
import { execFileSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';

const project = path.resolve(import.meta.dirname, '..');
const artifacts = path.join(project, 'runs/artifacts');
const url = new URL(process.env.APP_URL || 'http://127.0.0.1:8502');
const skipUpdate = process.env.SKIP_UPDATE === '1';
const check = (condition, message) => { if (!condition) throw Error(message); };
const contained = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative && !relative.startsWith('..') && !path.isAbsolute(relative);
};
check(process.env.SMOKE_ROOT, 'Explicit isolated SMOKE_ROOT required');
const smokeRoot = await realpath(path.resolve(process.env.SMOKE_ROOT));
check(contained(await realpath(artifacts), smokeRoot), 'SMOKE_ROOT must resolve inside runs/artifacts');
check(url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname), 'Only parent-started loopback HTTP app accepted');
check(url.port === '8502' || (skipUpdate && url.port === '8501'), 'Only isolated8502 or read-only8501 accepted');
const preservedPreview = skipUpdate && url.port === '8501';
const previewDeployed = process.env.PREVIEW_DEPLOYED === '1';
check(!previewDeployed || preservedPreview, 'PREVIEW_DEPLOYED requires read-only8501');
const appRoot = preservedPreview ? path.join(project, 'data/ui-preview-20261020-v1') : smokeRoot;
const output = path.resolve(process.env.ARTIFACT_DIR || path.join(artifacts, 'multifactor-20261021-v1/browser-v1'));
check(contained(artifacts, output), 'ARTIFACT_DIR must be inside runs/artifacts');
const temporary = path.resolve(process.env.TMPDIR || path.join(project, '.claude-tmp/sessions/fund-build'));
check(contained(path.join(project, '.claude-tmp'), temporary), 'Chrome TMPDIR must stay inside project .claude-tmp');
const chrome = process.env.CHROMIUM || '/home/zc/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome';
const libraryFile = process.env.CHROMIUM_LIBRARY_PATH_FILE || path.join(temporary, 'browser-libraries-1_xz6esi/library-path.txt');
const libraries = (await readFile(libraryFile, 'utf8')).trim();
await mkdir(output, { recursive: true });
check(contained(await realpath(artifacts), await realpath(output)), 'Artifact output symlink escapes runs/artifacts');
await mkdir(temporary, { recursive: true });

// No Workspace/Store constructors in audits. Read-only SQLite transactions also
// hash BLOB bytes rather than JSON-stringifying or silently discarding them.
const audit = () => JSON.parse(execFileSync(path.join(project, '.venv/bin/python'), ['-I', '-c', String.raw`
import hashlib,json,sqlite3,sys
from pathlib import Path
root,smoke,work=map(lambda value:Path(value).resolve(),sys.argv[1:4])
readonly=sys.argv[4]=='1'
deployed=sys.argv[5]=='1'
assert smoke.is_relative_to(root/'runs/artifacts') and smoke!=root/'runs/artifacts'
expected=root/'data/ui-preview-20261020-v1'
assert work==expected if readonly else work==smoke
file=work/'public.sqlite3'
assert file.is_file(), 'Parent must initialize the target database first'
originals=[root/'data/public.sqlite3',expected/'public.sqlite3']
if not readonly:
    for original in originals:
        assert file.resolve()!=original.resolve() and not file.samefile(original), 'Target aliases original data'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def value(item):
    return {'blob_bytes':len(item),'blob_sha256':sha(item)} if isinstance(item,bytes) else item
def inspect(db):
    c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=15)
    c.row_factory=sqlite3.Row
    c.execute('PRAGMA query_only=ON'); c.execute('BEGIN')
    assert c.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()[0]=='public'
    tables,baseline={},{}
    for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        quoted='"'+name.replace('"','""')+'"'
        count=c.execute('SELECT COUNT(*) FROM '+quoted).fetchone()[0]
        assert count<=25000, 'Audit row budget exceeded: '+name
        rows=[list(map(value,row)) for row in c.execute('SELECT * FROM '+quoted)]
        assert sum(len(encoded(row).encode()) for row in rows)<=32*1024*1024, 'Audit byte budget exceeded'
        ordered=sorted(rows,key=encoded)
        tables[name]={'count':count,'hash':sha(encoded(ordered).encode())}
        # Match the independently retained preimplementation protection report.
        old=sorted(json.dumps(row,ensure_ascii=False,sort_keys=True) for row in rows)
        baseline[name]={'count':count,'hash':sha(json.dumps(old).encode())}
    return c,tables,baseline
protected=json.loads((root/'runs/artifacts/multifactor-20261021-v1/protected-before.json').read_text())
catalog_path=root/'configs/research-catalog.json'
catalog=json.loads(catalog_path.read_text())
pins={}
for study in catalog['studies']:
    refs=[study['source'],*study.get('evidence',[])]
    refs += [window[key] for window in study['windows'] for key in ('inputs','report','manifest')]
    for pin in refs:
        source=(root/pin['path']).resolve()
        assert source.is_relative_to(root), 'Research pin escapes project'
        actual=sha(source.read_bytes()); assert actual==pin['sha256'], 'Retained research differs: '+pin['path']
        pins[pin['path']]=actual
assert len(pins)==10 and pins==protected['pins'], 'The ten protected research pins differ'
assert sha(catalog_path.read_bytes())==protected['catalog_sha256'], 'Research catalog changed'
original_state={}
for original in originals:
    c,tables,old=inspect(original)
    try:
        baseline=protected['databases'][str(original.relative_to(root))]
        if deployed and original==expected/'public.sqlite3':
            assert {name:old.get(name) for name in baseline}==baseline, 'Preview legacy business tables changed'
            new={'advice_evidence_captures':1,'advice_evidence_records':1,'advice_evidence_links':1,'daily_advice_checks':0,'advice_policy_activations':0}
            assert set(old)==set(baseline)|set(new), 'Unexpected deployed preview tables'
            assert all(old[name]['count']==count for name,count in new.items()), 'Unexpected deployed review rows'
            raw=c.execute('SELECT raw_bytes,raw_sha256,payload FROM advice_evidence_captures').fetchone()
            assert sha(raw[0])==raw[1]=='62bd12b70da3eefd61598ed75da1799ef28a6bfb9b8293762cb82d4b95b981b8', 'Deployed public fee raw evidence differs'
            metadata=json.loads(raw[2]); assert metadata['family']=='fees' and metadata['status']=='reviewed'
            assert metadata['source_url']=='https://fundf10.eastmoney.com/jjfl_018147.html', 'Deployed fee URL differs'
        else:
            assert old==baseline, 'Original database logical content changed: '+str(original)
        original_state[str(original.relative_to(root))]={'tables':tables,'file_sha256':sha(original.read_bytes())}
    finally: c.close()
c,inventory,_=inspect(file)
try:
    names=set(inventory)
    products=[json.loads(row['payload']) for row in c.execute('SELECT * FROM products')]
    matches=[product for product in products if product['symbol']=='018147' and product.get('share_class')=='C']
    assert len(matches)==1, 'Expected exact actual C-share identity'
    product=matches[0]
    captures=[]
    if 'advice_evidence_captures' in names:
        for row in c.execute('SELECT id,status,raw_sha256,length(raw_bytes) AS raw_bytes, payload FROM advice_evidence_captures ORDER BY observed_at,id'):
            item=dict(row); metadata=json.loads(item.pop('payload'))
            raw=c.execute('SELECT raw_bytes FROM advice_evidence_captures WHERE id=?',(item['id'],)).fetchone()[0]
            assert sha(raw)==item['raw_sha256']==metadata['raw_sha256'], 'Capture raw bytes hash mismatch'
            captures.append({**metadata, 'raw_bytes_count':len(raw)})
    records=[]
    if 'advice_evidence_records' in names:
        records=[json.loads(row[0]) for row in c.execute('SELECT payload FROM advice_evidence_records ORDER BY available_at,id')]
    from rmbfund.domain import now_iso
    from rmbfund.service import Workspace
    from rmbfund.storage import Store
    store=Store.__new__(Store); store.path,store.mode,store.conn,store._depth=file,'public',c,0
    workspace=Workspace.__new__(Workspace); workspace.root,workspace.mode,workspace.store=work,'public',store
    report=workspace.fund_review(product['id'],now_iso())
    result={'root':str(work),'hash':sha(encoded(inventory).encode()),'tables':inventory,
            'counts':{name:item['count'] for name,item in inventory.items()},'product':product,
            'captures':captures,'records':records,'report':report,
            'protected':{'originals':original_state,'catalog_sha256':protected['catalog_sha256'],'pins':pins}}
    print(json.dumps(result,ensure_ascii=False))
finally: c.close()
`, project, smokeRoot, appRoot, preservedPreview ? '1' : '0', previewDeployed ? '1' : '0'], { cwd: project, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }));
const before = audit();
await writeFile(path.join(output, 'before.json'), JSON.stringify(before, null, 2));
const protectedHash = (item) => JSON.stringify(item.protected);
const unchangedTable = (left, right, name) => JSON.stringify(left.tables[name] || null) === JSON.stringify(right.tables[name] || null);
const assertBlocked = (item) => {
  check(item.report.status !== 'ready' && item.report.receipt_state !== 'valid', 'Core-blocked review falsely ready');
  check(item.report.assessment?.usable === false && item.report.receipt === null, 'Missing core created a usable composite review');
  check((item.counts.daily_advice_checks || 0) === 0 && (item.counts.advice_policy_activations || 0) === 0, 'Blocked core consumed a daily check or activation');
  check(unchangedTable(before, item, 'advice_checks'), 'Original monthly checkpoint changed');
  for (const name of ['accounts','orders','ledger_events','journal','profiles']) check(unchangedTable(before, item, name), `Unexpected ${name} write`);
  check(protectedHash(item) === protectedHash(before), 'Protected originals/catalog/pins changed');
};
assertBlocked(before);
if (!preservedPreview) {
  check(before.counts.advice_checks === 1, 'Expected one preserved legacy monthly check');
  check(before.records.length === 1 && before.records[0].family === 'fees', 'Expected only actual reviewed fee evidence, not invented core');
  check(before.captures.length === 1 && before.captures[0].status === 'reviewed', 'Expected one explicitly reviewed actual capture');
  check(/^https:\/\//.test(before.captures[0].source_url) && !/synthetic|fixture|mock|demo/.test(before.captures[0].source), 'Fee provenance is not actual public material');
}

const profile = await mkdtemp(path.join(temporary, 'chrome-fund-review-'));
const browser = spawn(chrome, ['--headless=new','--no-sandbox','--disable-dev-shm-usage','--no-first-run',
  '--disable-background-networking','--remote-debugging-port=0',`--user-data-dir=${profile}`,'--window-size=1440,1100','about:blank'],
{ stdio: ['ignore','ignore','pipe'], env: {...process.env, TMPDIR:temporary, LD_LIBRARY_PATH:[libraries,process.env.LD_LIBRARY_PATH].filter(Boolean).join(':')} });
let ws, sessionId, seq = 0;
const pending = new Map(), errors = [], requests = [], screenshots = [];
const wait = (milliseconds) => new Promise(resolve => setTimeout(resolve, milliseconds));
let afterUpdate = before, updateClicks = 0, browserResult, lastPageState, failureCapture;
try {
  const endpoint = await new Promise((resolve,reject) => {
    let stderr=''; const timer=setTimeout(()=>reject(Error('Chromium readiness timeout')),20000);
    browser.stderr.on('data',chunk=>{stderr+=chunk; const found=stderr.match(/DevTools listening on (ws:\/\/\S+)/); if(found){clearTimeout(timer);resolve(found[1]);}});
    browser.once('error',error=>{clearTimeout(timer);reject(error);});
    browser.once('exit',code=>{clearTimeout(timer);reject(Error(`Chromium exited ${code}: ${stderr.slice(-2000)}`));});
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  ws.onmessage = ({data}) => {
    const message=JSON.parse(data);
    if(message.id && pending.has(message.id)) {const item=pending.get(message.id);pending.delete(message.id);clearTimeout(item.timer);message.error?item.reject(Error(JSON.stringify(message.error))):item.resolve(message.result);}
    if(message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    if(message.method==='Network.requestWillBeSent') requests.push({url:message.params.request.url,method:message.params.request.method,type:message.params.type});
  };
  const call = (method,params={},session=sessionId) => new Promise((resolve,reject)=>{
    const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error(`CDP timeout ${method}`));},25000);
    pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params,...(session?{sessionId:session}:{})}));
  });
  const target=await call('Target.createTarget',{url:url.href},null);
  ({sessionId}=await call('Target.attachToTarget',{targetId:target.targetId,flatten:true},null));
  const evaluate=async(expression,label='DOM evaluation')=>{
    const id=createHash('sha256').update(expression).digest('hex').slice(0,12);
    const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails) throw Error(`${label} expr-${id}: ${result.exceptionDetails.exception?.description || result.exceptionDetails.text}`);
    return result.result.value;
  };
  const saveFailure=async()=>{
    try {await writeFile(path.join(output,'failure-body.txt'),await evaluate('document.body.innerText'));const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(path.join(output,'failure.png'),Buffer.from(shot.data,'base64'));} catch {}
  };
  failureCapture=saveFailure;
  const waitFor=async(expression,label,timeout=45000)=>{
    const deadline=Date.now()+timeout;
    while(Date.now()<deadline){if(await evaluate(`Boolean((${expression}))`,label))return;await wait(250);}
    await saveFailure();throw Error(`Timeout waiting for ${label}`);
  };
  const settled=async(timeout=45000)=>{await wait(400);await waitFor(`document.querySelectorAll('[role=tab]').length===7 && !document.querySelector('[data-stale="true"]') && !document.querySelector('[data-testid=stStatusWidget]')?.innerText.includes('Running') && !document.body.innerText.includes('\\nStop\\n')`,'seven tabs and settled app',timeout);};
  const click=async(expression,label='mouse click')=>{
    await evaluate(`(()=>{const e=(${expression});if(!e)throw Error('Missing click target');e.scrollIntoView({block:'center',behavior:'instant'});})()`,label);
    await wait(120);
    const point=await evaluate(`(()=>{const e=(${expression});const r=e.getBoundingClientRect();if(!r.width||!r.height)throw Error('Click target not visible');return{x:r.x+r.width/2,y:r.y+r.height/2}})()`,label);
    for(const type of ['mouseMoved','mousePressed','mouseReleased']) await call('Input.dispatchMouseEvent',{type,...point,...(type==='mouseMoved'?{}:{button:'left',clickCount:1})});
  };
  const panel="document.querySelector('[role=tabpanel]:not([hidden])')";
  const summary=(label)=>`[...document.querySelectorAll('summary')].find(e=>e.innerText.trim().includes(${JSON.stringify(label)}))`;
  const expand=async(label,open=true)=>{const expression=summary(label);check(typeof await evaluate(`${expression}?.closest('details')?.open`)==='boolean',`Missing expander ${label}`);if(await evaluate(`${expression}.closest('details').open`)!==open)await click(expression,`expander ${label}`);await waitFor(`${expression}.closest('details').open===${open}`,`expander ${label} applied`);};
  const tab=async(label)=>{await click(`[...document.querySelectorAll('[role=tab]')].find(e=>e.innerText.trim()===${JSON.stringify(label)})`,`tab ${label}`);await waitFor(`document.querySelector('[role=tab][aria-selected=true]')?.innerText.trim()===${JSON.stringify(label)}`,`tab ${label} selected`);await settled();};
  const screenshot=async(name,selector=null)=>{
    if(selector)await evaluate(`document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block:'start',behavior:'instant'})`);
    await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:0,y:0});await settled();
    await waitFor("![...document.querySelectorAll('[role=tooltip]')].some(e=>e.getBoundingClientRect().width>0 && e.getBoundingClientRect().height>0)",'tooltips dismissed before screenshot',5000);
    const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    await writeFile(path.join(output,name),Buffer.from(shot.data,'base64'));screenshots.push(name);
  };
  const page=async(diagnostics=false)=>{
    await waitFor("document.querySelector('.st-key-advice_review_factors') && document.querySelector('.st-key-advice_timing') && document.querySelector('.st-key-advice_update button') && !document.querySelector('.st-key-advice_update button').disabled",'keyed factor/timing sections and enabled update');
    const state=await evaluate(`(()=>{const p=${panel};return{body:document.body.innerText,all:document.body.textContent,panel:p?.innerText,
      cards:[...p.querySelectorAll('[class*=st-key-advice_card_] h3')].map(e=>e.innerText.trim()),
      factors:p.querySelector('.st-key-advice_review_factors')?.innerText,timing:p.querySelector('.st-key-advice_timing')?.innerText,
      freshness:p.querySelector('.st-key-advice_freshness')?.innerText,risks:p.querySelector('.st-key-advice_risks')?.innerText,
      diagnostic:document.querySelector('.st-key-advice_diagnostics input')?.checked,json:p.querySelectorAll('[data-testid=stJson]').length,
      jsonText:[...p.querySelectorAll('[data-testid=stJson]')].map(e=>e.textContent).join('\\n'),
      update:[...p.querySelectorAll('button')].filter(e=>e.innerText.trim()==='更新数据并复核建议' && e.getBoundingClientRect().width>0 && e.getBoundingClientRect().height>0).map(e=>({disabled:e.disabled})),
      overflow:document.documentElement.scrollWidth>window.innerWidth+8}})()`);
    lastPageState=state;
    check(state.cards.length>=1 && state.cards.every(label=>label.includes('先等待')), 'Insufficient core UI offered action or lost honest WAIT');
    check(state.update.length===1 && !state.update[0].disabled,'One explicit enabled review update button required');
    check(state.factors.includes('关键资料不足') && /基金身份|投资范围/.test(state.factors) && /配置|持仓/.test(state.factors),'Core blockers not concrete factors');
    check(/费用|费率/.test(state.factors) && /估值/.test(state.factors) && /利率/.test(state.factors) && /汇率/.test(state.factors),'Actual/unknown factors absent');
    check(/日|每天|午夜/.test(state.timing) && /不是|不等于/.test(state.timing),'Daily timing/trading distinction missing');
    check(/未知|缺|不足|未/.test(state.panel) && /保证|试验|验证/.test(state.risks),'Missing evidence or experimental limits hidden');
    check(!/Traceback|ModuleNotFoundError|这部分内容暂时无法显示|暂时读不到已保存|区块的本地读取未完成/.test(state.body),'Visible app failure');
    check(state.diagnostic===diagnostics,'Diagnostic gate state differs');
    if(!diagnostics){check(state.json===0,'Raw JSON built before explicit diagnostics');check(!/\b(?:PUBLIC|DEMO|partial|source_blocked|JSON|hash)\b|哈希|回执|\.sqlite3|\/home\/|runs\/artifacts\//i.test(state.body),'Default exposes technical vocabulary or local path');check(!state.all.includes(appRoot),'Directory exposed before diagnostics');}
    check(!state.overflow,'Horizontal document overflow at active viewport');
    return state;
  };
  const passive=(baseline,label)=>{
    const current=audit();assertBlocked(current);check(current.hash===baseline.hash,`${label} changed isolated database rows or BLOBs`);
    const external=requests.filter(item=>{try{const target=new URL(item.url);return ['http:','https:','ws:','wss:'].includes(target.protocol)&&target.host!==url.host;}catch{return false;}});
    check(external.length===0,`Browser requested external URLs ${JSON.stringify(external)}`);return current;
  };
  const diagnostics=async()=>{
    const checkbox="document.querySelector('.st-key-advice_diagnostics input[type=checkbox]')";
    await click(`(${checkbox}).closest('label') || ${checkbox}`,'enable diagnostic checkbox');await settled();
    await waitFor(`${checkbox}?.checked && document.querySelector('[data-testid=stJson]')`,'explicit diagnostic JSON');
    await expand('按今天查看 / 回看过去',true);
    check((await evaluate('document.body.textContent')).includes(appRoot),'Browser app is not bound to expected target root');
    await screenshot(`review-diagnostics-${updateClicks ? 'updated' : 'default'}.png`,'[data-testid=stJson]');
    await expand('按今天查看 / 回看过去',false);
    await click(`(${checkbox}).closest('label') || ${checkbox}`,'disable diagnostic checkbox');await settled();
    await page();
  };
  await call('Runtime.enable');await call('Page.enable');await call('Network.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await settled();
  const labels=await evaluate("[...document.querySelectorAll('[role=tab]')].map(e=>e.innerText.trim())");
  check(JSON.stringify(labels)===JSON.stringify(['买卖建议','观察与数据状态','产品详情','历史回测','信息来源与决策清单','模拟账户 / 订单 / 账本','日志']),'Seven ordered tabs changed');
  const initial=await page();passive(before,'Initial page load');
  await writeFile(path.join(output,'review-default-body.txt'),initial.body);
  await screenshot('review-default-desktop.png','h1');
  await screenshot('review-factors-desktop.png','.st-key-advice_review_factors');
  await screenshot('review-timing-desktop.png','.st-key-advice_timing');
  await call('Emulation.setDeviceMetricsOverride',{width:800,height:1100,deviceScaleFactor:1,mobile:false});
  await page();await screenshot('review-default-narrow.png','h1');await screenshot('review-factors-narrow.png','.st-key-advice_review_factors');await screenshot('review-timing-narrow.png','.st-key-advice_timing');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await expand('可选：用钱计划与持有情况',true);
  await click("[...document.querySelectorAll('.st-key-advice_review_holding label')].find(e=>e.innerText.trim()==='还没买')",'optional holding answer');
  await settled();await page();passive(before,'Questionnaire holding edit');
  await click("[...document.querySelectorAll('.st-key-advice_review_holding label')].find(e=>e.innerText.trim()==='暂不填写')",'clear optional holding answer');
  await settled();await page();passive(before,'Questionnaire cleared without action unlock');
  await expand('可选：用钱计划与持有情况',false);
  await expand('查看资料来源和所属日期',true);
  if(!preservedPreview){const sourceText=await evaluate(`${summary('查看资料来源和所属日期')}.closest('details').innerText`);check(sourceText.includes(before.records[0].report_date)&&sourceText.includes(before.records[0].source_url),'Visible source dates/URL not actual fee evidence');}
  await screenshot('review-source-evidence.png',".st-key-advice_freshness");
  await expand('查看资料来源和所属日期',false);
  await diagnostics();passive(before,'Source and diagnostic expansion');
  await writeFile(path.join(output,'passive-before-update.json'),JSON.stringify(audit(),null,2));

  if(!skipUpdate){
    const started=Date.now();
    await click("[...document.querySelectorAll('button')].find(e=>e.innerText.trim()==='更新数据并复核建议' && e.getBoundingClientRect().width>0 && e.getBoundingClientRect().height>0)",'one explicit review update');updateClicks++;
    await waitFor("['检查已返回，但必要资料或条件尚未齐','更新没有完成，请稍后再试','这次更新未完成'].some(value=>document.body.innerText.includes(value))",'explicit update returned',180000);
    await settled(180000);
    const updated=await page();afterUpdate=audit();assertBlocked(afterUpdate);
    check(updated.panel.includes('部分基金资料的公开来源暂时无法核查'),'Explicit update hid actual source-blocked status');
    check(unchangedTable(before,afterUpdate,'advice_evidence_captures')&&unchangedTable(before,afterUpdate,'advice_evidence_records')&&unchangedTable(before,afterUpdate,'advice_evidence_links'),'Blocked source silently renewed or changed retained factors');
    check(afterUpdate.report.evidence.records.some(item=>item.id===before.records[0].id),'Known actual fee evidence disappeared after source blockage');
    check(afterUpdate.counts.snapshots>=before.counts.snapshots,'NAV source history lost');
    await writeFile(path.join(output,'after-update.json'),JSON.stringify(afterUpdate,null,2));
    await writeFile(path.join(output,'review-updated-body.txt'),updated.body);
    await screenshot('review-updated-desktop.png','h1');await screenshot('review-updated-factors.png','.st-key-advice_review_factors');await screenshot('review-updated-timing.png','.st-key-advice_timing');
    browserResult={elapsed_ms:Date.now()-started,nav_latest_date:afterUpdate.report.monitor?.latest_date,fee_report_date:before.records[0].report_date,fee_first_available:before.records[0].available_at};
    await diagnostics();passive(afterUpdate,'Post-update diagnostics');
  }
  await tab('历史回测');await waitFor("document.body.innerText.includes('固定列出两窗口')",'frozen historical studies');await screenshot('history-readonly.png');
  await tab('信息来源与决策清单');await waitFor("document.body.innerText.includes('研究来源')",'source information');
  await tab('模拟账户 / 订单 / 账本');await waitFor("document.body.innerText.includes('严格前向模拟未就绪')",'QDII forward gate');
  await expand('新建模拟账户',true);await settled();
  const blocked=await evaluate("(()=>{const button=[...document.querySelectorAll('button')].find(e=>e.innerText.trim()==='创建模拟账户');return !!button?.disabled && ![...document.querySelectorAll('button')].some(e=>e.innerText.trim()==='确认提交模拟委托')})()");
  check(blocked,'QDII paper-trading gate became available');await screenshot('forward-blocked.png');
  await tab('观察与数据状态');await tab('产品详情');await tab('日志');await tab('买卖建议');
  await page();const after=passive(afterUpdate,'All seven tabs and return');
  await screenshot('review-after-navigation.png','h1');
  check(errors.length===0,`Browser runtime errors ${JSON.stringify(errors)}`);
  await writeFile(path.join(output,'after.json'),JSON.stringify(after,null,2));
  const report={url:url.href,smoke_root:smokeRoot,app_root:appRoot,preserved_preview:preservedPreview,tabs:labels,
    explicit_update_clicks:updateClicks,outcome:skipUpdate ? 'WAIT_CORE_MISSING_READONLY' : 'WAIT_CORE_SOURCE_BLOCKED',source_status:skipUpdate ? 'not_refreshed' : 'source_blocked',
    actual_fee_evidence:before.records.filter(item=>item.family==='fees'),update:browserResult,
    passive_readonly:true,questionnaire_session_only:true,diagnostics_explicit_only:true,
    desktop1440_and_narrow800:true,old_monthly_unchanged:true,daily_checks:after.counts.daily_advice_checks||0,
    activations:after.counts.advice_policy_activations||0,protected_originals_unchanged:true,pins_unchanged:10,
    blob_hash_audits:true,no_trading_writes:true,qdii_forward_blocked:blocked,errors,screenshots,
    warning:'Engineering/source-blockage acceptance only; no complete real multifactor analysis or performance validation claimed.'};
  await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
} catch(error) {
  if(failureCapture)await failureCapture();
  if(lastPageState)await writeFile(path.join(output,'failure-page-state.json'),JSON.stringify(lastPageState,null,2));
  try{if(ws&&sessionId){const id=++seq;const result=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('failure capture timeout')),5000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression:'document.body.innerText',returnByValue:true},sessionId}));});await writeFile(path.join(output,'failure-body.txt'),result.result?.value||'');}}catch{}
  await writeFile(path.join(output,'failure.json'),JSON.stringify({error:String(error),update_clicks:updateClicks,errors,requests},null,2));throw error;
} finally {
  if(ws)ws.close();
  browser.kill('SIGTERM');
  if(browser.exitCode===null)await Promise.race([new Promise(resolve=>browser.once('exit',resolve)),wait(3000).then(()=>browser.kill('SIGKILL'))]);
  for(const item of pending.values())clearTimeout(item.timer);
}
