# 免费数据源与适配器边界

本项目的数据适配器只读取不需要付费、登录、Token的公开来源。显式研究动作另可调用用户已有模型API，使用其原服务额度；这不是公开数据源免费承诺，也不另购模型服务。**AKShare 是调用包装层，不是事实来源**：写入 `Fact.source` 时使用实际上游 `eastmoney:...` 或 `sina:...`，并在 `source_url`、`raw_records` 中保留来源。公开网页可达不代表字段语义、历史完整性、许可或服务稳定性已验证；每次刷新都必须保存本次 `observed_at`、来源、状态和警告。

## `AkshareProvider`

入口为：

```python
AkshareProvider(timeout=15, retries=1, min_interval=1).fetch(
    product, observed_at, start_date=None, end_date=None
) -> ProviderResult
```

它不依赖 `Store`，也不会在失败时生成演示数据。每次 AKShare 调用在独立的 `python -I` 子进程中运行，并由父进程硬超时终止；调用次数是 `retries + 1`，请求之间遵守 `min_interval`。因此 AKShare 底层若使用没有默认 timeout 的 HTTP 调用，也不会无限期占住刷新进程。缺少 `akshare`、空响应、返回结构异常、字段异常或日期/数值校验失败都以状态和警告报告。

`ProviderResult` 字段为 `provider`、`product_id`、`facts`、`raw_records`、`warnings`、`status`、`source_url`，`to_dict()` 可直接用于序列化。可能的状态包括 `ok`、`partial`、`error`、`timeout`、`missing_dependency`、`unsupported`。

### ETF 日线

首选东财：`fund_etf_hist_em(symbol, period="daily", start_date, end_date, adjust="")`。`adjust=""` 是不复权价格；适配器将实际日期、开高低收和成交量做有限数、正数、日期和高低价关系校验。当前官方文档未明确该历史接口的成交量单位，新浪文档虽标“手”但示例量级需要独立核验。因此两源成交量均只保留 `source_volume` 和 `source_volume_unit="unverified"`，不猜手/份乘数，也不写标准执行 `volume`；不把复权价用于执行行情。

如果东财调用失败或响应畸形，则**整次 ETF 请求**明确备用新浪 `fund_etf_hist_sina`，不会把东财和新浪按日期拼接。Fact.source 会是 `sina:fund_etf_hist_sina`，并带备用警告。

ETF 的历史行没有被虚构的来源发布时间；`available_at` 和 `observed_at` 是 `max(本次传入观察时间, 请求全部结束的真实 now)`，不会比实际抓取结束更早。没有请求区间内的有效数据时不会用区间外数据冒充覆盖。

### 场外基金

调用东财的 `fund_open_fund_info_em`：

- `indicator="单位净值走势"`：生成 `nav`，要求有可解析的 `净值日期` 和正的 `单位净值`；这不是盘中估值。
- `indicator="分红送配详情"`：只有同时取得权益登记日、除息日、发放/到账日和每份分红，才生成精确 `dividend`。
- `indicator="拆分详情"`：缺少能确立权益边界的完整日期或明确比例时不生成精确拆分权益；保留为 `announcement`，标注 `requires_manual_verification=true`。

缺少分红的 `record/ex/pay` 日期，或分红金额不完整时，不会用除息日推断登记日/到账日，不会把空值变成零，也不会默默产生权益事实。只返回警告或未核验公告。适配器不伪造历史申购/赎回开放状态、免费精准历史费率、来源发布时间或公告完整性。

`fund_purchase_em`、费率和概览接口可作为后续独立探针，但不是主 `fetch` 的硬依赖。

### 明确的 HTTP 备用适配器

当前`Workspace.refresh`对场外基金使用`EastmoneyProfileProvider(retries=0).fetch(...)`单次请求公开NAV；失败直接保留来源错误，不自动换源或重试。AKShare适配器仍可用于单独明确的研究，不是当前预测更新的自动后备。它直接请求 <https://fund.eastmoney.com/pingzhongdata/{symbol}.js>，只安全解析带产品代码校验的 `Data_netWorthTrend` JSON 数组；Fact.source 是 `eastmoney_js_json`，状态固定为 `partial`（即使拿到净值），因为该页面不证明完整历史、分红/拆分、申赎状态、费率或发布时间。来源切换保留各次失败快照，不把成功NAV说成综合信息接通。

ETF 还提供 `SinaDailyProvider(...).fetch(...)`，只读新浪公开 `CN_MarketData.getKLineData` 最近 20 条日 K。该地址与 `fund_etf_hist_sina` 同属新浪，**不是独立交叉验证**；仅支持 ETF，始终报告 `partial`，不宣称完整历史，不覆盖权益事件或费率。实测 URL 为 <https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh510300&scale=240&ma=no&datalen=20>。

## 公开网页探针

`run_probes(output_dir, timeout=12, product_limit=2)` 是低频连通性和覆盖检查，不生成 `Fact`，也不把探针结果当 NAV。建议 CLI 输出目录使用：

```text
runs/artifacts/source-probes/
```

每次运行在该目录新建独立 `probe-<observed-at>/`，外部响应放在各自的空子目录中；`latest.json` 和带时间戳的 `probe-report-*.json` 是汇总报告。不会在下载目录中运行解释器，也不使用系统临时目录。每个检查报告 `status`、真实 HTTP 延迟、HTTP 状态、coverage、errors、许可说明和 URL。

当前候选：

1. `https://fundgz.1234567.com.cn/js/000051.js`：只按白名单解析 `jsonpgz({...})`。这是盘中估值，不是最终净值，不能进入 paper NAV。
2. `https://fund.eastmoney.com/pingzhongdata/000051.js`：只按白名单解析已知 `Data_netWorthTrend = [...]` JSON 数组，绝不 `eval` 或执行 JavaScript。页面可达不等于字段、发布时间或历史覆盖已确认。
3. CSRC 基金披露、上交所基金公告和巨潮基金披露公开入口：只检查是否返回有用的基金公告文档列表；HTTP 200 的 JS 壳、导航页、登录/验证码页不算成功。公告正文、产品归属、发布时间和完整性仍须人工核验。

### 2026-10-20 本机实际检查

- 东财 `pingzhongdata/000051.js`：HTTP 200，安全 JSON 数组共 4,198 行。直接适配器请求 2026-10-01 之后的区间得到 8 条 `nav`，覆盖 2026-10-08 至 2026-10-19，完成观察时间 17:23:59+08:00；`partial`，不是完整权益覆盖。
- 新浪上述公开日 K JSON：HTTP 200，20 行。直接适配器过滤 2026-10-01 之后得到 9 条 `bar`，覆盖 2026-10-08 至 2026-10-20，完成观察时间 17:34:26+08:00；`partial`，成交量仅保留未独立核验单位的原值。
- `fundgz/000051.js`：HTTP 200 但不是预期 `jsonpgz` 数据；失败，未冒充最终 NAV。
- CSRC 入口 TLS 连接失败，上交所公告入口 HTTP 403，巨潮候选入口 HTTP 404；均记录失败并停止，不绕过限制。未宣称已接通官方公告。
- AKShare 官方文档 HTTP 200，仅用作字段参考；这不等于 AKShare 依赖已安装或封装接口已实际成功。默认测试完全离线。

原始正文和结构化报告保留在 `runs/artifacts/source-probes/` 的独立新目录，汇总见 `probe-report-2026-10-20T17-15-20.524596_08-00.json`；直接适配器结果见 `direct-fetch-2026-10-20T17-23-58.296921_08-00/result.json` 和 `sina-fetch-2026-10-20T17-34-25.300876_08-00/result.json`。

网络受限时如实记录失败；不绕过代理、登录、验证码、robots 或其他网络策略。AKShare 文档入口为 <https://akshare.akfamily.xyz/data/fund/fund_public.html>，文档存在不等于当前环境已安装或实际接口可用。

## 独立最近估算（2026-10-24接入）

`providers/intraday.py`提供`fetch_estimate(source, symbol)`，与日级适配器、Fact和Store完全独立；只接收公开来源ID和六位代码，不接收份额、成本或个人ID。来源失败不回退到其他网站，不执行下载内容。运行期单次HTTPS请求：正常证书校验、10秒网络超时、响应上限256KiB、无环境代理、无重定向、无自动重试；没有登录、Cookie、密钥或收费功能。

| 来源 | 当前实际数据 | 未确认部分 |
|---|---|---|
| 基金速查网 `https://fund.dayfund.com.cn/ajs/ajaxdata.shtml?showtype=getfundvalue&fundcode={code}` | 活跃页面脚本采用11字段pipe响应，索引5为估算百分数，7为估算值，9/10为来源日期/时间；018147和000051均取得过非空值 | 基准净值/日期、时区和币种未证实；没有原始分钟历史，未验证交易时段时间推进 |
| 持仓记录助手 `https://withlees.cn/api/h5/fund-estimates.json` | 公共JSON的精确代码键、changePercent、updatedAt和市场状态；核验21条中20条有值 | updatedAt只是整份快照生成时间，没有单基金行情时间/基准日；无018147 C类，不使用539002 A类替代 |

百分数均从百分点转换为内部小数比率，不改算涨幅。基金速查网的索引8不能统一当基准：000051与018147的数值口径不同。两家都不标为官方净值或已验证实时源。持仓记录助手未提供时间充分的基金曲线，不能把JSON重新生成误画成行情更新。

走势页的“最近估算”默认不联网；只在明确刷新/开启自动检查后请求选定估算来源。`st.cache_data(ttl=60, persist=None)`缓存成功和失败；定时fragment只持有公开字符串和会话采样，不持有Workspace/SQLite。取数失败关闭自动检查；明确拒绝/验证/证书失败会在当前会话锁住该来源，避免再次自动请求。来源声明的公开免费基础查看不等于稳定API或再分发许可，也未绕过其App付费功能。

估算不写`facts`/`snapshots`，不更新NAV检查时间、不用于持仓市值或预测输入/历史评分。真实核验原响应与限制保留在[`intraday-free-sources-20261024-2JXVVu`](../runs/artifacts/intraday-free-sources-20261024-2JXVVu/report.md)。交易时段动态性和逐只QDII覆盖仍待检验；本次接入的是**最近可取得的第三方估算**，不是实时行情承诺。

## 独立多因素资料（2026-10-21新增）

新资料使用`advice_evidence_captures/records/links`，不写NAV快照，不给NAV缓存续鲜。首次可用与最近成功核查分离；日期级披露不造午夜发布时间；同内容重抓不改首次已知，修订追加。读取不迁移新表，显式初始化/更新才建表并一致备份。

初版（2026-10-21）实际复核到的是原留存018147费率HTML（原始SHA-256 `62bd12b70da3eefd61598ed75da1799ef28a6bfb9b8293762cb82d4b95b981b8`），不是新成功抓取；新导入按真实复核时间可用，重复本地导入不更新核查。表格能支持一般管理/托管/C类销售服务和赎回费分段，**不能证明实际支付宝渠道与持有天数口径**。

初版未取得投资范围/近期实际持仓/现任经理任期/全基金合并规模的合格原件；不能从费率页标题、季度日期、搜索摘要或份额规模推导补齐。该阶段来源受阻与验收结论保留，后续新取证不改写历史。

### 免费来源扩展（实际取得为上海时间2026-10-22）

基金买卖网`www.jjmmw.com/fund/018147/`及公告转载正文可以正常读取。经实际附件链接取得文件编号，再从已允许的`pdf.dfcfw.com`取得并人工复核下列原始PDF；未访问被拒来源，也未将搜索摘要当原件：

| 原始PDF | 报告与用途 |
|---|---|
| [2026一季报](https://pdf.dfcfw.com/pdf/H2_AN202604211821364605_1.pdf) | 3月31日A/C合并净资产45.69亿元，仅导入历史规模比较，不把旧仓位当当前持仓。 |
| [2026中报](https://pdf.dfcfw.com/pdf/H2_AN202608281828588412_1.pdf) | 6月30日净资产75.90亿元、NAV分母权益81.60%、前十大约62.2%、GICS信息技术81.18%。保留报告分母、日期与有限持仓覆盖。 |
| [9月C类产品资料概要](https://pdf.dfcfw.com/pdf/H2_AN202609211829693880_1.pdf) | 9月17日编制、9月21日送出；份额/投资范围、三位经理全基金起任日、直接年度费1.80%与赎回费档。概要综合运作费率测算1.82%是另一口径。 |
| [基准变更公告](https://pdf.dfcfw.com/pdf/H2_AN202608211828278859_1.pdf) | 落款8月22日，9月21日起生效；90%MSCI新兴市场+5%中债-综合全价（总值）+5%人民币活期。镜像文件名8月21日不是已核实首次披露日。 |

实际字节、SHA-256、取得时钟、页表定位、失败/空返回/错基金文件及未采用字段详见新产物目录[`018147-source-expansion-20261021-v1`](../runs/artifacts/018147-source-expansion-20261021-v1/)。新增7条规范化记录已在隔离库验证并追加到8501持久预览，原留存费率及旧业务数据不改；并未自动发出日级检查或动作。数据首次可用使用真实导入时钟，不以报告披露日期倒填历史。

这些仍是公开转载的管理人署名文件，不是直接官网认证；同一东方财富上游不能因不同转载域名变成多个独立来源。国家表按证券交易所归类，不导成经济国家；资产负债表美元净额未完成衍生品/对冲适配，不导成未对冲币种敞口。新复合基准的严格估值与当前支付宝交易条件仍未知。当前预测更新只获取公开净值，不自动重抓PDF或假称背景资料已经续鲜；旧检查器专用的资料家族刷新状态已移除。

维护者可用`fund-review-init`和`fund-review-import ... --reviewed`复核导入带原始文件、公开URL和页表定位的资料。本地文件导入不再维护固定域名白名单：导入不会访问该URL，也不能凭域名认证文件真实；其他免费公开来源无需修改代码即可使用。URL仍需是无凭据的公开HTTPS地址，来源归属、原始字节、实际可用时间及定位照常记录。自动联网来源的范围不因此扩大。导入校验只验证契约与可定位性，不替代人工事实核对；PDF定位不冒充自动OCR。资料结构见[多因素说明](multifactor-review.md)；不用于账户、余额或个人持仓文件。

## 预测因子来源研究

新预测模型不把当前持仓、经理或公告倒填历史。首轮另行尝试两项历史市场资料：Stooq EEM.US只作为宽基新兴市场代理候选（不是本基金精确基准或持仓），官方FRED DEXCHUS只作为汇率候选。

本轮Stooq单次CSV请求返回404，FRED在工具网络/安全预检阶段被阻断；没有绕过、重试、使用镜像或编造因子。规范化文件为空表示不可用，不是零收益。这些获取失败不能证明因子有用或无用；当前模型仍明确为净值模型。原始响应与实际请求说明见`runs/artifacts/forecast-replacement-20261022-ESyKMj/factor-research/source-audit-20261022/outputs/source_availability_report.json`。

## 预测市场因子验收（2026-10-24）

本轮首个通过格式与语义验收的外部历史序列是 CBOE 公开 VIX 日收盘数据。原始下载、传输记录、校验脚本和规范化结果保存在[`forecast-optimization-20261024-n2EBUl/source-audit`](../runs/artifacts/forecast-optimization-20261024-n2EBUl/source-audit/)；采用的规范化文件是 `normalized/cboe-vix-normalized-validated-v2.json`。

- 采用范围为2023-01-03至2026-10-23，共984条；原始下载实际发生于2026-10-24。
- 它是VIX指数点的日收盘值，不是018147的精确持仓、基准或收益序列；不把它当纳指基金信号。
- 每条历史记录没有真实`published_at`或`available_at`。研究只按“净值所属日期前至少2个自然日的最近VIX收盘、最多允许7日陈旧”做明确的经济日期探索假设；这不是历史PIT可交易回放。
- VIX输入只允许通过显式离线研究命令传入；缺失时保留NaN并让因子候选明确回退，不补0、不借别的市场序列。因子模型与NAV-only模型分别记录candidate/input/protocol哈希。
- RBA官方汇率单次检查返回403后停止；iShares EEM页面未提供可验收的历史日线载荷，未接入。既有受阻的FRED、Stooq、ECB等不重试、不用别名或镜像绕行。
- 取得成功不代表许可了任意再分发；原始文件仅作为本地研究证据保存。

## 投资因素来源与显式更新（2026-10-25）

`providers/factors.py::fetch_factor_bundle(product_id=...)`只在明确更新时运行，不读个人份额或账户。纯解析器已用实际留存响应离线核对。来源审核与原件在`runs/artifacts/investment-factors-20261025-Xk29ok/source-qualification-agent/`，不覆盖上述失败研究。

| 角色 | 发布者与采用值 | 本次实际历史 |
|---|---|---|
| 市场 EEM / 行业 SOXX | Nasdaq公开historical，美元Close/Last；身份来自普通发行人资料，调整方式未知、非已验证总回报 | 各645条，2024-04-01—2026-10-23 |
| USD/CNY | Sina公开表央行中间价，明确100外币兑人民币，除100为CNY/USD；非现汇买卖或spot | 464条，2024-11-22—2026-10-23 |
| 美国2Y/10Y收益率 | Eastmoney `RPTA_WEB_TREASURYYIELD`的`EMG00001306/EMG00001310`，百分数（4表示4%） | 各472条，2024-12-04—2026-10-23 |

实时更新使用同一已核对路线与日期参数，Nasdaq/Sina终点保守到纽约前一个自然日；Sina最多12页、页间1.05秒，收益率一页有界，初次载入不保证全部历史。缺值不补零。SOXX发行人已披露2026-11-05开始拆股调整交易，届时当前来源契约停止更新，待重新核对价格调整再继续。

每次成功响应保存原件、URL、真实完成时间及hash；连接失败也留失败receipt，不伪造正文hash。运行期HTTPS正常证书校验，无重定向、环境代理、Cookie、重试或自动换源。任何首次获取/解析失败立即停止后续来源，部分成功证据保留，但不假称完整bundle。外部原件放独立新空目录，与脚本分开，不执行下载内容。

统一bundle为`schema_version=1,series={角色:{series_id,label,source,source_url,currency,unit,return_basis,captured_at,rows:[{date,value,available_at,historical_available_at:null,source_sha256}]}}`。历史发布时间未知保持null，`available_at`是实际本次取得，不回填为收盘日。Store对同经济值保留first-seen，修订追加；仅明确导入/更新时创建因素表。价格调整/历史vintage仍有局限，当前只合格于今日训练与明确假设下的历史重建，不叫PIT。来源成功和解析通过不代表投资预测有效。

## 公司基本面原件（2026-10-26）

新家族`issuer_fundamentals`复用现有证据表与`fund-review-import --reviewed`。三类窄契约是`financial_release`（每次发布打包指标）、`security_mapping`（准确股类/市场/币种/ADS比例）和`price_history`（打包价格窗口及调整口径）。没有新建通用ETL或每日自动抓财报；普通浏览只读，导入由真实时钟赋首次可用时间，不能由JSON填回过去。

本轮独立产物目录为`runs/artifacts/issuer-fundamentals-20261026-IvHGEj/`：

- NVIDIA官方新闻原发布页：FY2024Q4至FY2027Q2共11期；保留原GAAP/非GAAP、收入、毛利率、现金流、PPE＋无形资产现金购置和原下一季指引。程序首批只导入GAAP基线，非GAAP原值仍保留在审核档，FY2027定义变化不混算。FY2027Q3只知财季身份、未明示期末的指引保留`end=null`，不推造日期。
- TSMC独立合格原季度发布10期，2024Q1至2026Q2；另有原管理报告与官方ADS映射原件。NYSE的TSM ADS与台湾普通股分开；官方1ADS=5普通股只是股类关系，不是汇率。
- Nasdaq新股票历史各706行，2024-01-02至2026-10-23，**调整方法unknown，不接入历史EPS/P特征**。独立info单笔`lastSalePrice`、原日期、市场Closed及`isRealTime=false`可作明确审阅的当前名义报价，不推导全历史不复权；`currency=null`保持原件，USD使用美国上市市场美元报价惯例解释并披露，不伪造来源字段。
- 韩国补源严格限定两次请求、均200，无重试。Samsung Q2原PDF可支持收入、毛利率、购置PP&E现金支出和经营现金流四项初步合并K-IFRS数据；只导入当前分析，公开日期未知保持null。普通股EPS10,849韩元未注明basic/diluted，仅留审核旁表，不算TTM或估值。SK hynix Q2原发布虽有收入和2026-07-29发布日期，但未明确实际季度起止，暂留原件与审核表，不强行补日历或导入；76%是营业利润率，不是毛利率。两家公司均未填造数值指引、Q3实际值或价格。
- 所有原件、URL、SHA-256、实际取得时钟及页/表/行定位保留。原发布页今天的字节不等于历史不可变快照；日期级公开信息不伪造精确时刻。旧季比较表保留后公布来源日期。手工复核HTML定位允许合法合并单元格，仍要求实际完整表/行；自动费率解析继续要求其原先的精确未合并结构。

来源故障同样保留：NVIDIA季度索引429后停止；TSMC 2026Q3两份PDF先404后被研究代理换主机取得，违反既定不恢复边界，已停止并将该两份原件**隔离排除**，不进入合格指标/TTM/导入。具体URLs与顺序在`tsmc-source/logs/transport-qualification.json`，原件未覆盖或删除。Columbia研究页403、TWSE OpenAPI网络不可达也停止，无后续换工具恢复。

这些资料支持有日期的公司分析与条件算术；当前不证明18维概率模型可用，也不证明买卖有效。

## 当前在线判断路径（2026-10-26）

当前判断与历史资料留存是两条路径。进入当前判断时，`providers/current.py::collect_current`在内存中读取实际查询日可取得的公开单位净值、基金公告列表和已核对的相关发布列表；不要求先写`response.bin`、PDF或SQLite，也不把本次标题/摘录伪装成本地作者研究。总览、走势与历史回放不触发这条路径。

已实际核对并接入的默认来源：

- 东财基金`pingzhongdata/{代码}.js`：按实际查询日滚动窗口解析`Data_netWorthTrend`；来源日、取得时间和候选净值口径分开保存。
- 新浪基金公告列表：按基金代码动态发现公告条目。本轮实际检查时钟为上海时间2026-10-27，发现018147发布于2026-10-26的第三季度报告公告；详情页面未取得可解析正文时只显示`unparsed_document`/标题，不推断最新持仓、费率或买卖含义。
- NVIDIA官方RSS发布列表：只把发布者摘要或正文摘录作为未复核当前消息；不证明基金当前仍持有该公司。
- 韩国贸易、产业和能源部英文发布列表：只保留有日期的相关条目及在线摘录，计划/未确认事件不当作已实现盈利。

一次当前查询的结果供同一页面的头部估值、目标/替代比较、已记录组合估值、金额条件和保存判断共同使用；比较只使用本次查询实际通过身份、币种和时间核对的共同单位净值日期。无法刷新的一只基金不能与其他基金的当前序列悄悄混用；如同时有本地旧序列，只标为带原日期的历史参考。失败来源保留本次失败状态，不续鲜旧日期；当前会话停止的来源不因点击刷新而自动重试。

在线消息保留来源、发布时间精度、取得时间、摘录状态和与基金的关联理由。只有合格结构化披露才进入相应当前操作/披露计算；标题、摘要或未解析文件只作为待核对事实，不自动改变买卖动作。原作者研究日期、研究姿态和短期预测概率不被在线消息改写。保存判断时只封存实际展示的紧凑在线上下文，历史回放不再联网。

## 定向基金资料与政策/产业研究（2026-10-26）

`providers/followed.py::refresh_followed`仅接收明确公开身份、所选家族、显式来源计划和请求预算。只执行有界GET，正常TLS、无环境代理/凭据/重定向/自动重试或换源。各响应放独立新目录；失败目标保留停止状态，不用新host别名恢复。`Workspace`只传代码、份额等公开身份，不把个人份额、成本或预算交给来源。

NAV、披露、政策/产业分别返回job和family收据。仅取得PDF/页面标为`captured_unparsed`，不冒充提取成功；显式复核固定hash、字面摘录/位置和人工PDF文本才能产生结构化事实。`followed-review-retained`可零新请求复核成功留存原件，保留原捕获时间与本次真实复核可用时钟；它不是本次在线检查。披露通过已有原件导入器进入证据表；新闻保留复核事实和收据，由作者笔记论证，不强塞旧events家族。NAV只在原行存在时保存unitMoney，缺标记不补空。

`followed-state.json`保留各基金的来源hash、停止目标与更新/待复核变化；局部刷新不抹其他基金基线，连续不变检查不清除未复核变化。候选与来源配置每次显式安装追加实际可用版本；历史as-of不采用未来配置。本段留存路径不在浏览时自动运行，也不自动发现新文章；上文的当前在线路径会读取动态列表。两者都不做全市场抓取或每日自动投资解释。

本轮真实调用18次（6定向搜索+12GET），取得四只基金共300净值观察；73个共同日期不是所有候选均合格的证明。018147/539002 A与100055 A两组比较可用，118001份额类别仍缺失。真实持仓均为2026-06-30，不是当前仓位；总回报和当前支付宝条件未核对。

采用官方富国产品资料/中报、易方达产品资料/中报、留存建信资料，以及[NVIDIA 2026-10-21供应链计划](https://nvidianews.nvidia.com/news/nvidia-and-supply-chain-partners-expand-capacity-to-accelerate-ai-infrastructure-buildout)与[韩国2026-09-01战略物项修订](https://english.motir.go.kr/eng/article/EATCLdfa319ada/2720/view)。扩产是计划，不是利润；出口许可的具体型号、目的地、例外和逐公司影响仍未知。`sources/citations.json`保留日期、定位和摘录；`receipt-accounting.json`如实记载首批收据计数缺陷，原收据未改写，后续实现已修。

18维资格补源另有两次单独请求：Nasdaq data-providers传输失败，TSMC FAQ403，均停止。没有解决历史价格/币种与TSMC同证券盈利口径；新研究入口输出真实不可用，没有新准确率。上述资料和首次真实CLI导入在`runs/artifacts/fund-buyer-20261026-dYH2MV/`，原8501与原库没有新采集、发行或部署。

## 在线正文与API研究（2026-10-27）

当前路径新增`providers/documents.py`：HTML按相关正文节、PDF通过本机Poppler的标准输入/输出提取页文本，在RAM使用，不要求先写PDF或全文入库。扫描件、标题页、附件壳与提取错误各自保留状态，不从它们猜数字。消息短摘录仍供页面速看，研究输入不再只有前2400字；若仅选取相关窗口，单独说明选择范围与未覆盖部分。

`collect_current`提供`evidence_by_product`正文包和`record_evidence_by_product`连续原文定位。每份正文包含真实URL、产品身份、原披露时间及精度、本次取得时间、状态与页/节。只有正文中确实存在的逐字摘录能绑定为`online_evidence`；规范化JSON不冒充发布者原文，旧摘录保留原取得时间。实际采用的结构化经济记录按精确页/节与原文映射保留为`record_basis`，用于跨会话变化比较，不把同URL的全部记录都当作被引用，也不把尚未采用的在线记录或全文入库。同一旧定位的不同摘录保留独立片段，不能跨片段拼成新的连续引文。`public_document_v1`可读取明确发现的公开文件URL，不把历史PDF钉成永远最新的默认报告。

本轮实网取得：

- 四个默认来源返回可用响应；新浪C类资料概要有完整静态正文，可窄提取经理及公开费率。公开管理/托管/C类服务费为净值内费用，不再重复从预演收益扣除；这些不是支付宝渠道条件。
- 已实际发现的[9月C类概要PDF](https://pdf.dfcfw.com/pdf/H2_AN202609211829693880_1.pdf)与[2026中报PDF](https://pdf.dfcfw.com/pdf/H2_AN202608281828588412_1.pdf)本轮请求200并按页提取成功。中报窄解析可支持2026-06-30权益占NAV 81.60%、NVIDIA占NAV 9.38%，只覆盖该发行人，不冒称全组合。已有同文档更完整披露不被新窄解析覆盖。
- NVIDIA公司发布及韩国部门政策正文已取得；扩产计划与政策规则仍须通过经营、价格及披露持仓论证，不自动变成利好分数。
- **10月26日第三季度报告公告详情仍仅“详见附件”，没有取得可用报告正文。**上述六月中报不是Q3；不把公告日期当持仓所属日。
- 免费发现检索的Bing匿名探测返回302后停止，没有跟随或换入口恢复，尚无通过实网验收的通用免费检索路线。动态已验证发布列表与实际附件仍可使用，不把这项缺口写成已接通搜索。

真实收据与窄解析记录见`runs/artifacts/research-loop-20261027-OeXBjvye/materials/`。本轮也通过既有公开导入接口把公共研究基础与候选安装原8501工作区，引用重建为目标库实际定位；未复制整库或个人库，旧研究日期仍2026-10-26，详见同目录上级`baseline-installed.json`。

`Workspace.update_research`只在用户明确更新时调用薄`research_runner`与所选协议客户端。输入为公开基金、正文、已有复核记录、程序计算和候选比较，不含个人持仓金额。输出使用显式payload v2，区分事实/计划/假设/模型判断，分别论证新买、持有、追加，按实际引用解释变化；`api_generated_research`来源与人工复核v1分开。最终研究及采用摘录/计算沿用既有证据表保存，不建全文仓库或第二任务数据库。未配置、API失败或校验失败保留旧研究原日期；尚未有真实API配置和实际调用验收，协议模拟与合成测试不能替代它，也不能证明投资有效。

## 会话助手实际研究（2026-10-28）

018147已实际读到[虹点基金转载的Q3正文](https://hongdianfund.com/FundAnnouncement/Detail/539002/0000000000001sjh5r)：报告期7月1日—9月30日，送出10月26日；不再把10月27日的附件壳失败当作今天仍不可研究的结论。股票76.46%为净资产分母，74.62%为总资产分母。采用五大持仓作明确partial画像，四家36.63%、加闪迪44.30%；未把期货名义合约市值混入股票权重。两项科技指数多头名义合约市值12.4936亿元/合计净资产118.4399亿元≈10.55%，不是公允价值比例、实际股票仓位或精确净杠杆。

[易方达财富C页](https://www.efundcf.com.cn/fund/018/018147.shtml)与[A页](https://www.efundcf.com.cn/fund/539/539002.shtml)用于独立交叉核对证券身份及公开费率，不叫支付宝费率。100055采用[富国官方Q3 PDF](https://www.fullgoal.com.cn/wbs-file/fund_report/20261027/CN_50100000_100055_FB030030_20260024.pdf)：权益91.20%、仍押AI基础设施、期末无金融衍生品，与018147不是同授权或低科技风险替代。

公司研究使用实际取得的NVIDIA FY27Q2及现金流、台积电Q3独立报道、SK10月28日初步Q3、三星Q2实际/Q3指引与匹配价格。三家价格/最近单季EPS机械年化覆盖27.71%九月底披露净值，不当TTM/预测PE。美国财政部10月27日十年期4.99%只作美元市场折现参照，不当人民币现金回报。

最终v2以`interactive_assistant`/`assistant_generated_research`追加原公开工作区，URI为`rmb-research://assistant/...`，与API及旧作者笔记身份分开，非人工复核。64个采用引用、16项事实/判断、6组程序计算与4条采用经济记录均保留真实时间；只存研究采用的片段，不导入全文。产物位于`runs/artifacts/decision-research-20261028/`。应用真实API请求仍为0，未完成真实API互操作验收；研究判断本身则已经完成，不能再用API未配置代替研究。

新增403目标及未采用材料见`issuers/source-checks.json`；旧隔离TSMC文件未恢复。基金代理发生一次临时目录停止边界违规，已如实报告并停止，PDF未隐删；该操作不是获取授权或成功恢复受限网站的依据。

## CSV 导入

入口为：

```python
load_csv(path, product, observed_at, kind) -> list[Fact]
```

CSV 是不可执行、不可信文本；实现只使用 `csv` 标准库，不使用 `eval`/`exec`。文件限制为 8 MiB、50,000 行、每个单元格 16,384 个字符，要求 UTF-8、无重复表头、无未知字段、同类事实单一严格 schema。`NaN`、`Infinity`、指数溢出、布尔冒充数字、未来事件、冲突重复日期、错误产品和超范围数据均拒绝整份导入。

支持的核心表头如下（除列出的必需/可选字段外还可有 `product_id`、`symbol`、`source`、`source_url`、`observed_at`、`available_at`、`published_at`、`metadata` provenance 列）：

| kind | 必需列 | 可选列/约束 |
| --- | --- | --- |
| `bar` | `event_date,open,high,low,close,adjustment,currency` | `volume,volume_unit`；必须 `adjustment=raw`、`currency=CNY`，有 volume 时必须明确 `volume_unit=share` |
| `nav` | `event_date,unit_nav,currency` | `accumulated_nav`；单位净值必须为正 |
| `calendar` | `event_date,is_open` | `exchange=SSE` 或 `SZSE`；布尔值只接受 `true/false` |
| `announcement` | `event_date,title` | `text,url`；始终标记未核验 |
| `status` | `event_date,purchase_allowed,redeem_allowed` | `reason`；布尔值只接受 `true/false` |
| `dividend` | `event_date,record_date,ex_date,pay_date,cash_per_unit,currency` | 需要 `event_date=ex_date`，日期顺序必须正确 |
| `split` | `event_date,record_date,ex_date,pay_date,ratio` | 日期顺序必须正确；不能用单一折算日猜精确权益 |

CSV 中任何 `observed_at`、`available_at`、`published_at` 只保留在 `payload.metadata.csv_provenance`，不能把过去时间冒充本次抓取，也不能让未来时间变成 Fact 的可用时间。Fact 的 `observed_at` 与 `available_at` 始终是调用方传入的真实观察时间（传 `None` 时取当前带时区时间），`quality` 为 `unverified`。演示数据须由上层另行显式导入，不由适配器生成。
