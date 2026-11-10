# token-usage

一个**插件式**的 LLM token 用量与成本追踪 CLI。用纯 Python 标准库实现，无第三方运行时依赖。

## 快速开始

```bash
python3 -m token_usage.cli --help          # 查看所有命令
python3 -m token_usage.cli sample --days 14 --per-day 6 --project demo
python3 -m token_usage.cli report --group model
python3 -m token_usage.cli forecast --days 30
python3 -m token_usage.cli budget --action set --limit 5 --period monthly
```

运行测试：

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

## 架构

```
token_usage/
  models.py     数据模型：UsageRecord / ModelPricing / Budget（dataclass，可双向序列化）
  pricing.py    价格目录 PricingCatalog + 成本计算（区分缓存/非缓存输入）
  storage.py    JSONL + JSON 持久化（append-only，原子重写）
  analytics.py  纯函数聚合：filter / totals / group_by / 预测 / 预算评估
  insights.py   派生洞察：异常检测 / 周期对比 / 模型效率 / 集中度 / 浪费
  patterns.py   时间模式：小时·星期分布 / 移动平均 / 趋势 / 突发性
  optimize.py   成本优化：模型替换 / 缓存机会 / 重复提示 / 量化节省
  attribution.py 成本归属：规则匹配 / 分摊 / 覆盖率 / 发票 / 对账
  drivers.py    变化归因：三因子分解 / 模型贡献 / 结构变化 / 单价变动
  simulation.py 概率模拟：零填充日序列 / bootstrap / 分位数 / 超支概率
  quality.py    数据质量评分：10 项检查 / 分值归因 / 修复命令 / 整改排序
  peers.py      横向对标：指标归一化 / 排名 / 离散度 / 重定价 / 离群检测
  registry.py   插件契约与注册表（核心与命令之间的唯一接缝）
  plugins/      内置命令，每个文件一个 Plugin
  cli.py        通用参数解析入口（不认识任何具体命令的选项）
```

数据默认存放在 `./.token-usage/`（可用 `--home` 或 `TOKEN_USAGE_HOME` 覆盖）。

## 命令

| 命令 | 作用 |
|---|---|
| `record` | 追加一条用量记录（写路径） |
| `list` | 列表展示记录 |
| `report` | 按 model/project/tag/day/week/month 聚合，带 ASCII 条形图 |
| `stats` | 描述性统计（min/median/max、最忙一天） |
| `forecast` | 基于历史的最小二乘花费预测 |
| `budget` | 设定/检查花费预算（超支返回退出码 1） |
| `pricing` | 查看/编辑模型单价 |
| `export` | 导出 CSV / JSON |
| `sample` | 生成确定性的演示数据 |
| `import` | 导入厂商用量导出（OpenAI / Anthropic / 通用 JSON·CSV，自动识别格式） |
| `compare` | 当前周期 vs 上一周期（等长窗口，月中对比也公平） |
| `anomaly` | 按 z-score 标记花费异常日（spike / dip） |
| `tags` | 标签清单与各标签花费，并提示未打标签的记录 |
| `doctor` | 一次性全量体检：完整性、集中度、模型效率、浪费、异常、预算 |
| `dedupe` | 去除重复记录（默认 dry-run，`--apply` 才落盘） |
| `prune` | 按条件删除记录（无参数时拒绝执行，防误删） |
| `edit` | 按索引修正单条记录，改动前展示 diff |
| `merge` | 合并另一个 workspace 或归档到当前库 |
| `archive` | 把旧记录移入归档 / 恢复 / 删除归档 |
| `config` | 查看/编辑工作区默认值（如默认 project、group） |
| `preset` | 保存/回放命名选项组合，长命令变短命令 |
| `alert` | 声明式告警规则：定义条件、检查、可作 CI 门禁 |
| `plan` | 期末花费预测 + 预算结论 + 每日可用额度 + what-if 情景 |
| `quota` | 按历史占比把预算分配到各模型/项目，支持保底额度 |
| `pattern` | 时间模式分析：周几/几点最贵、趋势、周末对比、连续段、突发性 |
| `optimize` | 量化成本优化建议：模型替换、缓存机会、重复提示、输出浪费、小调用批处理 |
| `chargeback` | 成本归属：按规则把花费分摊到团队/成本中心，含覆盖率、发票、对账 |
| `drivers` | 花费变化归因：把涨跌拆成 volume / mix / unit_price / interaction 四项 |
| `simulate` | 期末花费概率模拟：分位数区间、超支概率、两种方法交叉验证 |
| `quality` | 数据质量打分（0~100）：10 项检查、扣分可归因、给出修复命令、按严重度排序整改 |
| `peers` | 多工作区横向对标：先归一化再比较，含排名、离散度、重定价上限、离群检测 |

## 迭代记录

**第 1 次**：核心层 + 9 个基础命令（record/list/report/stats/forecast/budget/pricing/export/sample）。

**第 2 次**（+980 行）：新增数据导入与洞察能力，**未改动任何既有核心逻辑**。

- 新增 `ingest.py`：把厂商导出解析为 `UsageRecord`，三种格式 + 自动识别 + 容错（坏行跳过并计数，不中断）。
- 新增 `insights.py`：异常检测、周期对比、模型单位经济性、花费集中度、浪费估算。
- 新增 5 个插件：`import` / `compare` / `anomaly` / `tags` / `doctor`。
- 唯一改动是 `plugins/__init__.py` 的 `_BUILTIN_MODULES` 追加 5 行 —— 这正是设计好的扩展点。

**第 3 次**（+1020 行）：新增数据生命周期治理，**未改动任何既有核心逻辑**。

- 新增 `maintenance.py`：去重指纹、条件清理、字段编辑（重建校验）、多源合并、完整性扫描。
- `storage.py` 追加归档读写（`write_archive` / `read_archive` / `list_archives` / `delete_archive`），复用既有的「写临时文件再 rename」原子模式。
- 新增 5 个插件：`dedupe` / `prune` / `edit` / `merge` / `archive`。
- **安全设计**：所有破坏性操作默认 dry-run，必须显式 `--apply` 才落盘；`prune` 无任何条件时直接拒绝执行。

**第 4 次**（+816 行）：新增配置与预设能力。

- 新增 `config.py`：默认值校验（类型 + 枚举白名单）、预设 CRUD、容错反序列化（坏配置行跳过而非报错）。
- `storage.py` 恢复 `load_config` / `save_config`；`registry.PluginContext` 增加惰性 `config` 属性。
- `cli.py` 新增 `_resolve_options`：展开 `--preset` 并合并默认值。
- 新增 2 个插件：`config` / `preset`。
- **优先级设计**：配置默认值 < preset < 显式 CLI 参数。显式参数永远最高，所以一次性覆盖不会被配置挡住。

**第 5 次**（+948 行）：新增声明式告警规则引擎。

- 新增 `alerts.py`：`Rule` 数据模型（7 种指标 × 4 种比较符 × 5 种维度）、评估引擎、严重度排序、CI 退出码映射。
- `storage.py` 追加 `load_alerts` / `save_alerts`。
- 新增 1 个插件：`alert`（规则 CRUD + `check`）。
- **关键设计**：规则是**数据不是代码**，存于 `alerts.json` 可手改；无数据时状态为 `unknown` 而非 `ok`，避免数据缺口伪装成通过。

**第 6 次**（+853 行）：新增前瞻性预算规划。

- 新增 `planning.py`：周期边界计算、期末预测、预算结论（on_track/at_risk/exceeded）、每日可用额度、配额分配、what-if 情景模拟。
- 新增 2 个插件：`plan` / `quota`。
- **关键设计**：`days_elapsed + days_remaining` 恒等于周期长度（今天只算一次），否则每日额度会被高估；预测用周期内日均而非回归，因为单周期内天数太少，回归不如均值稳定。

**第 7 次**（+915 行）：新增时间模式分析。

- 新增 `patterns.py`：小时/星期分布（零填充固定形状）、移动平均、趋势（斜率 + r²）、连续段、突发性（变异系数）、季节性强度、周末对比。
- 新增 1 个插件：`pattern`（7 种视图）。
- **关键设计**：
  - 小时桶是 **UTC**，不做时区偏移——不知道用户时区时，错误的偏移比诚实的 UTC 标签更糟。
  - 趋势同时给出 `r_squared`：斜率陡但拟合差时明确告警「这是噪声不是趋势」。
  - 周末对比用**日均**而非总额：一个月约 22 个工作日 vs 8 个周末日，比总额永远偏向工作日。

**第 8 次**（+929 行）：新增量化成本优化建议引擎。

- 新增 `optimize.py`：把 `insights` 的「观察」转成「可执行建议 + 量化节省」——同档模型替换（`--tier-drop` 控制可降几档）、从未命中缓存的模型、重复提示形状、低输出浪费、小调用批处理。
- 新增 1 个插件：`optimize`（6 种视图，别名 `optimise` / `savings`）。
- **关键设计**：
  - 每条建议的金额都由**重新计价真实观测到的 token** 得出，不是估算；无法诚实计算的（批处理）返回 `savings_usd: None` 而非编造数字。
  - 建议之间会**重叠**（缓存与替换作用于同一批 token），因此 `total_savings` 按模型取最大值而非求和，避免出现「节省 150% 花费」的荒谬结论。
  - 嵌入模型（`text-embedding-*`）输出单价为 0，永远不能替代生成模型，替换候选会显式排除它们。

**第 9 次**（本次，+1035 行）：新增成本归属与分摊（chargeback）。

- 新增 `attribution.py`：`OwnerRule` 规则模型（project/model/tag/note 四类谓词，**首个匹配生效**）、按权重分摊、覆盖率统计、孤儿记录、规则建议、单主体发票、独立对账。
- `storage.py` 追加 `load_owners` / `save_owners`（`owners.json`）。
- 新增 1 个插件：`chargeback`（7 种视图 + 规则增删，别名 `owners` / `attribution`）。
- **关键设计**：
  - 未匹配的记录**不丢弃**，而是作为 `unattributed` 一等公民上报。一份悄悄藏起 30% 花费的账单，比没有账单更糟。
  - `coverage`（覆盖率）放在摘要第一屏：先看这个数，才知道后面的数字能不能信。
  - `reconcile` 独立重算总额，验证「已归属 + 未归属 = 真实总额」。分摊权重或舍入出错时会以 `drift` 数字暴露，而不是等到季度结账才被人发现。
  - 规则文件里写错字段名（如 `projcet`）会**报错而非静默匹配空**——静默失败会伪装成「没有花费」。
  - `--action add` 会立即回显该规则命中了多少条记录；命中 0 条时明确告警，避免用户误以为覆盖率提升了。

**第 10 次**（+907 行）：新增花费变化归因。

- 新增 `drivers.py`：把周期涨跌精确拆成 **volume / mix / unit_price / interaction** 四项，各项恒等于观测差额；另有按模型的 volume/price 拆分、token 口径的结构变化、混合单价的独立视图。
- 新增 1 个插件：`drivers`（4 种视图，别名 `driver` / `why`）。
- **关键设计**：
  - `compare` 只说「涨了多少」，本模块回答「为什么涨」——是被涨价了，还是活多了？这两种情况需要完全相反的应对，单一差额无法区分。
  - 四个因子**相加恒等于观测差额**，`--view summary` 直接把这道校验打印出来，让算术可验证而不是靠信任。
  - 交互项单独保留，不摊进其他三项：随意分配会让报告看起来比数据本身更整洁。
  - 单价是**混合单价**（每 token 成本），缓存命中率或输入输出比变化都会让它移动——这是有效信息，因此作为独立因子报告而非隐藏。
  - 只在单边出现的模型没有可观测单价，继承另一侧价格并标记 `assumed_price`；这个假设写在输出里，不静默套用。
  - **无基线时不假装分解**：一侧为空时恒等式会把全部差额推进 `interaction`，算术正确但读起来像「量和价一起动了」。这种情况标记 `no_baseline`，直接说「无同期基线可对比」并给出换窗口的建议。

**第 11 次**（本次，+873 行）：新增期末花费概率模拟。

- 新增 `simulation.py`：**零填充**日序列、`bootstrap` 与 `normal` 两种重采样方法、线性插值分位数（P5~P95）、超支概率、所需削减额、双方法交叉验证。
- 新增 1 个插件：`simulate`（4 种视图，别名 `sim` / `risk`）。
- **关键设计**：
  - `plan` 只给一个点估计，回答不了「我有多少概率超支」。本模块给分布：`P50` 是典型值，`P90`/`P95` 是「能坏到什么程度」。
  - 日序列必须**零填充**：没有调用的那天是真实的零花费日，丢掉它会把均值抬高、把离散度压小。
  - `bootstrap` 无分布假设但只能产出历史区间内的值，趋势上涨时会低估风险；`normal` 能超出历史区间，却会抽出负花费。两种方法都实现并**报告分歧**——分歧大说明结论取决于假设而非数据。
  - `normal` 抽出的负值**钳制到零并计数上报**：一个悄悄产出不可能数值的模拟，不该在不知情的情况下被信任。
  - 无 `--limit` 时 `exceed_probability` 返回 `None` 而非 `0`——「0% 概率超支」和「没有预算可比」是两件事。
  - 结果**默认固定种子**，同样的数据和参数永远得到同样的 P90，报告里的数字可被复现和追问。
  - 削减超过 50% 时明确说「这是能力变更，不是调优」——不把一个不可能达成的目标包装成可执行数字。

**第 12 次**（本次，+994 行）：新增数据质量评分与整改清单。

- 新增 `quality.py`：10 项检查（重复、未来时间戳、零 token、未定价模型、成本钉值漂移、未打标签、默认项目、异常 token 量、日期断档、超期保留），每项按「影响比例 × 严重度上限」扣分，汇总为 0~100 分与等级。
- 新增 1 个插件：`quality`（4 种视图，别名 `qa` / `audit`）。
- **关键设计**：
  - `doctor` 只回答「有没有明显坏了」，本模块回答「这份数据能不能拿去汇报、最该修哪一项、用哪条命令修」。每一分扣减都能追溯到唯一一项检查，分数永远可以被展开成理由。
  - **严重度按后果定，不按记录数定**：一条未来日期记录会污染所有周期统计，一千条未打标签只是少一个报表维度，两者不能同分。
  - **每项扣分有上限**：否则一次「十万行全无标签」的导入就能把分数压到 0，把其他所有问题藏在后面。
  - **失败永不免费**：按比例扣分时，极小占比的严重缺陷会四舍五入成 `-0.0`，看起来像「不用管」。所有失败项保底扣 `MIN_PENALTY` 分。
  - **无法评估要说明**：记录数不足 10 条时不做离群检验，输出 `not evaluated: needs 10 records`，而不是静默通过、暗示「已检查且干净」。
  - 离群检测用**中位数 + MAD** 而非均值 + 标准差：一条 9 亿 token 的脏数据会把均值和标准差一起抬高，从而把自己伪装成正常值——均值法恰好漏掉它要抓的那条记录。
  - `--view checks` 过滤展示时明确提示「分数仍按全部检查计算」：隐藏检查不能让分数变好看，否则分数就没有意义。
  - 退出码分级：默认仅在存在 critical 或分数低于 60 时返回 1；`--strict` 任何失败项即返回 1，供 CI 门禁使用。

**第 13 次**（本次，+1030 行）：新增多工作区横向对标。

- 新增 `peers.py`：把多个 workspace 的用量**先归一化再比较**——`cost_per_1m`、`cost_per_call`、`output_ratio`、`cache_ratio` 四个效率指标，另有排名（带并列名次与百分位）、离散度（min/median/max、max/min 倍数、变异系数）、按最优单价重定价、离群检测。
- 新增 1 个插件：`peers`（6 种视图，别名 `bench` / `benchmark`），`--from a,b,c` 逗号分隔，`--include-self` 把当前工作区作为 `self` 参与对比。**只读**，不写入、不合并、不移动任何数据。
- **关键设计**：
  - 总额永远只说明「谁用的 token 最多」，回答不了「我们是贵还是只是忙」。所以一切比较都基于归一化指标，而不是原始金额。
  - **样本太小的 peer 不参与对标**：一次侥幸命中缓存的调用不该定义「业界最优」。低于 token 阈值的 peer 仍会列出但标记 `eligible: False`。
  - **不像的工作负载被标记，而不是静默比较**：嵌入型 peer 的输出占比接近 0、单价天然更低，把它排在对话型 peer 前面是分类错误。输出占比差距超阈值时明确告警「不是在干同一种活」。
  - **重定价节省是算术上限而非可执行目标**：`savings_vs_best` 用 peer 自己的 token 按最优单价重新计价，算术精确，但前提（该工作负载能以那个单价被服务）写在 `note` 里。实测一个对话型 peer 与嵌入型 peer 对比会算出「省 99.7%」——这个数字必须带警告出现，`savings` 视图已单独复述该前提，而不只是在摘要里说。
  - 离群检测用**中位数 + MAD**；当超过半数 peer 同价（多个团队用同一模型的常见情形）MAD 为 0，此时回退到比值检验并标明所用规则，而不是报告「没有离群」——那会把唯一值得查的 peer 藏起来。

## 如何迭代（第二条要求）

架构的唯一接缝是 `registry.Plugin`。新增功能**无需修改核心**，只需三步：

1. 在 `token_usage/plugins/` 新建一个文件，例如 `latency.py`；
2. 定义一个 `Plugin` 子类并调用 `register()`：

```python
from token_usage.registry import Plugin, PluginContext, register

class LatencyPlugin(Plugin):
    name = "latency"
    help = "Show a latency-style summary."

    def run(self, ctx: PluginContext) -> int:
        ctx.out(f"累计 {len(ctx.records())} 条记录")
        return 0

register(LatencyPlugin())
```

3. 把模块名追加到 `token_usage/plugins/__init__.py` 的 `_BUILTIN_MODULES`。

也可以完全不动仓库：把 `.py` 文件丢进 `<home>/plugins/`，或用
`TOKEN_USAGE_PLUGIN_MODULES` 指定可导入的模块名，CLI 启动时自动发现。

因此"明天在今天的代码上继续生成"= 新增一个插件文件 + 注册一行，核心层、存储格式、其他命令都不受影响。
