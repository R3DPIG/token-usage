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

## 迭代记录

**第 1 次**：核心层 + 9 个基础命令（record/list/report/stats/forecast/budget/pricing/export/sample）。

**第 2 次**（本次，+980 行）：新增数据导入与洞察能力，**未改动任何既有核心逻辑**。

- 新增 `ingest.py`：把厂商导出解析为 `UsageRecord`，三种格式 + 自动识别 + 容错（坏行跳过并计数，不中断）。
- 新增 `insights.py`：异常检测、周期对比、模型单位经济性、花费集中度、浪费估算。
- 新增 5 个插件：`import` / `compare` / `anomaly` / `tags` / `doctor`。
- 唯一改动是 `plugins/__init__.py` 的 `_BUILTIN_MODULES` 追加 5 行 —— 这正是设计好的扩展点。

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
