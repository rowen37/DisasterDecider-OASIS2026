# EXTENDING — 新灾种插件接入指南

本项目是一个**灾种可扩展的空间决策 Agent**（disaster decider）。洪水是
当前唯一注册的参考实现；新增灾种（野火 / 台风 / …）不需要改动任何
共享层文件，只写一个插件文件。

## 架构契约（四层 + 插件）

```
app/
├── main.py              # 入口：路由 → create_skill 工厂 → 报告（零灾种知识）
├── agent.py             # 综合层：通用反幻觉规则；灾种规则由插件注入
├── skills/
│   ├── registry.py      # ★ 插件注册表（关键词路由 / 工厂 / 规则提取）
│   ├── flood_skill.py   # ★ 参考插件：FloodSkill（~2000 行完整实现）
│   └── *_skill.py       # 原子 Skill（可复用的数据获取单元）
├── engine/              # 纯计算层：<hazard>_<role>_engine 或共享 <role>_engine
├── ops/                 # 基础设施层：MCP 生命周期 / 超时重试 / 指标
└── social_good.py       # 公平指标（VWUN / equity gap，全灾种通用）
```

灾种知识只允许存在于两处：**插件 Skill** 与 **插件专属 MCP server**。
共享层（main / agent / registry / verification）不 import 任何灾种模块。

## 五步接入新灾种

以野火为例：

### 1. 写编排 Skill 并注册

```python
# app/skills/wildfire_skill.py
from .registry import register_skill

@register_skill(
    hazard="wildfire",
    keywords=("wildfire", "blaze", "burn scar"),        # 强关键词
    weak_keywords=("burn area",),                        # 弱关键词
    unsupported_hazard_words=("earthquake", "flood"),    # 拒绝误路由
)
class WildfireSkill:
    name = "wildfire_impact_assessment"

    OUTPUT_RULES = """
WILDFIRE-SPECIFIC RULES
...（综合层注入的禁语/允许表述，洪水版见 FloodSkill.OUTPUT_RULES）
"""

    @staticmethod
    def OUTPUT_VALIDATOR(output: str, packet: dict) -> None:
        """LLM 输出禁语校验（如未被验证的火势蔓延断言）。"""
        ...

    def __init__(self, state, verifier, hitl, mcp, logger):
        ...  # 与 FloodSkill 相同的构造签名（工厂约定）

    async def run(self, target, station_id=None, raw_task=None,
                  overrides=None) -> SkillResult:
        ...
```

### 2. 在 `app/skills/__init__.py` 加一行 import

```python
from .wildfire_skill import WildfireSkill   # 注册在 import 时完成
```

### 3. 在 `app/models.py` 的 `HazardType` 加一个字面量

```python
HazardType = Literal["flood", "wildfire"]
```

（API schema 用；路由真值以注册表为准。）

### 4. （可选）灾种专属计算放 `app/engine/wildfire_*_engine.py`

纯函数、无 I/O，命名遵循 `<hazard>_<role>_engine` 约定。

### 5. （可选）数据源做成 MCP server 并注册进 `config/mcp.json`

参考 `mcp_servers/gee_flood_extent.py`（结构化 status/error_code、
来源与元数据诚实标注）。返回契约见下。

接入完成 —— `main.py` 的工厂分发、关键词路由、综合层规则注入与
输出校验**自动生效**，无需改动。

## 插件必须遵守的输出契约

1. **SkillResult**：`status / summary / evidence / spatial_objects /
   validation_issues / next_actions`（`app/models.py`）。
2. **证据不虚构**：缺失数据走 `data_gaps`/`substitutions` 记账
   （参考 `flood_risk_engine.compute_decision_indices`），绝不折叠成 0。
3. **HITL**：否决型检查点必须捕获 `ask_async` 返回值并真正阻断
   后续流程（参考 `flood_skill.py` 的 `human_denied_mobilization`）。
4. **公平账本（可选但推荐）**：灾种若有逐单元暴露数据（人口×灾面×
   覆盖），调用 `social_good.compute_vulnerability_weighted_unmet_need`
   与 `compute_equity_gap`，结果挂到优化输出的 `equity_ledger`。

## 回归验证

```bash
python -m pytest tests/ -v          # 全部离线
python -m pytest tests/test_demo_flood_run.py -v -s   # 端到端 demo
```

`tests/test_demo_flood_run.py` 是新灾种的验收模板：把 FakeMCP 的
fixture 换成你的数据源响应即可复用整套断言结构。

## 已注册灾种

| hazard | 插件 | 能力 |
|---|---|---|
| flood | `FloodSkill` | 站点验证 → 多源融合 → GEE 淹没 → 面积加权暴露 → SVI → Pareto 资源优化 + 公平账本 → CDRI/EPS |

（地震实现曾存在于历史提交，已按"单灾种极致 + 通用接口"决策移除；
接入方式即本文件所述，无需恢复旧代码。）
