# EXTENDING — 新灾种插件接入指南

本项目是一个**灾种可扩展的空间决策 Agent**（disaster decider）。洪水是
当前唯一注册的参考实现；新增灾种（野火 / 台风 / …）不需要在共享路由、
综合层或服务入口中增加灾种分支，但仍需增加一个包导入和一个 API 类型值。

## 架构契约（四层 + 插件）

```
app/
├── main.py              # 入口：路由 → create_skill 工厂 → 报告（零灾种知识）
├── agent.py             # 综合层：通用反幻觉规则；灾种规则由插件注入
├── skills/
│   ├── registry.py      # ★ 插件注册表（关键词路由 / 工厂 / 规则提取）
│   ├── flood_skill.py   # ★ 参考插件：FloodSkill（当前约 4000 行完整实现）
│   └── *_skill.py       # 原子 Skill（可复用的数据获取单元）
├── engine/              # 纯计算层：<hazard>_<role>_engine 或共享 <role>_engine
├── ops/                 # 基础设施层：MCP 生命周期 / 超时重试 / 指标
└── social_good.py       # 公平指标（VWUN / equity gap，全灾种通用）
```

灾种算法和输出规则只放在两处：**插件 Skill** 与 **插件专属 MCP
server**。共享层（main / agent / registry / verification）不写灾种分支；
`skills/__init__.py` 仅负责导入插件以触发注册，`models.HazardType` 仅维护
API 可接受的字符串集合。

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

接入完成 —— 完成第 2、3 步的注册/schema 小改动后，`main.py` 的工厂
分发、关键词路由、综合层规则注入与输出校验自动生效；无需在这些共享
流程中添加 `if/elif` 灾种分支。

## 插件必须遵守的输出契约

1. **SkillResult**：`status / summary / evidence / spatial_objects /
   validation_issues / next_actions`（`app/models.py`）。
2. **证据不虚构**：缺失数据走 `data_gaps`/`substitutions` 记账
   （参考 `RiskEngine.compute_decision_indices`），绝不折叠成 0。
3. **HITL**：否决型检查点必须捕获 `ask_async` 返回值并真正阻断
   后续流程（参考 `flood_skill.py` 的 `human_denied_mobilization`）。
4. **公平账本（可选但推荐）**：灾种若有逐单元暴露数据（人口×灾面×
   覆盖），调用 `social_good.compute_vulnerability_weighted_unmet_need`、
   `compute_equity_gap` 与阈值无关的
   `compute_coverage_concentration_index_or_none`，结果挂到优化输出的
   `equity_ledger`。
5. **容量约束分配**：逐单元暴露数据应转换为 `demand_records`，交给
   `SupplyDemandEngine` 分配到带坐标和容量的设施；容量来源不可靠时必须
   保留 `capacity_assumed`，不可把估算容量当成已验证事实。
6. **设施筛选、路线与分配不得混为一种距离**：当前洪水插件先用
   30 km/h × 30 分钟得到 15 km 的设施筛选服务半径；首次查询成功但为空
   时才扩展到 60 分钟 / 30 km。该圆形范围只是跨地点统一的规划代理，
   不是路网等时圈。地图救援路线另用 OSM 路网最短路径，时间为路线长度
   ÷ 30 km/h；容量分配则由 `SupplyDemandEngine` 使用需求点到设施的直线
   距离。15/30 km 只控制 GIS 地图路线的 POI 层；资源优化候选由独立的
   `get_available_resources` 获取；版本化环境模板显式传入 30 km，只有省略
   `radius_km` 时服务端才回退为 50 km。其候选计划的
   `response_time` 目标仍采用 60 km/h 直线距离启发式。新灾种如果采用
   不同口径，必须分别记录筛选范围、候选发现范围、路线算法、速度假设和
   数据缺口，不得声称实时交通或精确清空时间。
7. **社区要求必须有来源**：SVI、公平指标和操作员选择不等于社区意见。
   无障碍、多语言、宠物安置、接送等要求只能作为带来源、可编辑、可审计
   的输入进入；不得在代码里硬编码为某个社区的偏好，也不得从 SVI 推断。
   当前洪水插件已支持操作员输入带来源的事件需求，并依据
   `config/community_facilities.json` 和显式 OSM 标签将方案标记为
   confirmed / unknown / unmet。这是可审计的建议方案，暂不自动
   改写主推荐；没有实时社区数据接口。

## 回归验证

```bash
uv run python -m pytest -q          # 全部离线
uv run python -m pytest tests/test_demo_flood_run.py -v -s   # 端到端 demo
```

`tests/test_demo_flood_run.py` 是新灾种的验收模板：把 FakeMCP 的
fixture 换成你的数据源响应即可复用整套断言结构。

测试数量以 `uv run python -m pytest -q` 的当次输出为准，避免文档中的手工
计数过期。其中设施服务区回归测试明确断言 15 km 首查、空结果后
30 km 回退，以及回退后路线产物存在。

## 已注册灾种

| hazard | 插件 | 能力 |
|---|---|---|
| flood | `FloodSkill` | 站点验证 → 多源融合 → GEE/建模淹没范围 → 面积加权暴露 → SVI → 15/30 km 设施服务区 + OSM 路线 → 容量约束 Pareto 资源优化 + 公平账本 → CDRI/EPS |

（地震实现曾存在于历史提交，已按"单灾种极致 + 通用接口"决策移除；
接入方式即本文件所述，无需恢复旧代码。）
