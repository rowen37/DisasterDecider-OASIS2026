# 洪水评估文本测试案例

三个可直接粘贴运行的测试案例（全部基于官方真实数据核对），覆盖新决策链的三条路径：

| 案例 | 事件 | 预期路径 |
|------|------|----------|
| TC-1 | Lodi NJ 2026-09-13（真实小洪水，卫星峰后无市内覆盖） | 卫星零市内重叠 → **建模回退** |
| TC-2 | Imelda 2019 Friendswood TX（项目保留验证集事件） | **卫星观测范围**（峰后 1 天有景） |
| TC-3 | Lodi NJ 2026-09-22（无洪水阴性对照） | 无触发 → **明确报告"无范围"**，不误报 |

运行方式：使用 web UI 的地点/测站/事件日期三栏，或调用
`POST /api/assess` 并分别传入 `query`、`station_id` 和 `event_date`。
CLI REPL 当前只接受 query，不提供独立的 `event_date` 参数。
`event_date` 早于今天 UTC 即自动进入历史回放模式（严格时间对齐）。

### 三个案例共用的设施与路线口径

- 设施候选以评估地点为中心，仅查 `hospital,shelter`。按统一规划速度
  30 km/h，先查 30 分钟 / 15 km；仅当该查询成功但结果为空时，扩展到
  60 分钟 / 30 km。
- 当前 POI 请求最多保留 10 个返回设施，路线选择其中直线距离最近的一项；
  因此只能证明“返回候选中的最近”，不能证明设施服务最合适。
- 15/30 km 是直线服务半径代理，不是真实道路等时圈，也不是实时交通。
- 找到设施后，救援路线才使用 OSM 路网；显示时间 = 路线长度 ÷ 30 km/h。
  首先尝试避开洪水多边形；若图被切断，可输出明确标记
  `route_avoids_flood = false` 的可达性参考路线。道路下载使用目标到设施的
  2 km 半宽走廊，基础图不连通时才扩为 4 km；两种路线复用同一张图。
- 资源分配中的 p95 距离又是另一口径：当前为暴露需求点到候选设施的
  直线分配距离。除非另行配置速度上下限，否则不输出转运时间范围。
- 本次 15/30 km 修改不控制资源优化器的候选发现。版本化环境模板为
  `get_available_resources` 显式传入 30 km；未传 `radius_km` 时服务端
  默认查 50 km。
  候选计划的 `response_time` 目标仍使用 60 km/h 直线距离启发式。测试时
  必须把它与 30 km/h 的地图救援路线分开记录。

---

## TC-1 Lodi, New Jersey — 2026-09-13（回退路径）

### 输入（可直接粘贴）

```
地点 (target):        Lodi, New Jersey
USGS 测站 (station):  01391500
事件日期 (event_date): 2026-09-13
```

可复制的 query：
`assess flooding near Lodi New Jersey using USGS station 01391500`

日期仍需单独填入 `event_date = 2026-09-13`；系统不从 query 正文提取日期。

### 官方基准数据（已核对）

| 项 | 值 | 来源 |
|----|----|----|
| 事件日窗口 | 2026-09-13T00:00Z .. 09-14T00:00Z（当地 09-12 20:00–09-13 20:00 EDT） | — |
| 峰值水位 | **6.79 ft @ 2026-09-13T14:15 EDT**（97 个样本） | USGS IV 00065 |
| 窗口末水位 | 4.67 ft @ 20:00 EDT（已回落到 action 之下） | USGS IV 00065 |
| NWPS 阈值 (LODN4) | action 5.0 / minor 5.5 / moderate 7.0 / major 8.0 ft | api.water.noaa.gov |
| 水文等级 | minor（6.79 ≥ 5.5） | 按官方阈值 |
| Sentinel-1 覆盖 | 事件窗 [09-13, 09-16] **0 景**；最近可用景 09-17T22:50Z（lag +4 天，需前向扩窗）；峰后影像裁剪到 Lodi 市界**零重叠** | GEE COPERNICUS/S1_GRD |

### 预期执行链

1. 历史模式（event_date < 今天）；warning/forecast/weather/precipitation
   四个 current-only 源被跳过并在 next_actions 披露。
2. `get_flood_observation(start_dt, end_dt)` 返回
   `observation_semantics = "window_peak"`，
   `water_level = 6.79`（**不是** 4.67），附完整 `window_summary`。
3. 佐证门：6.79 ≥ action 5.0 → `gauge_at_or_above_action_stage`（阈值已验证）→ 采纳。
4. GEE 前向扩窗找到 09-17 景（lag +4 ∈ [-1, +6]，时间门通过），
   30km AOI 检出 ~79 km²；但与 Lodi 市界交集为零 →
   `SAR_EXTENT_NO_INCITY_OVERLAP`（warning，已知单景对城市假阴性）。
5. 峰值触发成立 → **建模回退**：OSM 河网锚定测站取最近水道
   （Saddle River 干流 + Sprout Brook），超阈 1.79 ft 插值半径 ~0.74 km，
   裁剪到市界 ≈ 4.0 km²（约占城市 68% → 另发 `EXTENT_MODEL_OVERBROAD`）。
6. 下游 GIS（地图/受淹建筑/救援路径/暴露人口/公平账本）全部产出，
   全部标注 `extent_provenance = "modeled_stage_buffer"`。

### 预期输出要点

- summary 首句含 **"peaked at 6.79 ft"** 和窗口信息（不是 "reported 4.67 ft"）。
- validation_issues 应包含：`SAR_EXTENT_NO_INCITY_OVERLAP`、
  `EXTENT_FALLBACK_MODELED`、`EXTENT_MODEL_OVERBROAD`、
  `NWPS_STAGEFLOW_SKIPPED_HISTORICAL`。
- 合规的 09-17 前向扩窗景不应产生 `TIME_ALIGNMENT_MISMATCH`。
- `spatial_objects` 含 `flood_inundation_extent`，source 以
  **"Modeled stage-buffer extent"** 开头，confidence ≈ 0.4。
- CDRI：`hazard_basis = "water_severity"`（major-referenced 饱和映射，
  (6.79−5.0)/(8.0−5.0) = 0.597 → severity ≈ 0.373），范围只作空间上下文。
- next_actions 含 "MODELED stage-buffer" 替代说明。

### 通过判据

- [ ] `water_level = 6.79` 且 summary 含 "peaked at"
- [ ] 返回了洪水范围（`flood_inundation_extent` 存在且 provenance 为
      modeled_stage_buffer）——**本次重构前此查询返回空**
- [ ] `EXTENT_FALLBACK_MODELED` 在 issues 中
- [ ] 暴露人口、受影响设施数 > 0
- [ ] 设施搜索统计记录 15 km 首查；若首查为空，则记录 30 km 回退档
- [ ] 若生成路线，`travel_time_min` 与路线长度 ÷ 30 km/h 一致

### 已知边界（不算失败）

- 回退范围是保守上界（走廊全覆盖，无法分辨走廊内何处实际过水），
  已由 `EXTENT_MODEL_OVERBROAD` 披露。
- 时间对齐账本对 09-17 景的 offset = +4 天，历史模式严格窗口 [-1, +6]
  判定为 aligned；若 NWPS/上游时间戳格式变化导致误判 misaligned，
  系统仍会走回退路径，最终结果不变（这正是回退的意义）。

---

## TC-2 Imelda 2019 — Friendswood, Texas（卫星观测路径，已验证事件）

项目保留验证集事件（`config/validation_events.json`，
`role = held_out_validation`，USGS 官方核定数据）。

### 输入（可直接粘贴）

```
地点 (target):        Friendswood, Texas
USGS 测站 (station):  08077600
事件日期 (event_date): 2019-09-18
```

可复制的 query：
`assess flooding near Friendswood Texas using USGS station 08077600`

日期仍需单独填入 `event_date = 2019-09-18`；系统不从 query 正文提取日期。

### 官方基准数据（已核对）

| 项 | 值 | 来源 |
|----|----|----|
| 事件日窗口 | 2019-09-18T00:00Z .. 09-19T00:00Z（当地 09-17 19:00–09-18 19:00 CDT） | — |
| 官方核定峰值 | **11.64 ft @ 2019-09-18T11:15 CDT**（资格等级 A：USGS approved） | manifest + USGS IV 复核一致 |
| 窗口末水位 | 10.47 ft @ 19:00 CDT（仍在涨落高位） | USGS IV |
| NWPS 阈值 | action 7.0 / minor 12.0 / moderate 16.0 / major 21.0 ft | api.water.noaa.gov |
| 水文等级 | **action**（11.64 < 12.0 minor；逼近 minor）——manifest 期望值即 "action" | 官方阈值 |
| Sentinel-1 覆盖 | 事件窗内有景：09-18T20:18Z（**峰值后 4 小时**）与 09-19T08:23Z；基线景需回扩至 09-13 | GEE 实测 |
| GEE 检出（30km AOI） | ~33.3 km² / ~406 多边形；latest_post_scene = 2019-09-19T12:23Z（lag +1 天） | GEE 实测 |

### 预期执行链

1. 历史模式；current-only 源跳过并披露。
2. 峰值选择：`water_level = 11.64`，`window_peak` 语义，
   `window_summary.end_stage_ft = 10.47`。
3. 佐证门：11.64 ≥ action 7.0 → `gauge_at_or_above_action_stage`。
4. GEE：观测窗 [09-18, 09-21] 直接有景（无需前向扩窗），
   acquisition lag = +1 天 ∈ [-1, +6]，时间门通过；
   基线窗 [09-15, 09-18) 空自动回扩至 09-13 找到基线景。
5. **卫星足迹被采纳**：裁剪到 Friendswood 市界 → 市内洪水范围 →
   下游 GIS 全链产出，`extent_provenance = "satellite_sar"`。
6. CDRI hazard = water_severity（(11.64−7)/(21−7) = 0.331 → ≈ 0.249），
   范围作空间上下文；历史回放资源优化仍执行（训练/复盘场景）。

### 预期输出要点

- summary 含 "peaked at 11.64 ft"。
- `flood_inundation_extent` 的 source 为
  "Google Earth Engine / Sentinel-1 SAR"（或等价）——**观测范围，非建模**。
- `extent_provenance = "satellite_sar"`；
  `time_alignment.items` 中 satellite_sar 状态为 aligned（offset +1 天）。
- issues 不含 `EXTENT_FALLBACK_MODELED`（回退未触发）。
- 水文等级判读：按官方阈值落在 **action** 档（与 manifest 期望一致，
  距 minor 仅 0.36 ft——文案不得声称 "minor flooding"）。

### 通过判据

- [ ] `water_level = 11.64`（与 manifest 官方核定值一致——峰值提取回归锚点）
- [ ] 洪水范围为卫星观测（provenance = satellite_sar），面积 > 0
- [ ] 卫星 acquisition 时间 ∈ [09-17, 09-24]，时间门通过
- [ ] 等级判读为 action 档（不夸大为 minor/moderate）
- [ ] 暴露人口/受影响设施/救援路径全部产出
- [ ] 设施搜索统计明确记录实际使用的 15 km 或 30 km 档位

### 已知边界（不算失败）

- GEE 面积随基线景选择（回扩深度）和 -3dB 阈值浮动，
  33 km² 是 30km AOI 的值；裁剪到市界后显著缩小。断言"面积 > 0"
  而不是具体数值。
- 历史模式无 NWS 警报佐证（current-only），佐证完全来自水位峰值——
  这是设计行为。

---

## TC-3 Lodi, New Jersey — 2026-09-22（阴性对照：无洪水日）

### 输入（可直接粘贴）

```
地点 (target):        Lodi, New Jersey
USGS 测站 (station):  01391500
事件日期 (event_date): 2026-09-22
```

### 官方基准数据（已核对）

- 窗口内 97 个样本，峰值仅 **2.25 ft**（基流），远低于 action 5.0 ft。

### 预期行为

1. 峰值 2.25 < action 5.0 → 不触发（`extent_fallback_skipped:
   gauge_below_action_stage`）。
2. 若卫星返回"无检出/零面积"：**不生成洪水范围**，next_actions 明确
   "No flood extent was produced ... CDRI hazard is based on water-level
   ratio only"，不静默。
3. CDRI：water_severity = 0（(2.25−5.0) 负值截断为 0）→ hazard = 0 →
   CDRI = 0（"无险不成险"公理），标签 Low 且**不**带 degraded 后缀
   （若 SVI/暴露等组件缺失则带，属诚实披露）。
4. 资源优化：证据门（水位远低于行动阈值）预期阻断动员类输出
   ——宁可少动用，不得无险动员。

### 通过判据

- [ ] `water_level = 2.25`，summary 无 "peaked at ... reached action stage" 误导
- [ ] **无** `flood_inundation_extent` 空间对象
- [ ] **无** `EXTENT_FALLBACK_MODELED`（未触发 = 未误报）
- [ ] next_actions 显式说明无范围及原因
- [ ] CDRI = 0 或接近 0

### 已知边界（不算失败）

- 卫星对该日可能检出零星水面变化（湿土/植被误报）：佐证门不成立时
  检出只记为 `unverified_surface_water_candidate` 上下文层
  （`SAR_EXTENT_UNCORROBORATED` warning），不得进入地图与暴露计算。

---

## 附：三案例对决策链的覆盖矩阵

| 环节 | TC-1 Lodi 洪水 | TC-2 Imelda | TC-3 无洪水 |
|------|----------------|-------------|-------------|
| window_peak 峰值选择 | ✔（6.79 ≠ 末端 4.67） | ✔（11.64 官方核定） | ✔（2.25 拒绝触发） |
| 佐证门（水位主判据） | ✔ 触发 | ✔ 触发 | ✔ 不触发 |
| GEE 前向扩窗 | ✔（找 09-17 景） | —（窗内直接有景） | — |
| 卫星时间门 | ✔（+4 天边缘通过后零市内重叠） | ✔（+1 天） | — |
| 建模回退 | ✔ 接管 | — 未触发 | — 未触发 |
| 过宽披露 | ✔ OVERBROAD | — | — |
| 不误报 | — | — | ✔ |
| 下游 GIS/公平账本 | ✔（建模范围驱动） | ✔（观测范围驱动） | — |
| 设施服务区 15→30 km | ✔（有范围时检查） | ✔（有范围时检查） | — |

数据核对时间：2026-09-30（USGS IV、NWPS、GEE、Nominatim 全部实时拉取）。

---

## 补充的平衡验证样本

上述三个案例覆盖空间执行路径；正式水文验证集另外加入两个完全未参与
Harvey/Friendswood 设计的事件，使五个等级各有一个样本：

| 等级 | 事件 | 测站 | 官方峰值 | 验证层级 |
|------|------|------|----------|----------|
| below_action | Lodi 2026-09-22 | 01391500 | 2.25 ft | 水文分类 |
| action | Imelda / Friendswood 2019-09-18 | 08077600 | 11.64 ft | 完整 Agent |
| minor | Lodi 2026-09-13 | 01391500 | 6.79 ft | 水文分类 |
| moderate | Richmond 2020-11-14 | 02037500 | 18.33 ft | 水文分类 |
| major | Ida / Manville 2021-09-02 | 01400500 | 27.66 ft | 完整 Agent |

这里的“平衡”是按官方水文等级各一个样本，不是声称五个样本足以证明统计
泛化。当前完整 Agent 脚本只检查十项明确列出的运行状态与产物条件；范围
精度、CDRI 分档和转运时间精度仍需另有真实地面数据才能验证。这十项自动
验收检查也不包含 SVI 内容正确性、受灾建筑数量或救援路线是否存在；
本文件对这些项目的勾选属于额外的人工/文本案例验收，不能与十项脚本
检查混称。这些案例也没有提供社区需求，因此不验证社区适配。
当前系统可接收带来源的事件需求，并将设施/方案标记为 confirmed、
unknown 或 unmet；这些标记是建议信息，尚不自动改写主推荐。
