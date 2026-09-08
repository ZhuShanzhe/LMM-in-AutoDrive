# 三场景统一 VLA 最终提交测试报告

日期：2026-08-09
最终代码提交：`cb712ab`
最终权重：`universal-three-scene-sensor-policy-vla-v6` Stage-8

## 1. 交付结论

三个场景使用一套 `LightweightDecisionAdapter`、一份 `model.pt`、固定 `UnifiedSensorBatch`、同一个 `UniversalVLAController` 和同一套 FSM / 时序风险监督器 / Route PID。在线策略不读取场景、事件、指令 ID 或 CARLA actor 真值。

权重 SHA-256：

```text
53e949b37c84d6010ab45bfd473cb9d39a88cd89cd7729f55d3e9bb1baddaad3
```

部署配置 SHA-256：

```text
40164752c522779330a2a2f68a869968eaacb075eb409bc91813143a3ef9c39e
```

Hugging Face：`UNIC0RN-Zhu/universal-three-scene-sensor-policy-vla-v6`

## 2. 三场景结果

| 场景 | 距离/状态 | 指令/事件 | 碰撞 | 非法/受限车道 | fallback | 结论 |
|---|---|---|---:|---:|---:|---|
| 场景一 Town04 | 4977.011/4995 m，运行器 SUCCESS | 15/15，左转完成 | 0 | 0 | 0 | 通过 |
| 场景二 Town05 | 8000.913 m，route completed | 15/15，4/4 | 0 | 0 | 0 | 通过 |
| 场景三 Town05 雨夜 | 6000.404 m，route completed | 7/7 | 0 | 0 | 见启动说明 | 行车与事件通过 |

### 场景一

证据目录：`universal_v6_stage8_scene1_full5km_r30_20260809`

- 场景状态：`SUCCESS / route_completed_without_collision_or_illegal_lane_invasion`；
- 路线进度 4977.011/4995 m，实际轮迹里程 4790.869 m；
- 15 条文本指令全部发出，指定左转进入/退出路口并完成约 89.374° 航向变化；
- 碰撞 0，非法车道侵入 0，fallback 0；
- 5 次车道传感器事件均为可跨越虚线；
- 旧指标把转弯目标速度从巡航值切到 15 km/h 后的 5 个制动过渡帧误记为“超速”。代码已改为按 CARLA 道路限速统计，但按用户决定未重跑场景一；原始日志保留，不改写。

### 场景二

证据目录：`universal_v6_stage8_full8km_r19_20260809`

- 路线 8000.9126 m 全程完成；15 条指令全部播报且路线审计 mismatch 0；
- 4 个小事件全部 `RESOLVED`；70 辆交通车、21 名行人；
- 碰撞事件 0、接触样本 0、车道侵入 0、受限标线侵入 0；
- VLA 决策 10,972 次，fallback 0；
- 完整决策延迟 mean 22.150 ms、p95 33.344 ms；传感器到决策 p95 33.876 ms，120 ms 内比例 100%；
- 结构化日志模式下在线 VLA 传感器仍启用；100 m 内完整物理，远场使用 hybrid physics。

### 场景三

完整证据目录：`universal_v6_stage8_scene3_full6km_r32_20260809`
启动修复 smoke：`universal_v6_stage8_scene3_warmup_smoke_r33_20260809`

- 官方雨夜湿滑配置，路线 6000.404 m 完成；
- 7 个事件全部 `RESOLVED`：动态加塞、施工预警、锥桶收窄、施工区、行人横穿、受阻车道、施工区恢复；
- 碰撞 0，受限标线侵入 0，无效车道样本 0；51 次车道传感器事件均为可跨越虚线；
- 受阻车道先等待目标车道前 35 m / 后 20 m 的不安全间隙打开，再合法向左变道；
- 后向物理风险触发 `physical_rear_radar_acceleration_escape` 42 次，速度受道路与路线限速包络约束；
- 完整决策 22,268 次，传感器到决策 mean 24.473 ms、p95 39.063 ms、最大 75.581 ms，120 ms 内比例 100%。

r32 在车辆开始行驶前记录 1 次“无同步多视角 RGB”的安全驻车，导致旧严格摘要的 fallback 为 1。最终代码把该正常启动状态单独记为 `sensor_warmup_safe_hold_count`；r33 smoke 产生 133 次模型决策，fallback 0、预热驻车 1、0 碰撞、0 非法车道。项目按用户要求不再完整重跑 6 km，因此报告明确使用“r32 完整行为证据 + r33 启动链路修复证据”，不声称它们是同一次运行。

## 3. 前后向风险通用策略

- 前车或规划走廊内障碍有碰撞风险：制动/让行；只有目标车道合法且视觉安全时才变道；
- 后车加速逼近且 TTC/闭合速度危险：当前速度低于道路与路线限速时加速脱险；达到上限后只允许合法安全变道；
- 文本停车、前向紧急风险、实线/地图合法性和道路限速具有更高优先级；
- 该逻辑只读传感器、路线和文本语义，不读场景或事件 ID。

## 4. 模型独立测试与限制

独立测试集 4578 条：动作准确率 91.37%，宏平均 93.27%，紧急制动 77.37%，视觉风险总体 75.91%，high 风险 37.80%，速度 MAE 2.34 km/h，反事实准确率 85.52%。

high 风险视觉头仍是主要不足；当前可用性依赖时序确认、规划走廊物理雷达和可审计规则层。服务器上的 nuScenes 和 Waymo 资产未用于该 Stage-8 权重训练，提交材料不将其列为训练来源。

## 5. 测试状态

- 生命周期修复后完整测试：569 passed + 177 subtests；
- 最终传感器预热修复专项测试：92 passed；
- r33 启动 smoke：fallback 0，预热驻车 1，0 碰撞，0 非法车道。

日志包保存原始结构化 JSON/JSONL 与精简摘要，不包含视频、逐帧 RGB、训练数据或模型权重。权重由 Hugging Face 和最终 Docker 镜像提供。
