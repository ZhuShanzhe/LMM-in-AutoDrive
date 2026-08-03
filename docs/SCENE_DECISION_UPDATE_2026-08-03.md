# 场景 2/3 与 VLA 决策链路更新说明（2026-08-03）

## 1. 更新结论

本次更新聚焦仿真场景和决策模块，目标是减少固定规则/FSM 对复杂任务的限制，并为后续感知、语音、控制、评测模块留出稳定接口。

更新后的主决策链路为：

```text
语音/文本指令 ──> DrivingIntent（含复合步骤）
多相机/LiDAR ───> WorldState / 语义实体
车辆状态 ───────> Ego state
天气/道路状态 ──> Environment observation
                         │
                         ▼
              多源时间戳与可用性检查
                         │
                         ▼
                    VLA 动作提议
                         │
                         ▼
       VLAFirstDecisionCoordinator（步骤/机动记忆）
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     确定性安全约束通过          不通过/超时/低置信度
            │                         │
            ▼                         ▼
       ControlDecision 1.0       安全停车或规则降级接口
            │
            ▼
          CARLA 控制器
```

VLA 现在拥有正常工况下的动作选择权。确定性部分只处理不可绕过的安全条件，不再用 canonical 规则动作逐帧限制 VLA。旧 `legacy_fsm` 路径仍然保留，用于回归对照、故障降级和分阶段联调。

## 2. 决策模块改动

### 2.1 新增 VLA-first 决策协调器

新增 `lightweight_vla_adapter/src/decision_coordinator.py`，主要能力如下：

- 接收视觉、语音、车辆状态、环境四类输入的健康状态，检查缺失、陈旧、未来时间戳和跨源时间偏差；
- 直接接收 `VLADecisionProposal 1.0` 作为正常决策来源；
- 使用 `DecisionCoordinatorState 1.0` 保存复合指令当前步骤、已完成步骤、正在执行的机动、连续阻塞帧数和重规划原因；
- 连续阻塞达到阈值后输出 `replan_requested=true`，供 VLA/规划服务重新生成动作；
- 支持通过 `StepFeedback` 推进“避让 → 变道 → 超车 → 返回原车道”等复合步骤；
- 支持状态持久化和 `restore()`，进程重启后可恢复任务进度；
- 支持注入外部 `fallback_decision`。未注入时采用保守停车/紧急制动，不会使用随机初始化模型或无门控动作。

不可绕过的约束包括：

- 指令解析状态必须有效；
- 四类输入必须可用且时间对齐；
- VLA 置信度和推理延迟必须满足阈值；
- VLA 指向的目标实体必须经过语义对齐；
- `RiskAssessment` 要求紧急制动或减速时不得执行更激进动作；
- 变道前目标车道必须明确为安全；
- 转向动作必须有目标位置。

默认参数位于 `lightweight_vla_adapter/configs/student_base.json` 的 `decision_runtime`：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `mode` | `vla_first` | 默认使用 VLA 主决策 |
| `minimum_vla_confidence` | 0.55 | VLA 最低置信度 |
| `maximum_source_age_s` | 0.35 s | 单源最大数据年龄 |
| `maximum_source_skew_s` | 0.15 s | 多源最大时间偏差 |
| `maximum_inference_latency_ms` | 150 ms | 与项目既有完整决策预算一致的硬超时 |
| `blocked_frames_before_replan` | 3 帧 | 触发动态重规划的连续阻塞帧数 |
| `maximum_streams` | 128 | 限制驻留任务状态，避免长时间运行内存无界增长 |

### 2.2 Pipeline 接口

`LightweightVLAPipeline.decide()` 新增以下可选参数：

```python
proposal, coordinator_state, control_decision = pipeline.decide(
    batch,
    driving_intent,
    world_state,
    semantic_alignment,
    risk_assessment,
    candidate_entity_ids=candidate_entity_ids,
    prior_state=prior_coordinator_state,
    feedback=step_feedback,
    input_health=multisource_health,
    fallback_decision=rule_fallback_decision,
    decision_mode="vla_first",  # 或 legacy_fsm
)
```

兼容边界如下：

- CARLA 控制端输出仍是 `ControlDecision/1.0.0`，无需修改现有 PID/轨迹跟踪接口；
- `legacy_fsm` 仍调用原 `advance_vla_control_plan()`；
- `vla_first` 的 `prior_state` 必须是 `DecisionCoordinatorState`，避免误把旧 FSM 状态混入新链路；
- 离线入口 `run_offline_inference.py` 已透传 `decision_mode`、`input_health`、`fallback_decision` 和持久化状态；
- 未安装 PyTorch 时，JSON 契约和协调器仍可导入；只有张量校验与模型推理明确要求 PyTorch。

### 2.3 新增契约

- `lightweight_vla_adapter/schemas/multisource_health.schema.json`
- `lightweight_vla_adapter/schemas/decision_coordinator_state.schema.json`

多源健康状态最小示例：

```json
{
  "vision": {"available": true, "timestamp_s": 12.50, "frame_id": "carla_250"},
  "voice": {"available": true, "timestamp_s": 12.43, "confidence": 0.97},
  "vehicle_state": {"available": true, "timestamp_s": 12.50},
  "environment": {"available": true, "timestamp_s": 12.48}
}
```

旧调用方暂未提供健康状态时，协调器会生成标记为 `inferred=true` 的兼容数据。正式比赛链路应由各模块提供真实时间戳，不能长期依赖推断值。

## 3. 场景 2 更新

### 3.1 配置细化

`scene_2_town05_runtime.json` 新增：

- 湿路面摩擦缩放、车灯反射、大车水雾、相机雨痕/眩光/曝光恢复等环境参数接口；
- 慢车在接近路边入口时二次降速，形成“跟车减速 → 判断间隙 → 变道超车”的连续任务；
- 横穿行人增加启动延迟、道路中段迟疑和遮挡来源描述；
- 公交乘客按时间间隔依次上下车，增加最短停站时间、左灯起步和再并线语义；
- 自行车加入 14 → 9 → 16 km/h 的速度变化，为动态超车与重新决策提供扰动；
- 每个特殊事件声明预期观测模态；
- 新增策略/评测真值隔离开关，禁止策略直接读取 CARLA 真值。

### 3.2 运行逻辑

`town05_scene2.py` 已实现：

- `ScriptedWalker` 的错峰启动和中途迟疑；
- 慢车二次降速；
- 自行车分段速度曲线；
- 公交完成乘客交互后解除手刹、接入 Traffic Manager 并驶离站点；
- 慢车和自行车事件的明确 `RESOLVED` 状态；
- `observation()` 稳定接口，仅输出事件生命周期和预期模态，不输出 Actor 位姿或预设危险距离。

这使场景 2 不再只是四个相互独立的演示动作，而能连续触发：行人避让、公交站交互、跟车减速、变道超车、返回车道和再次调整速度。

## 4. 场景 3 更新

`scene_3_emergency_6km_runtime.json` 新增：

- 更低湿路面摩擦系数、积水范围、轮胎水雾、雨痕、对向灯眩光和曝光恢复参数；
- 切入车辆转向灯提前量、变道持续时间和并线后制动扰动接口；
- 施工锥桶反光和两阶段渐变封道描述；
- 施工人员迟疑、观察自车和反光服属性；
- 故障维护车双闪与后方警示三角架距离接口；
- 150 ms 多源同步阈值和 150 ms 完整决策延迟阈值；
- 策略端禁止 CARLA 真值、评测端允许真值的硬性配置。

`emergency_scene_3_events.py` 已让切入车辆显示左转向灯，让故障维护车显示双闪。`EmergencyEventScheduler.policy_observation()` 新增不含 Actor 精确位姿、未来触发距离和预设间隙的同步观察接口。

注意：`surface_and_visibility` 中的摩擦、水雾、雨痕和概率扰动是稳定配置接口；当前版本已由天气、Actor 灯光和事件行为覆盖可直接实现的部分。材质级摩擦触发器、真实镜头雨滴着色器和概率采样器仍需在有 CARLA 环境时接入和调标，不能把配置声明当成已经完成的物理验证。

## 5. 修改/新增文件清单

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `lightweight_vla_adapter/src/decision_coordinator.py` | 新增 | VLA 主决策、复合步骤、机动记忆、动态重规划、安全降级 |
| `lightweight_vla_adapter/src/pipeline.py` | 修改 | 接入 `vla_first` / `legacy_fsm` 双模式及新接口 |
| `lightweight_vla_adapter/src/contracts.py` | 修改 | 无 PyTorch 环境下仍可使用 JSON 决策契约 |
| `lightweight_vla_adapter/__init__.py` | 修改 | 推理依赖改为可选导入，导出协调器接口 |
| `lightweight_vla_adapter/scripts/run_offline_inference.py` | 修改 | 透传健康状态、状态恢复、降级决策和决策模式 |
| `lightweight_vla_adapter/configs/student_base.json` | 修改 | 新增决策运行参数 |
| `lightweight_vla_adapter/schemas/multisource_health.schema.json` | 新增 | 多源输入健康契约 |
| `lightweight_vla_adapter/schemas/decision_coordinator_state.schema.json` | 新增 | 协调器持久化状态契约 |
| `lightweight_vla_adapter/tests/test_decision_coordinator.py` | 新增 | VLA 主决策、安全约束、重规划和步骤推进单测 |
| `experiment/CARLA/configs/scene_2_town05_runtime.json` | 修改 | 场景 2 现实化细节和接口约束 |
| `experiment/CARLA/scenarios/complex/town05_scene2.py` | 修改 | 行人/公交/慢车/自行车动态行为和观察接口 |
| `experiment/CARLA/run_complex_avoidance_town05.py` | 修改 | 校验策略/评测真值隔离约束 |
| `experiment/CARLA/configs/scene_3_emergency_6km_runtime.json` | 修改 | 场景 3 雨夜、施工、切入与接口细化 |
| `experiment/CARLA/emergency_scene_3_events.py` | 修改 | 切入转向灯与维护车双闪 |
| `experiment/CARLA/run_emergency_response_6km.py` | 修改 | 场景 3 接口校验和无真值事件观察 |
| `experiment/CARLA/schemas/environment_observation.schema.json` | 新增 | 天气、路面与能见度融合接口，不含事件真值 |
| `experiment/CARLA/schemas/scene_event_observation.schema.json` | 新增 | 场景生命周期同步接口，不含 Actor 位姿与未来触发信息 |

## 6. 组员模块的问题与联动方向

### 6.1 语音识别与指令解析

当前风险：复杂指令虽然可拆步骤，但步骤完成依赖外部反馈；ASR/解析置信度、时间戳和纠错结果没有形成统一在线健康信号。

建议：

- 输出 `voice` 健康状态，至少包含 `available`、`timestamp_s`、`confidence`、`request_id`；
- 为“避让后变道超车再返回”等指令生成稳定 `step_id` 和依赖关系；
- 对指代不清、噪声导致动作/方向不确定的指令返回 `NEEDS_CLARIFICATION`，不要猜测；
- 将解析耗时拆成 ASR、翻译、语义解析三段写入统一延迟日志。

联动点：直接写入新增 `multisource_health`，步骤结果写入 `StepFeedback`，由协调器推进复合任务。

### 6.2 场景理解与感知

当前风险：项目仍大量使用 CARLA Actor 真值构造 WorldState，模型视觉结果可能只用于展示或影子评测；动态目标的速度、轨迹和遮挡置信度不足。

建议：

- 正式策略链路只消费相机/LiDAR/跟踪器输出；CARLA 真值仅进入独立评测记录器；
- 对行人、自行车、公交、切入车辆、锥桶和施工人员输出连续轨迹、速度、遮挡率、预测占用区和不确定度；
- 用 `simulation_frame` 对齐多相机、LiDAR、车辆状态，拒绝邻帧补齐；
- 针对本次新增的行人迟疑、自行车变速、公交驶离和切入灯光建立专项数据与回归集。

联动点：语义对齐后的 `entity_id` 是 VLA 目标合法性的硬约束；感知健康度进入 `vision` 和 `environment` 健康状态。

### 6.3 VLA 模型训练与推理

当前风险：现有模型仍是单帧高层动作分类器，动作空间只有九个基础动作，难以独立表达复合机动和长期任务进度。

建议：

- 训练样本加入 1–3 秒历史窗口、上一动作、当前机动阶段和已完成步骤；
- 标签从单一 action 扩展为 action、目标车道/目标体、期望速度、短时轨迹、完成概率和需要重规划概率；
- 加入场景 2/3 的遮挡行人、公交驶离、连续变道、切入、施工合流和临时阻塞难例；
- 推理服务输出真实置信度校准结果，持续监控 P95/P99 延迟和超时率；
- 使用新协调器做 VLA-first 影子评测，对比 `legacy_fsm`，不要用 autopilot 行为冒充 VLA 成功。

联动点：协调器的 `maneuver`、`blocked_frames`、`replan_requested` 可回灌为训练上下文和难例标签。

### 6.4 轨迹规划与车辆控制

当前风险：`ControlDecision` 仍是离散动作 + 目标速度，连续变道、超车回正和湿滑路面平滑控制主要依赖下游临时规则。

建议：

- 在不改变 `ControlDecision 1.0` 的前提下新增可选 `TrajectoryProposal` 接口，由局部规划器生成 2–4 秒轨迹；
- 对曲率、横向加速度、横摆角速度、加加速度（jerk）设置约束；
- 变道完成条件基于车道中心偏差、航向差和稳定持续时间，随后生成 `StepFeedback.COMPLETED`；
- 湿滑路面根据 `environment_observation` 降低目标加速度和转向变化率；
- 控制器必须对决策超时、状态断流和重规划窗口执行可预测的最小风险动作。

联动点：控制器反馈是协调器推进复合步骤的唯一可靠依据，不能用“目标暂时消失”代替动作完成。

### 6.5 系统集成与评测

当前风险：模块各自有日志，但尚缺统一时钟、端到端 trace、长时间稳定性和三场景统一完成率看板。

建议：

- 统一 `request_id + simulation_frame + monotonic timestamp`；
- 记录 `visible_at → voice_issued_at → parse_completed_at → policy_started_at → action_started_at → event_resolved_at`；
- 增加每个场景的任务级指标：完整步骤完成率、错误动作率、动态重规划成功率、最小 TTC、舒适性、P95/P99 延迟；
- 至少执行 30–60 分钟连续运行、不同随机种子和传感器丢帧/延迟注入；
- 将 `carla_truth_allowed_for_policy=false` 设为启动时硬校验，防止比赛演示无意回退到真值规则。

## 7. 推荐联调顺序

1. 感知和语音模块先补齐 `multisource_health` 与统一时间戳。
2. 控制模块实现稳定的 `StepFeedback`，至少覆盖变道完成、停车完成、目标通过和失败/取消。
3. 在场景 2 使用 `vla_first` 影子模式跑“行人 → 超车 → 公交 → 自行车”连续任务，同时保留 `legacy_fsm` 对照。
4. 在场景 3 注入感知延迟、帧丢失和不安全目标车道，确认安全约束与动态重规划触发。
5. 最后才让 VLA 输出作用于真实 CARLA 控制，并按三场景任务完成率、延迟和稳定性统一验收。

## 8. 本次验证情况

已完成：

- 所有修改 Python 文件通过 `py_compile`；
- 所有修改 JSON 配置和新增 Schema 通过 JSON 解析；
- VLA-first 协调器 5 个单元测试通过：正常 VLA 变道、语义动作匹配、危险车道拒绝、连续阻塞重规划、复合步骤反馈推进；
- 场景 2 配置验证通过；
- 场景 2 的 3 个 Town05 离线测试通过；
- 场景 3 `--validate-config-only` 通过。

未执行：

- 未启动 CARLA 0.9.16、GPU 模型或完整端到端流程；
- 未验证材质级湿滑摩擦、相机雨滴/水雾和长时间随机扰动；
- 仓库现有完整场景 3 测试入口依赖缺失的 `scene3_video_preview` 模块，故本机无法收集该测试文件；这不是本次改动引入，但正式环境验收前应补齐或修正依赖。
- 本机解释器为 Python 3.9.13，而项目声明环境为 Python 3.12.13；部分既有测试使用 `dict | None` 等 3.10+ 类型语法，无法在本机解释器中收集。

因此，本次结论是“代码结构、契约、配置和不依赖运行环境的测试已通过”，不能等同于 CARLA 全流程或比赛指标已经验收。
