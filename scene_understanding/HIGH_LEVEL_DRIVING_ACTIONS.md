# 高层驾驶动作设计

## 1. 目标与范围

本设计面向“驾驶意图 -> 风险评估 -> 决策模块 -> CARLA 控制指令”之间的高层动作层。目标是：

- 构建基础操控、复杂避障、应急响应三类场景框架；
- 定义适用于当前决策模块的高层动作空间；
- 设计规则决策模块与有限状态机组合执行结构；
- 明确定义安全优先级、异常处理与容错机制；
- 记录决策触发条件与决策原因，便于后续 CARLA 具体指令映射与审计。

本设计假定当前已实现的扁平控制动作为：

- `keep_lane`
- `accelerate`
- `decelerate`
- `stop`
- `emergency_brake`
- `lane_change_left`
- `lane_change_right`
- `turn_left`
- `turn_right`

同时保留“多步骤计划执行器”语义，包括步骤状态、`on_blocked` 策略，以及显式 `StepFeedback` 机制。

## 2. 场景分类

### 2.1 基础操控

适用于常规驾驶、车道保持、速度控制和普通转弯。核心目标是保持行车稳定性、完成指令、避免轻微偏差。

典型动作组合：

- 保持车道：`keep_lane`
- 加速：`accelerate`
- 减速：`decelerate`
- 停车：`stop`
- 转向：`turn_left` / `turn_right`

触发条件示例：

- 目标车道保持、当前无重交通威胁
- 前方目标速率高于当前速度，且风险评估未建议减速
- 路口左转/右转指令已对齐目标位置

### 2.2 复杂避障

适用于前方障碍物、慢车、行人、受限车道、跨车道变道等复杂场景。重点在于语义对齐、风险判断、可行性检查和受阻策略。

高层动作包括：

- 变道避障：`lane_change_left` / `lane_change_right`
- 慢行等待：`decelerate`
- 停车等待：`stop`
- 继续保持：`keep_lane`（作为安全备选）

触发条件示例：

- 前方障碍物、慢车风险高且变道安全
- 目标车道不可行且当前策略为“等待可安全执行”
- 语义对齐失败，且当前步骤 `on_blocked` 为 `WAIT_FOR_SAFE` 或 `SKIP_STEP`

### 2.3 应急响应

适用于碰撞风险、突然障碍、失控车辆、紧急停车等高风险场景。此类场景必须覆盖异常处理、降级策略、保命优先。

核心动作：

- 紧急制动：`emergency_brake`
- 立即停车：`stop`
- 安全减速：`decelerate`
- 保持车道或就近停止：`keep_lane`

触发条件示例：

- 风险评估返回 `recommended_action` 为 `emergency_brake`
- `risk_level` 为 `high`
- 目标实体即将进入碰撞路径，且 TTC / 交汇角度超过安全阈值
- 语义对齐出现关键失败且当前步骤无可行安全继承策略

## 3. 高层动作空间设计

### 3.1 核心动作定义

| 高层动作 | 语义说明 | 主要用途 | 安全优先级 |
|---|---|---|---|
| `keep_lane` | 保持当前车道和方向 | 常规巡航、完成当前意图、应急退让 | 低 |
| `accelerate` | 增大纵向速度 | 提高速度、超车、完成加速意图 | 低 |
| `decelerate` | 缓慢降低速度 | 跟车、靠近障碍、等待、安全降速 | 中 |
| `stop` | 终止纵向前进 | 交叉口停车、不可行状态、安全兜底 | 高 |
| `emergency_brake` | 紧急制动、快速停止 | 即刻躲避碰撞、极高风险 | 最高 |
| `lane_change_left` | 左侧变道 | 避障、超车、左转前准备 | 中 |
| `lane_change_right` | 右侧变道 | 避障、超车、右转前准备 | 中 |
| `turn_left` | 左转 | 执行路口左转意图 | 中 |
| `turn_right` | 右转 | 执行路口右转意图 | 中 |


### 3.2 高层动作拓展说明

- 对于 `OVERTAKE`、`YIELD`、`AVOID` 等复杂意图，决策层可先映射为上述基础动作中的一种组合或序列。
- 例如 `OVERTAKE` 在当前计划语义中，若变道安全则映射为 `lane_change_left/right`，否则映射为 `accelerate` 或 `decelerate`。这保持接口稳定，同时为后续 CARLA 具体控制提供可插拔转换。
- `PULL_OVER` 这类意图可以在高层设计中视为“连续 `decelerate` + `stop`”，在 CARLA 映射阶段补充车道边缘、停止位置等定位信息。

## 4. 规则决策模块设计

### 4.1 决策输入

决策模块接收以下三类输入：

1. 驾驶意图 `DrivingIntent`
2. 世界状态 `WorldState`
3. 风险评估 `RiskAssessment`
4. 语义对齐 `SemanticAlignment`

其中，`DrivingIntent` 和 `SemanticAlignment` 提供目标意图与实体匹配；`RiskAssessment` 提供安全门控；`WorldState` 提供车辆速度、车道与环境上下文。

### 4.2 决策输出

决策模块输出一个高层驾驶动作字典，至少包含：

- `action`：高层动作名称
- `target_speed_kmh`：目标速度
- `target_lane`：变道方向（如有）
- `target_location`：转向目标位置（如有）
- `emergency`：是否紧急动作
- `decision_status`：决策状态，如 `READY`、`BLOCKED`、`SAFE_FALLBACK`
- `reason`：决策原因标签
- `blocked_reason_codes`：若阻塞则输出原因码列表

### 4.3 决策规则优先级

1. `emergency_brake` 或风险评估明确要求紧急制动时，立即覆盖所有普通意图。
2. 解析状态非 `VALID` 时，输出 `stop` 作为安全兜底。
3. 语义对齐失败或目标不可见时，按照当前步骤 `on_blocked` 策略执行：
   - `WAIT_FOR_SAFE` -> `decelerate` / `keep_lane` 等安全等待动作；
   - `SKIP_STEP` -> 保持车道并推进下一可行步骤；
   - 其它或缺失 -> `stop`。
4. 变道动作必须同时满足 `RiskAssessment.lane_change.left/right.is_safe`。若不安全，则退回 `decelerate` / `keep_lane`。
5. 转向指令必须验证目标位置或目标车道有效，避免盲目 `turn_left` / `turn_right`。
6. 动作输出应当保留决策原因与触发条件，便于审计与后续映射。

### 4.4 规则决策示意伪代码

```python
if risk_assessment['recommended_action'] == 'emergency_brake' or risk_assessment['risk_level'] == 'high':
    return emergency_brake_decision(reason='risk_emergency_brake')

if driving_intent['parse_result']['status'] != 'VALID':
    return stop_decision(reason='parse_invalid')

step = current_active_step(driving_intent, prior_state)
if step_unaligned_or_blocked(step, semantic_alignment):
    policy = step.get('on_blocked', 'SAFE_STOP')
    return blocked_policy_decision(policy, reason_codes)

if step['action'] == 'CHANGE_LANE':
    direction = step['parameters']['direction']
    if risk_assessment['lane_change'][direction.lower()]['is_safe']:
        return lane_change_decision(direction, reason='aligned_safe_lane_change')
    return decelerate_decision(reason='unsafe_lane_change')

if step['action'] == 'TURN':
    if has_valid_target_location(step, semantic_alignment):
        return turn_decision(direction, target_location, reason='aligned_turn')
    return stop_decision(reason='turn_target_invalid')

return keep_lane_or_speed_decision(step, world_state, risk_assessment)
```

## 5. 有限状态机执行管理

### 5.1 核心状态集合

高层执行器使用有限状态机管理步骤执行过程。当前实现的步骤状态包括：

- `PENDING`
- `ACTIVE`
- `WAITING`
- `COMPLETED`
- `SKIPPED`
- `BLOCKED`
- `FAILED`
- `CANCELLED`

计划状态包括：

- `ACTIVE`
- `COMPLETED`
- `BLOCKED`
- `FAILED`
- `CANCELLED`
- `SAFE_FALLBACK`

### 5.2 状态切换逻辑

1. 初始化时，若意图可执行且解析有效，则将第一个步骤置为 `ACTIVE`；否则进入 `SAFE_FALLBACK`。
2. 每一帧决策都基于当前 `active_step_id` 进行语义对齐和风险评估。
3. 若当前 `ACTIVE` 步骤被阻塞：
   - `WAIT` 策略 -> 将步骤置为 `WAITING`，继续输出保护动作；
   - `SKIP` 策略 -> 将步骤置为 `SKIPPED`，激活下一个依赖完成的步骤；
   - `STOP` 策略 -> 将步骤置为 `BLOCKED`，输出 `stop`。
4. 明确收到 `StepFeedback` 后：
   - `CONTINUE` -> 保持 `ACTIVE`，可能继续输出同一个决策；
   - `COMPLETED` -> 置当前步骤为 `COMPLETED`，激活下一个步骤或完成计划；
   - `FAILED` / `CANCELLED` -> 终止计划，进入 `FAILED` / `CANCELLED`。

### 5.3 组合动作执行

复杂驾驶任务由多个步骤组合而成：

- `OVERTAKE` 可能拆分为 `lane_change` + `accelerate` + `keep_lane`
- `PULL_OVER` 可能拆分为 `decelerate` + `turn` + `stop`
- `YIELD` 可能体现为 `keep_lane` + `decelerate` + `wait`

有限状态机保证每个步骤只有在显式反馈后才推进，避免因为感知短暂丢失而误判完成。

## 6. 决策触发条件与原因记录

每次高层决策应保存以下审计信息：

- `source_step_id`：当前活动步骤或最近终止步骤 ID
- `decision_status`：`READY` / `BLOCKED` / `SAFE_FALLBACK`
- `reason`：主要触发原因标签
- `blocked_reason_codes`：如果被阻塞，列出具体原因码
- `parse_status`：原始解析状态
- `parse_confidence`：解析置信度
- `risk_level`：当前风险等级
- `recommended_action`：风险评估给出的建议
- `step.on_blocked`：当前步骤阻塞策略

建议的原因标签示例：

- `risk_emergency_brake`
- `parse_invalid`
- `semantic_target_missing`
- `unsafe_lane_change`
- `blocked_wait_for_safe`
- `blocked_safe_stop`
- `plan_completed`
- `plan_blocked_safe_stop`

## 7. 安全优先级与容错机制

### 7.1 安全优先级原则

1. `emergency_brake` 优先于所有其他动作。
2. `stop` 优先于 `keep_lane` / `accelerate` / `lane_change`。
3. 风险评估建议的 `decelerate` 在无紧急制动时优先于加速或变道。
4. 目标对齐缺失时，不执行目标依赖性强的动作（如转弯、变道）。
5. 在多步骤计划中，若依赖尚未完成，系统应保持当前安全动作，避免冒进。

### 7.2 异常处理

- 输入结构错误：抛出异常或进入安全兜底；但在实际运行中，需尽快记录并报警。
- 数据不一致：若 `request_id`、`frame_id`、`parse_status` 不一致，立即拒绝执行并输出 `stop`。
- 语义对齐失败：按 `on_blocked` 策略处理，优先使用 `WAIT` 或 `STOP`，避免无效动作。
- 风险评估不可用或异常：退回 `decelerate` 或 `stop`，绝不使用 `accelerate`。

### 7.3 决策容错

- 保持“最近安全动作”原则：在当前帧无法产生明确新动作时，输出最近一次经过风险门控的安全动作。
- 对于多步骤计划，避免因一次语义对齐失败直接跳到后续步骤，必须显式 `SKIP` 或 `COMPLETED`。
- 对于 `WAITING` 步骤，可在后续帧重新恢复 `ACTIVE`，但不得自动推进。

## 8. CARLA 映射建议

高层动作应保持抽象，后续由 CARLA 控制层将其映射为具体纵横向控制指令：

- `keep_lane` -> 纵向速度目标 + 车道保持控制
- `accelerate` / `decelerate` -> 速度目标调整
- `stop` / `emergency_brake` -> 速度目标 0 + 刹车力度
- `lane_change_left` / `lane_change_right` -> 目标车道切换
- `turn_left` / `turn_right` -> 曲线轨迹或路口转向执行

建议后续映射模块依据高层动作同时接收：

- `target_speed_kmh`
- `target_lane`
- `target_location`
- `emergency`

以便构造 CARLA `VehicleControl` 或路径跟踪命令。

## 9. 说明

下一步可以基于本设计完成以下工作：

1. 将本文件中的高层动作空间映射到 CARLA 控制接口；
2. 在 CARLA 层实现 `action -> VehicleControl` 的安全门控转换；
3. 为每个高层决策输出补全审计字段；
4. 校准 `WAIT_FOR_SAFE` / `SKIP_STEP` / `SAFE_STOP` 等阻塞策略的具体 CARLA 行为；
5. 复用现有 `scene_understanding/src/control_decision.py` 和 `scene_understanding/src/control_plan_executor.py` 中的动作语义。
