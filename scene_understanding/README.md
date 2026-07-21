# Scene Understanding

场景理解模块负责把结构化驾驶指令与 CARLA 世界状态对齐，生成确定性的风险判断、控制决策和多步骤计划状态。模块通过 JSON 文件与指令解析模块和车辆控制模块联调，不直接依赖其他成员的 Python 包内部实现。

## 运行环境

- CARLA：0.9.16
- Python：3.10 及以上（闭环实验已在 Python 3.11.8 验证）
- CARLA Python API：必须与服务端保持 0.9.16
- 闭环实验需要正在运行的 CARLA 服务端和可用的团队控制模块
- JSON 对齐、风险判断和单元测试不要求启动 CARLA 服务端

## 目录结构

```text
scene_understanding/
├── core/       # WorldState、CARLA 采集、传感器、风险与视觉结果处理
├── src/        # 指令对齐、控制决策和多步骤计划执行器
├── scripts/    # JSON 联调命令与 CARLA 闭环实验入口
├── schemas/    # JSON Schema 和示例
├── prompts/    # 当前提示词与历史提示词归档
├── tests/      # 模块全部测试
└── *.md        # 各接口与闭环实验说明
```

## JSON 联调接口

| 文件 | 生产方 | 消费方 | 作用 |
|---|---|---|---|
| `driving_intent.json` | 结构化指令解析模块 | 本模块 | 驾驶步骤、目标、依赖关系和阻塞策略 |
| `world_state.json` | CARLA 世界状态采集器 | 对齐与风险模块 | 主车、交通参与者、车道、环境和传感器事件 |
| `semantic_alignment.json` | 本模块 | 控制决策模块 | 将“行人、慢车、前车、车道”等目标关联到实体 |
| `risk_assessment.json` | 本模块 | 控制决策模块 | 距离、TTC、碰撞风险和左右变道安全性 |
| `control_decision.json` | 本模块 | 团队控制模块 | 单帧安全门控后的扁平控制动作 |
| `control_plan_state.json` | 本模块 | 下一帧计划执行器 | 多步骤计划状态和当前活动步骤 |
| `step_feedback.json` | 控制器或实验评估器 | 本模块 | 当前步骤的完成、失败或跳过反馈 |

下游 `control_decision.json` 已与团队控制模块的 `control.protocol.normalize_intent` 接口联调。主要动作包括：

```text
keep_lane
accelerate
decelerate
stop
emergency_brake
lane_change_left
lane_change_right
turn_left
turn_right
```

## 基本联调流程

### 1. 语义对齐

```bash
python -m scene_understanding.scripts.align_driving_intent \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --output outputs/semantic_alignment.json
```

目标不可见或不支持时会明确输出未匹配结果，不会虚构场景实体。

### 2. 风险评估

```bash
python -m scene_understanding.scripts.assess_risk \
  --world-state inputs/world_state.json \
  --output outputs/risk_assessment.json
```

风险输出包含安全跟车距离、TTC、目标风险、碰撞与压线事件，以及左右变道安全判断。

### 3. 单步控制决策

```bash
python -m scene_understanding.scripts.build_control_decision \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --output outputs/control_decision.json
```

风险规则始终高于普通驾驶动作。目标未匹配、车道不安全或输入状态无效时，模块按照 `on_blocked` 策略减速或停车。

### 4. 多步骤计划推进

初始化计划：

```bash
python -m scene_understanding.scripts.advance_control_plan \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --state-output outputs/control_plan_state.json \
  --decision-output outputs/control_decision.json
```

后续帧使用 `--state` 读取上一状态，并可用 `--feedback` 提交当前步骤的显式执行结果。计划支持 `PENDING`、`ACTIVE`、`WAITING`、`COMPLETED`、`SKIPPED` 和 `FAILED` 等步骤状态。

## CARLA 闭环验证

已在 CARLA 0.9.16 中分别完成以下闭环实验：

1. 行人横穿时减速避让，行人通过并确认零碰撞后完成步骤；
2. 关联前方慢车，在合法长直路段完成左变道并稳定保持目标车道；
3. 在超车道加速，使同一慢车从主车前方变为至少后方 8 米，且全程零碰撞。

对应文档：

- `PEDESTRIAN_CONTROL_EXPERIMENT.md`
- `LANE_CHANGE_CONTROL_EXPERIMENT.md`
- `OVERTAKE_CONTROL_EXPERIMENT.md`
- `CONTROL_PLAN_EXECUTION.md`
- `CONTROL_DECISION.md`

实验运行入口位于 `scripts/run_*_control_experiment.py`。实验输出和完整时间线属于运行证据，不提交到源代码目录。

## 测试

在仓库根目录运行：

```bash
python -m unittest discover -s scene_understanding/tests -v
```

当前测试集共 120 项，覆盖：

- JSON 结构和确定性校验；
- WorldState 坐标与相对运动；
- CARLA Actor、车道、交通灯与传感器采集；
- 风险评估和车道安全判断；
- 指令目标语义对齐；
- 控制决策安全优先级；
- 多步骤计划状态推进；
- 行人避让、变道和超车完成条件。

## Schema 与示例

所有稳定 JSON 契约位于 `schemas/`，可直接用于模块间字段确认。示例位于 `schemas/examples/`，包括：

- `world_state.example.json`
- `semantic_alignment.example.json`
- `risk_assessment.example.json`
- `control_decision.example.json`
- `control_plan_state.example.json`
- `step_feedback.example.json`

更完整的场景理解和视觉推理说明见 `schemas/README.md`。

## 当前边界

- 三个闭环步骤目前通过持久化 JSON 状态在独立 CARLA 场景中依次验证，并非同一场景进程中的一次连续演示。
- 当前超车完成条件是超过慢车并稳定保持超车道，尚未包含返回原车道。
- 风险阈值属于确定性研究规则，需要在更多场景和速度分布中继续标定。
- 视觉模型基线已经跑通，但交通灯存在重复检测，交通标志召回仍需改进。
- CARLA 启动和 GPU 图形库兼容方式由部署环境决定，不随本模块提交大型镜像或运行产物。
