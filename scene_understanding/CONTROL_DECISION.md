# 控制决策 JSON 接口

该接口用于连接指令解析模块、场景理解模块输出与 CARLA 控制器，同时避免各模块的 Python 包产生直接耦合。

## 输入与输出

该命令读取同一请求、同一 WorldState 帧对应的四份 JSON 文档：

- `driving_intent.json`：指令解析模块生成的结构化用户驾驶指令；
- `world_state.json`：CARLA 场景的度量化快照；
- `semantic_alignment.json`：驾驶意图目标与 Actor 或车道的语义对齐结果；
- `risk_assessment.json`：基于确定性规则生成的目标风险和变道安全结果。

命令会输出一份 `control_decision.json`，其格式可由控制分支中的 `control.protocol.normalize_intent` 函数直接接收：

```bash
python -m scene_understanding.scripts.build_control_decision \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --output outputs/control_decision.json
```

对应的 Schema 位于 `scene_understanding/schemas/control_decision.schema.json`，完整示例位于 `scene_understanding/schemas/examples/control_decision.example.json`。

## 确定性安全规则

1. 所有输入中的 `request_id`、`frame_id` 和解析状态必须保持一致。
2. 解析结果不是 `VALID` 时，系统采用安全的 `stop` 兜底动作。
3. 显式的 `STOP` 和 `EMERGENCY_BRAKE` 指令会被保留。
4. 风险评估推荐的 `emergency_brake` 或 `decelerate` 优先级高于普通驾驶指令。
5. 必需目标未匹配时，按照当前步骤的 `on_blocked` 策略处理：`WAIT_FOR_SAFE` 对应减速等待，`SKIP_STEP` 对应保持车道并跳过步骤，其他策略或缺失策略对应停车。
6. 只有对应方向的车道安全判断为安全时，系统才会输出变道动作。
7. 转向指令如果缺少规划模块提供的目标位置，系统会阻止执行，而不会盲目转向。
8. 在顺序计划中，已完成语义对齐且未指定其他变道方向的 `OVERTAKE` 步骤会映射为 `accelerate`；紧急制动、风险减速和目标对齐安全门控仍具有更高优先级。

## 当前执行范围

该适配器每次始终输出一个扁平化控制决策。为了向后兼容，默认选择驾驶意图中的第一个步骤；使用有状态控制计划执行器时，则由执行器提供 `source_step_id`，以评估当前活动步骤。

控制计划执行器只有在收到与当前帧及步骤匹配的显式反馈后，才会推进步骤依赖关系，详见 `CONTROL_PLAN_EXECUTION.md`。

兼容性测试使用团队控制器的 `control.protocol.normalize_intent` 接口。该 JSON 适配器本身不依赖 CARLA 运行时；闭环实验使用团队的 CARLA 0.9.16 运行环境和控制模块。
