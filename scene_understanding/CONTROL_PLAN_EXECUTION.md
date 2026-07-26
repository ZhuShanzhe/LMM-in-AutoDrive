# 多步骤控制计划执行

该模块用于在连续 CARLA 帧之间推进结构化的多步骤 DrivingIntent，并将每一次状态转换保存为 JSON。系统不会仅因为目标从感知结果中消失，就推断某个步骤已经完成；控制器或评估器必须提供显式的 `StepFeedback` 事件。

## 文件说明

- `control_plan_state.json`：持久化保存每个驾驶意图步骤的状态；
- `step_feedback.json`：当前活动步骤的显式执行结果；
- `control_decision.json`：发送给 CARLA 控制器的单个扁平化动作。

稳定的数据契约位于：

- `scene_understanding/schemas/control_plan_state.schema.json`；
- `scene_understanding/schemas/step_feedback.schema.json`；
- `scene_understanding/schemas/control_decision.schema.json`。

## 初始化计划

处理第一帧时，不要传入 `--state` 或 `--feedback`：

```bash
python -m scene_understanding.scripts.advance_control_plan \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --state-output outputs/control_plan_state.json \
  --decision-output outputs/control_decision.json
```

初始活动步骤为 `intent.steps[0]`。步骤受阻时，按照其 `on_blocked` 策略处理：

- `WAIT_FOR_SAFE`：将步骤保持为 `WAITING`，并输出安全兜底动作；
- `SKIP_STEP`：将该步骤标记为 `SKIPPED`，随后评估下一步骤；
- `SAFE_STOP` 或未知策略：将计划终止为 `BLOCKED` 并停车。

## 不完成当前步骤并继续评估

处理新的 WorldState 帧时，传入上一帧状态但不传入反馈。命令会依据新的语义对齐和风险文件，重新评估同一个活动步骤。因此，当环境条件转为安全时，处于 `WAITING` 状态的步骤可以自动恢复，但系统不会在没有显式反馈的情况下推进到下一步骤。

```bash
python -m scene_understanding.scripts.advance_control_plan \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state_next.json \
  --semantic-alignment outputs/semantic_alignment_next.json \
  --risk-assessment outputs/risk_assessment_next.json \
  --state outputs/control_plan_state.json \
  --state-output outputs/control_plan_state.json \
  --decision-output outputs/control_decision.json
```

状态文件采用原子替换方式写入，因此 `--state` 和 `--state-output` 可以指向同一路径。

## 完成并推进步骤

反馈中的计划 `request_id` 和步骤 ID 必须分别与当前计划及 `active_step_id` 一致。例如，`scene_understanding/schemas/examples/step_feedback.example.json` 用于完成 `step_1`：

```bash
python -m scene_understanding.scripts.advance_control_plan \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state_next.json \
  --semantic-alignment outputs/semantic_alignment_next.json \
  --risk-assessment outputs/risk_assessment_next.json \
  --state outputs/control_plan_state.json \
  --feedback inputs/step_feedback.json \
  --state-output outputs/control_plan_state.json \
  --decision-output outputs/control_decision.json
```

收到 `COMPLETED` 后，系统会检查 `depends_on`，随后激活下一步骤。`FAILED` 和 `CANCELLED` 会终止计划并输出 `stop`。全部步骤完成后，计划状态变为 `COMPLETED`，系统按照当前主车速度输出 `keep_lane`。

## 职责边界

执行器负责步骤顺序、持久化状态、依赖关系以及经过安全门控的动作选择，但不负责判断某个物理驾驶动作是否已经真实完成。

在早期 JSON 联调中，可以人工生成反馈；在 CARLA 闭环实验中，反馈必须由控制或评估层依据可测量条件生成，例如达到目标速度、车道 ID 已变化或已经到达路线目标。
