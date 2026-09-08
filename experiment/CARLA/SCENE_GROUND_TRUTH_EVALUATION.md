# 场景二/三逐帧真值与影子评测

## 为什么要单独记录真值

ASR 与 ModernBERT 指令解析可以用固定、带标注的语音/文本集独立测量。多模态语义
对齐和控制决策不同：它们必须知道同一 CARLA 仿真帧中哪个事件真实存在、关键
Actor 在哪里、目标车道是否已有安全间隙，以及此时哪些控制动作允许或禁止。

本实现将三类数据严格分开：

1. `frame_ground_truth.jsonl` 只读取 CARLA 状态、事件配置和确定性事件控制器；
2. `semantic_predictions.jsonl` 由待测多模态模型输出；
3. `control_decisions_shadow.jsonl` 由待测控制链在影子模式输出，不作用于车辆。

三者只能用 `simulation_frame` 精确连接，不允许用相邻帧填补。真值生成器不读取
任何模型输出，评测器也不会把 CARLA autopilot 的行为冒充成 VLA 控制结果。

## 真值质量等级

| 等级 | 含义 | 是否进入可申报指标 |
|---|---|---|
| `OBSERVED` | 配置要求的真实 Actor/运行阶段均能从本帧 CARLA 状态取得 | 是 |
| `PARTIAL` | 只观察到部分必要证据 | 否 |
| `PROXY` | 有普通交通参与者作为场景占位，但没有对应事件行为控制 | 否 |
| `SCHEDULE_ONLY` | 只有里程调度标签，没有物理 Actor 证据 | 否 |

场景三的切入车辆、施工标志/锥桶、施工车辆、横穿工人、阻塞车和安全间隙均有
事件 Actor，可形成 `OBSERVED` 区间。场景二当前的公交、行人、自行车和普通车辆
仍有若干 `PROXY` 区间，两个后续路口任务还是 `SCHEDULE_ONLY`。因此当前场景二
可以验证接口、同步和覆盖缺口，但不能据此宣称语义对齐达到比赛阈值。

## 运行时记录

场景二正式命令增加：

```bash
python experiment/CARLA/maps/decorate_complex_scene.py \
  ... \
  --record-ground-truth \
  --ground-truth-every-n 1
```

场景三正式命令增加：

```bash
python experiment/CARLA/run_emergency_response_6km.py \
  ... \
  --record-ground-truth \
  --ground-truth-every-n 1
```

每条 `frame_ground_truth.jsonl` 包含：

- `simulation_frame`、仿真时间和路线里程；
- 自车与关键 Actor 的位姿、速度、车道；
- Actor 相对自车的纵向/横向距离、闭合速度和可计算的 TTC；
- 当前事件、风险标签、允许/禁止的控制动作；
- Actor 缺失情况、真值质量和数据来源。

JSON Schema 位于：

- `schemas/frame_ground_truth.schema.json`
- `schemas/semantic_prediction.schema.json`
- `schemas/control_decision_shadow.schema.json`

## 待测模型输出

语义模型逐帧输出示例：

```json
{
  "schema_version": "SemanticPrediction/1.0.0",
  "scene_id": "scene_3_emergency_6km",
  "simulation_frame": 12345,
  "active_event_ids": ["scene3_blocked_lane"],
  "risk_labels": ["blocked_lane", "unsafe_target_lane_gap"],
  "model_id": "vla-semantic-model"
}
```

控制链影子输出示例：

```json
{
  "schema_version": "ControlDecisionShadow/1.0.0",
  "scene_id": "scene_3_emergency_6km",
  "simulation_frame": 12345,
  "action": "decelerate",
  "safety_gate_status": "APPROVED",
  "latency_ms": 84.2
}
```

影子输出只记录“如果接管会怎么做”，不得调用 `vehicle.apply_control`。正式闭环测试
需在影子评测通过后另行进行。

## 生成评测报告

```bash
PYTHONPATH="$PWD/experiment/CARLA" \
python -m evaluation.shadow_evaluation \
  --ground-truth "$RUN_DIR/frame_ground_truth.jsonl" \
  --semantic-predictions "$RUN_DIR/semantic_predictions.jsonl" \
  --control-decisions "$RUN_DIR/control_decisions_shadow.jsonl" \
  --output "$RUN_DIR/shadow_evaluation_report.json"
```

报告包含：

- `event_detection`：事件集合精确匹配、micro/macro F1；
- `risk_label_alignment`：风险标签精确匹配、micro/macro F1；
- `action_compatibility_rate`：控制动作是否属于本帧允许集合；
- `unsafe_action_false_approval_rate`：禁止动作被安全门批准的比例；
- 延迟中位数和 P95；
- 真值覆盖率、精确帧匹配覆盖率和缺失事件类别。

只有 `OBSERVED` 真值帧不少于 30、覆盖至少 3 个事件类别且预测精确帧覆盖率不低于
95% 时，子报告才标为 `MEASURED`。否则返回
`INSUFFICIENT_EVIDENCE` 并列出原因。`MEASURED` 也只代表测量条件成立，不等于已
达到比赛阈值，更不等于完成闭环控制验收。

## 建议验收顺序

1. 固定语料分别测 ASR 和指令解析；
2. 用场景三先跑通逐帧语义预测和影子控制；
3. 根据场景二报告中的 `PROXY/SCHEDULE_ONLY` 缺口补真实事件 Actor；
4. 场景二重新采集真值，确认全部目标事件进入 `OBSERVED`；
5. 冻结数据和模型后统计最终语义指标；
6. 最后进行带安全门的闭环车辆控制测试。
