# 行人避让控制实验

`run_pedestrian_control_experiment` 是结构化指令“避让横穿行人，然后变道并超车”第一步骤对应的 CARLA 0.9.16 闭环测试。该实验组合使用团队已有模块，不复制或修改其他模块的内部实现：

- 行人横穿场景负责创建主车和行人；
- 场景理解采集器在每个同步仿真帧生成经过校验的 `WorldState`；
- 语义对齐和风险评估读取当前帧；
- 控制计划执行器输出一份 JSON 格式的 `ControlDecision`；
- 团队 PID 控制器将该决策应用于真实 CARLA 主车；
- 根据 CARLA 实测状态为当前活动步骤生成 `StepFeedback`。

## 完成判定语义

只有以下条件全部满足时，实验才会将 `step_1` 报告为 `COMPLETED`：

1. 此前已经观测到一名处于 `crossing_ego_path` 状态的行人；
2. 该行人已不再横穿，并到达道路另一侧，即 `relative_position_ego_m.lateral <= -2.5` 米；
3. 主车速度相较于实测初始速度至少下降 3 m/s；
4. 碰撞传感器未报告任何碰撞。

系统绝不会仅根据目标消失判定步骤完成。发生碰撞、实验超时或计划过早进入终止状态时，系统会生成 `FAILED` 反馈。

系统会在处理 `SAFE_STOP` 的目标缺失策略之前检查完成条件，以确保行人刚刚正确通过的第一帧被记录为成功，而不会被错误判断为目标缺失导致的阻塞。

## 运行方法

运行实验前必须启动 CARLA，并确认世界中没有上次实验遗留的车辆、行人或传感器。

```bash
python -m scene_understanding.scripts.run_pedestrian_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --output-dir outputs/pedestrian_control_experiment
```

指定的输出目录必须是一个尚不存在的新目录，其中包含：

- `timeline.jsonl`：逐帧测量结果、决策和实际控制量；
- `step_feedback.json`：`step_1` 的物理执行结果；
- `control_plan_state.json`：应用反馈后的持久化计划状态；
- `control_decision.json`：新激活步骤或终止步骤对应的决策；
- `semantic_alignment.json` 和 `risk_assessment.json`：最终帧的数据契约；
- `summary.json`：简要实验结果；
- `error.json`：仅在发生非预期异常时生成。

行人子实验成功后，完整指令计划通常会保持为 `ACTIVE`，活动步骤推进至 `step_2`。如果独立行人场景中不存在慢车，`step_2` 会正确保持为 `WAITING`；实验不会虚构车辆，也不会声称后续步骤已经执行。
