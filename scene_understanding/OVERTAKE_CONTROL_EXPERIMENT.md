# 超车控制实验

`run_overtake_control_experiment` 从成功变道实验生成的 `ACTIVE step_3` 状态继续执行。实验开始时，主车位于超车道，已完成语义关联的慢车位于相邻车道前方。只有语义对齐成功且风险评估允许时，计划才会输出 `accelerate`；已有的紧急制动和减速规则始终具有更高优先级。

只有满足以下全部条件时，实验才会报告 `COMPLETED`：

1. 实验已经测量并确认所匹配的慢车最初位于主车前方；
2. 同一辆慢车仍然存在于 `WorldState` 中；
3. 该车辆在主车坐标系中的纵向位置至少位于主车后方 8 米；
4. 主车距离超车道中心线不超过 0.9 米；
5. 主车航向与该车道方向的对齐度不低于 0.95；
6. 后向净距和车道条件连续五帧保持成立；
7. 实验过程中未发生碰撞。

实验将加速后的目标速度限制在 40 km/h。该上限只会修正普通的 `accelerate` 动作，不会削弱 `decelerate`、`stop` 或 `emergency_brake` 决策。

## 运行方法

```bash
python -m scene_understanding.scripts.run_overtake_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state_after_lane_change.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --spawn-index 60 \
  --output-dir outputs/overtake_control_experiment
```

新输出目录中包含逐帧时间线、测量反馈、最终语义对齐和风险结果、终止时的计划状态、控制器决策以及实验摘要。

最终步骤只能依据实际测得的后向安全净距判定完成；目标从感知结果中消失或仅经过一定时间，都不能作为完成依据。
