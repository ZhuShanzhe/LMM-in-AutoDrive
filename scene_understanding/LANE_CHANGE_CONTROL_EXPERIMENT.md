# 变道控制实验

`run_lane_change_control_experiment` 从行人避让实验生成的 `ACTIVE step_2` 计划状态继续执行。实验选择一个具有合法、同向左侧相邻车道的 CARLA 地图出生点，在主车前方生成一辆速度较慢的车辆，并将经过安全门控的 JSON 决策发送给团队 PID 控制器。

实验不会因为检测到车道线或已经发出转向命令，就直接认定变道完成。只有以下 CARLA 测量条件全部满足时，`step_2` 才会收到 `COMPLETED` 反馈：

1. 语义对齐已经关联到驾驶意图中的慢车；
2. 同一辆慢车仍然存在于当前 `WorldState` 中；
3. 主车的 `lane_id` 等于所选择的左侧目标车道 ID；
4. 主车距离目标车道中心线不超过 0.9 米；
5. 主车航向与目标车道方向的对齐度不低于 0.95；
6. 上述几何条件连续五帧保持成立；
7. 实验过程中未发生碰撞。

合法变道过程中必然可能跨越车道线，因此压线事件会作为实验依据保留，但不会单独被判定为失败。

在车辆驶入相连道路区段时，CARLA/OpenDRIVE 可能为车辆分配新的 `road_id`，因此实验会记录 `road_id` 作为证据，但不会要求它始终等于预扫描值；车道 ID、车道中心偏移和航向对齐仍必须满足要求。

地图确认车辆已进入目标车道后，实验会向控制器提供该车道前方的显式目标点，并在测量条件稳定期间持续保持该目标，防止重复的 `lane_change_left` 动作再次请求向左变道。

## 运行方法

运行实验前必须启动 CARLA，并确认世界中没有上次实验遗留的车辆、行人或传感器。

```bash
python -m scene_understanding.scripts.run_lane_change_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state_after_pedestrian.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --spawn-index 1 \
  --output-dir outputs/lane_change_control_experiment
```

指定的输出目录必须是一个尚不存在的新目录。目录中包含逐帧 `timeline.jsonl`、基于测量生成的 `step_feedback.json`、更新后的计划状态和决策、最终语义对齐与风险数据，以及 `summary.json`。

一次成功运行只会完成 `step_2` 并激活 `step_3`。超车属于另一个独立的物理驾驶动作，必须根据自身测量结果生成反馈；本实验不会自动将超车步骤标记为完成。
