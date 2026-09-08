# 控制与决策接入

| 文件 | 作用 |
|---|---|
| `protocol.py` | 统一 `DrivingIntent`、`ControlDecision` 和控制动作字段 |
| `scene_bridge_policy.py` | 场景理解结果到规则 FSM 决策的桥接 |
| `scene_understanding_json_policy.py` | 读取并校验场景理解输出 |
| `scheduled_scene_bridge_policy.py` | 按路线进度触发指令并推进计划 |
| `structured_vla_scene_bridge_policy.py` | 结构化 BEV、VLA proposal 与安全门接入 |
| `voice_schedule_policy.py` | 文本指令调度和 DrivingIntent 缓存 |
| `route_adapter.py` | 将高层动作绑定到当前路线和目标车道 |
| `step_completion.py` | 判断速度、车道、转向和复合动作步骤是否完成 |
| `safety_supervisor.py` | 对最终动作执行时序安全约束 |
| `pid_controller.py` | 将目标速度、路线和车道目标转换为油门、制动和转向量 |
| `live_perception_bridge.py` | 实时感知帧与场景语义融合桥接 |
| `motion_contract.py` | 高层动作的速度与转向约束 |
