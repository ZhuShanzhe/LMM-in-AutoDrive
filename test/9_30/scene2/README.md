# 场景 2 测试结果与问题定位

本轮车辆沿路线前进 **189.82 / 8,000.60 m**，最高观测车速 **44.93 km/h**，碰撞和车道线侵入均为 0。仅触发第 1 条指令，随后持续停车，以 `route_stalled` 结束，未完成场景。

## 主要问题

第一条复合指令为“保持当前车道，将车速调整至45公里每小时，在前方路口右转，然后保持车道继续行驶”。车辆能够完成起步和加速，但进入转弯步骤时缺少执行目标。

1. **约 00:25.2，帧 28005：转弯目标缺失。** 当时车速约 44.07 km/h，风险为低。TURN 步骤只有 `direction=RIGHT`，未获得 `target_location`；最终动作变为 `stop`，原因为 `active_plan_not_ready`，阻塞码为 `turn_target_location_missing`、`safe_stop`。
2. **计划持续阻塞。** 后续 948 次有效决策包含相同阻塞原因，其中 780 次还叠加高风险紧急制动。首次停车发生在低风险帧，所以高风险并非最初阻塞的原因。
3. **约 02:00，帧 29901：指令超时。** 调度器出现 `command dispatch halted: TIMEOUT`，此后共记录 1,721 次安全停车回退，直到停滞保护结束。

路线审计中，第一条指令的右转几何为 `BOUND`，约 305.7 m 处存在右转段；车辆尚未到达便停车。应优先检查**路线几何如何传递为当前步骤的转弯目标和就绪证据**，以及 TURN 步骤是否过早进入执行，而不是仅调整油门或风险阈值。

相关代码：
- `experiment/CARLA/run_complex_avoidance_town05.py`：配置 DrivingIntent 的构建与传递。
- `lightweight_vla_adapter/src/driving_plan_runtime.py`：`_road_target`、`_effective_step` 的目标补齐。
- `experiment/CARLA/control/command_dispatch.py`：计划超时后的调度处理。

此外，路线审计发现后续 4 条指令存在几何不匹配，详见 `route_command_audit.json`；这些任务本次未到达，与当前首次停车原因分开处理。本条 DrivingIntent 来源为 `CONFIGURED_PLAN`，上述目标缺失不能归因于 ModernBERT 的文本解析结果。

## 日志

| 文件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 结果统计、首次阻塞与首次超时记录（含异常堆栈） |
| [logs/decision_trace.jsonl](logs/decision_trace.jsonl) | 1,200 条有效决策及 1,721 条回退记录 |
| [logs/key_frames.json](logs/key_frames.json) | 关键帧的模型输出、DrivingIntent、计划前后状态和执行约束 |
| [logs/execution_trace.jsonl](logs/execution_trace.jsonl) | 控制提交与执行反馈记录 |
| [logs/runtime_feedback.jsonl](logs/runtime_feedback.jsonl) | 约每秒一次的实际速度、路线进度、交通与安全反馈 |
| [logs/commands.jsonl](logs/commands.jsonl) | 实际宣布的第 1 条指令 |
| [logs/route_command_audit.json](logs/route_command_audit.json) | 路线与指令几何审计 |

决策日志保留原始字段值并删减重复历史；时间点对应本轮视频。`execution_feedback` 是决策前上一物理步的状态，不是当前决策执行后的结果。建议先查看帧 **28005**，再查看帧 **29901**。
