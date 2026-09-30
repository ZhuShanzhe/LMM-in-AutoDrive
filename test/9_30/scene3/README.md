# 场景 3 测试结果与问题定位

本轮车辆沿路线前进 **775.27 / 6,000.41 m**，最高观测车速 **31.66 km/h**。碰撞为 0，有 **4 次虚线压线**，禁止跨越线侵入为 0。车辆在首条指令的 900 m 触发点之前持续停车，以 `route_stalled` 结束；8 条预设指令和 7 个事件均未验证。

## 主要问题

车辆前段能正常加速，后段在信号灯附近停住。日志表明：即使灯态识别已为绿灯、信号灯约束已释放，风险与纵向约束仍会阻止重新起步。

1. **约 03:03.15，帧 85041：高风险覆盖前进提案。** 灯态为 `GREEN`，状态为 `observed_green_release`。基础模型目标约 24.97 km/h、低风险概率约 0.83；时序模型有效输出目标约 0.91 km/h、高风险概率约 0.875，最终动作被门控改为紧急制动。
2. **约 03:04.05，帧 85059：恢复目标再次被压为零。** 恢复逻辑给出 12 km/h 目标，但纵向约束处于 `RISK_SPEED_CAP`，将目标限制为不超过当前车速。当时车速为 0，因此最终目标仍为 0，且 `allow_positive_acceleration=false`。原因码虽然是 `stationary_high_risk_liveness`，实际并未恢复行驶。
3. 全程有 **375 帧**同时满足“观测绿灯已释放、风险为高、实际车速低于 1 km/h、目标速度为 0”。约 **03:14.9** 后不再有大于 1 km/h 的行驶。因此持续停车不能全部解释为正常等红灯。

优先分开检查两点：**时序模型为何持续给出高风险**，以及**恢复策略与纵向风险上限是否存在互相抵消**。相关代码为 `experiment/CARLA/universal_vla_controller.py` 的模型风险、恢复策略及约束调用顺序，以及 `lightweight_vla_adapter/src/longitudinal_contract.py` 的 `enforce_longitudinal_contract`。这些记录确定了停车路径，但不能仅凭绿灯或单张画面判定所有高风险均为误报。

另有两项记录：
- 帧 **81699、81763、81775、81803** 出现虚线压线，需结合车辆轨迹核查前段横向控制。
- 启动帧 **81376** 有 1 次 `sensor_warmup_safe_hold`，后续产生 3,743 次有效决策；这次预热等待不是后段持续停车的原因。

## 日志

| 文件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 结果、风险与动作统计、压线记录和首次预热等待 |
| [logs/decision_trace.jsonl](logs/decision_trace.jsonl) | 3,743 条有效决策及 1 条预热等待，含信号灯和纵向约束字段 |
| [logs/key_frames.json](logs/key_frames.json) | 关键帧的模型输出、传感器状态及完整信号灯/执行约束 |
| [logs/control_feedback.jsonl](logs/control_feedback.jsonl) | 逐物理步的速度、车道、油门、刹车和转向反馈 |
| [logs/command_schedule.jsonl](logs/command_schedule.jsonl) | 8 条预设指令及触发位置；是计划表，不是已触发记录 |

决策日志保留原始字段值并删减重复历史；时间点对应本轮视频。`execution_feedback` 是决策前状态，`control_feedback.jsonl` 可按仿真帧关联后续执行。建议优先查看帧 **85041、85059**。本次运行于 10 月 1 日完成，随同一轮场景 1、2 结果归档。
