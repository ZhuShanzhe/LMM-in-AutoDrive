# 王皓然决策执行链路交接

## 已完成范围

本轮只处理指令条件顺序、相对调速、安全门、FSM 和执行反馈，不修改语音、感知、
场景真值或模型权重。

- `ADJUST_SPEED` 在步骤激活帧把实测车速转换为唯一绝对目标。默认相对量为
  5 km/h，DrivingIntent 的定量接口使用 `speed_delta_mps`；同一请求后续帧、风险等待及
  恢复均复用该目标，不重复累加。
- `ControlPlanState` 升级为 `1.1.0`。每步增加 `speed_reference_kmh` 和
  `resolved_target_speed_kmh`，供控制日志、同源回放和独立评测核对。
- 物理紧急风险优先于普通停车和其他动作；风险要求减速、目标丢失或目标车道不安全时，
  不发出横向动作。高风险制动不会被换道等待超时覆盖。
- `WAITING` 仍是活动步骤，不是完成。目标恢复或目标车道获得低风险历史后，只恢复同一
  步骤；必须收到匹配请求和步骤的物理完成反馈才可推进。
- 相对目标速度完成要求连续稳定样本；在线计划默认要求保持 0.5 秒。单帧命中、等待、
  迟到反馈、错误请求 ID、超时或越过指令窗口均不能释放下一条指令。

## 给仿真与评测框架的接口

刘旭侧无需读取控制器内部变量。每帧保留以下已有 JSON 即可：

1. `WorldState`、`SemanticAlignment`、`RiskAssessment`；
2. `ControlPlanState 1.1.0`；
3. `ControlDecision 1.0.0`；
4. 有物理证据时生成的 `StepFeedback 1.0.0`；
5. `command_dispatch` 状态与事件。

建议在三场景各 1000 帧同源回放中至少检查：

- 同一 `request_id + step_id` 的 `resolved_target_speed_kmh` 全程不变；新步骤激活后才允许
  重新取 `speed_reference_kmh`；
- `decision_status=BLOCKED` 或步骤为 `WAITING` 时，不产生该步骤的 `COMPLETED`；
- 危险换道帧的 `target_lane` 必须为空，输出应为停车或当前车道降级动作；
- 恢复换道必须仍是原请求、原步骤，且目标车道风险证据为低；
- `TIMEOUT`、`ACTIVE_COMMAND_WINDOW_EXPIRED`、`FAILED`、`CANCELLED` 均不增加
  `completed_commands`；
- 完成统计使用独立真值；模型执行反馈只作为链路日志，不能代替任务验收。

指令派发会按输入契约自动选模式：全部命令都带有效 `DrivingIntent` 时使用
`completion_serial`，因此场景 2 必须完成当前计划后才派发下一条；未结构化的场景 1/3
保持 `route_latest`。配置里的 `command_dispatch_mode` 仍可显式覆盖自动选择。

协议和运行说明见 `scene_understanding/CONTROL_PLAN_EXECUTION.md` 与
`experiment/CARLA/control/COMMAND_DISPATCH.md`。

## 本机验证与边界

已完成纯 CPU 语法、JSON 和定向回归。覆盖相对目标锁存、速度稳定完成、目标丢失等待、
危险换道阻断与恢复、超时不推进、紧急风险优先级及协议扁平化。

本机没有启动 CARLA 服务，也没有 J6P 环境和三场景 1000 帧同步数据。因此这里只声明
代码和接口回归通过，不声明物理闭环、200 ms 端到端时延、功耗、512 MB 内存、FLOPs、
J6P 利用率或三场景精度达标；这些项目应由公共仿真/评测框架按同源输入补测。
