# 代表性运行日志

| 目录 | 内容 |
|---|---|
| `scene_1_basic/` | 5 km 基础语音操控场景的指标、事件和抽样遥测 |
| `scene_2_complex/` | 8 km 多模态 VLA + FSM 闭环的摘要、指令、事件和抽样记录 |

JSONL 文件按时间顺序保存记录；JSON 和 CSV 文件保存运行汇总与统一指标。视频位于相邻的
`可视化材料/`。为控制提交包体积，高频逐帧中间缓存不随包交付。

## 场景一文件

- `metrics.json`、`metrics.csv`：统一指标汇总；
- `events.jsonl`：指令触发、执行和车辆事件；
- `telemetry_sampled.jsonl`：抽样车辆遥测；
- `README.md`：本次运行摘要。

该次运行状态为 `SUCCESS`，路线约 5.03 km，碰撞 0，非法压线 0。

## 场景二文件

- `summary.json`：路线、动作计划、碰撞和交通流汇总；
- `commands.jsonl`：8 条输入指令及解析结果；
- `events.jsonl`：闭环事件；
- `pipeline_sampled.jsonl`：感知、语义对齐、VLA、FSM 和控制抽样时延；
- `README.md`：本次运行摘要。

该次运行完成约 8.00 km，碰撞 0、车道侵入 0。日志使用
`scene_2_submission_8_runtime.json`，其中复杂障碍事件关闭；因此它证明多模态闭环可运行，
不单独作为完整复杂避障事件覆盖率的证明。
