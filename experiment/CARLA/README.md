# CARLA 闭环仿真

本目录实现基础语音操控和复杂避障场景的构建、决策接入、车辆控制、日志与视频记录。

## 入口

| 文件 | 作用 |
|---|---|
| `run_control_experiment.py` | 基础场景统一运行入口，接入规则 FSM、场景理解和结构化 VLA 策略 |
| `run_scene2_closed_loop.py` | 复杂避障 8 km 全闭环入口 |
| `scene2_closed_loop.py` | 实时感知、语义融合、VLA 推理与 FSM 安全门编排 |
| `run_complex_avoidance_town05.py` | Town05 场景构建与独立演示入口 |
| `perception_fusion_adapter.py` | 将实时感知结果转换为场景语义融合输入 |
| `carla_bootstrap.py` | 加载 CARLA Python API |

## 子目录

| 目录 | 作用 |
|---|---|
| `configs/` | 路线、指令、触发点、交通流和闭环参数 |
| `scenarios/` | 基础、复杂、应急、行人和验证场景 |
| `control/` | 决策接入、路线适配、安全监督、完成判定和 PID |
| `continuous/` | 连续路线、场景事件和交通流管理 |
| `evaluation/` | 相机、HUD、事件、日志、指标和视频 |
| `perception/` | CARLA 车辆状态与传感器数据结构 |
| `tests/` | 场景、协议、决策和控制回归测试 |
| `tools/` | 配置生成、路线检查、摘要与诊断工具 |

## 决策链

### 规则 FSM

```text
DrivingIntent + 场景状态 + 风险
  -> high_level_driving_actions
  -> control_plan_executor
  -> ControlDecision
  -> route_adapter / PID
```

### VLA + FSM

```text
实时感知 + DrivingIntent
  -> StructuredBEVRasterizer
  -> LightweightVLAPipeline
  -> safety_bridge
  -> ControlDecision
  -> safety_supervisor / PID
```

两条链路使用同一 `ControlDecision` 协议，场景代码无需感知上游决策来源。

## 环境

- Linux
- Python 3.12.13
- CARLA 0.9.16
- PyTorch 2.11.0 + CUDA 13.0
- NVIDIA RTX 5090，SM120

从提交包根目录加载统一路径：

```bash
source submission_env.sh
```

CARLA 服务端应先启动并监听默认 `127.0.0.1:2000`；端口和运行参数以各入口
`--help` 及 `configs/` 内配置为准。

## 随附实测

| 场景 | 配置 | 结果 |
|---|---|---|
| 场景一 | `configs/basic_voice_urban_5km.json` | 约 5.03 km，状态 `SUCCESS`，碰撞 0，非法压线 0 |
| 场景二 | `configs/scene_2_submission_8_runtime.json` | 约 8.00 km，计划 `COMPLETED`，碰撞 0，车道侵入 0 |

场景二抽样闭环记录共 127 帧，帧管线时延 p50/p95/max 为
`29.360/35.188/65.276 ms`。其中感知 p95 为 `29.159 ms`，语义对齐 p95 为
`0.482 ms`，VLA 推理 p95 为 `3.935 ms`，VLA + FSM p95 为 `4.559 ms`。

原始摘要、指令、事件和抽样管线日志位于
`基础赛道提交材料/06_仿真测试全量报告/原始时序数据/`。

## 结果边界

随附场景二日志使用 8 条安全闭环指令配置，复杂障碍事件处于关闭状态。目录仍保留
`configs/scene_2_town05_runtime.json` 的 15 条组合指令与车辆、行人、公交站和骑行者事件设计，
但提交材料不把当前 8 km 日志表述为完整复杂避障验收。

## 回归测试

在提交包根目录执行：

```bash
python -m pytest -q experiment/CARLA/tests
```

也可按根目录 `README.md` 统一运行指令解析、场景理解、VLA 和 CARLA 四部分回归测试。
