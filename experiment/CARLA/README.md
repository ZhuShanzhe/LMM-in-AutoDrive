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
