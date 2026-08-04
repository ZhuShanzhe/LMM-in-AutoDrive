# XH-202602 智能驾驶多模态闭环系统

本目录是基础赛道两场景审核版，包含基础语音操控 5 km 和多模态闭环 8 km 的代码、权重、配置、代表性日志与提交文档。场景三代码和设计配置暂时保留，但不纳入本版结果或达标结论。

## 系统链路

```text
语音指令
  -> ASR、降噪与术语约束翻译
  -> ModernBERT / DrivingIntent 1.2
  -> 实时感知、场景语义融合与风险评估
  -> 规则 FSM 或 VLA + FSM
  -> ControlDecision 1.0
  -> 路线适配、PID 与 CARLA VehicleControl
  -> 事件、指标与视频记录
```

## 提交目录

| 路径 | 作用 |
|---|---|
| `automatic_speech_recognition/` | 语音识别、降噪、翻译和测试入口 |
| `structured_command_parser/` | 将文本指令解析为 `DrivingIntent 1.2` |
| `scene_understanding/` | 实时感知、实体对齐、风险评估和高层动作 |
| `lightweight_vla_adapter/` | 轻量 VLA 训练、推理、评测与安全门 |
| `experiment/CARLA/` | 场景、交通流、闭环控制、日志与可视化 |
| `models/` | 上游预训练权重、团队最终权重、许可证与校验清单 |
| `基础赛道提交材料/` | 题目要求的六类文档、报名表和两场景佐证 |

内部计划、题目原文件、基线调研、Git 历史、缓存和原始训练数据不进入本审核目录。

## 当前场景

| 场景 | 入口 | 配置 | 实测材料 |
|---|---|---|---|
| 场景一：基础语音操控 5 km | `experiment/CARLA/run_control_experiment.py` | `experiment/CARLA/configs/basic_voice_urban_5km.json` | `基础赛道提交材料/06_仿真测试全量报告/原始时序数据/scene_1_basic/` |
| 场景二：8 km 多模态闭环 | `experiment/CARLA/run_scene2_closed_loop.py` | `experiment/CARLA/configs/scene_2_submission_8_runtime.json` | `基础赛道提交材料/06_仿真测试全量报告/原始时序数据/scene_2_complex/` |

场景二随附运行记录是 8 条安全闭环指令版本，用于验证感知、VLA、FSM 和控制链路。完整复杂障碍事件配置仍保留在 `experiment/CARLA/configs/scene_2_town05_runtime.json`，本版不把简化运行结果表述为完整复杂避障验收。

## 快速检查

```bash
source submission_env.sh
sha256sum -c models/SHA256SUMS

python -m pytest -q \
  structured_command_parser/tests \
  scene_understanding/tests \
  lightweight_vla_adapter/tests \
  experiment/CARLA/tests
```

服务器审核结果：模型文件校验全部通过；上述四部分共 `455 passed，177 subtests passed`。

## 视频放置

Windows 审核目录预留：

```text
基础赛道提交材料/06_仿真测试全量报告/可视化材料/
├── scene_1_basic.mp4
└── scene_2_complex.mp4
```

两个视频已在 Windows 提交目录中归档，并随最终 ZIP 一并交付。文件级校验值见
`基础赛道提交材料/06_仿真测试全量报告/可视化材料/README.md`。
