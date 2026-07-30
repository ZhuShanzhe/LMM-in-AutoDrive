# XH-202602 智能驾驶多模态闭环系统

本分支整理基础语音操控与复杂避障场景的提交代码、模型接口、运行配置和代表性日志。

## 系统链路

```text
语音指令
  -> ASR 与术语约束翻译
  -> ModernBERT / DrivingIntent 1.2
  -> 实时感知、场景语义融合与风险评估
  -> 规则 FSM 或 VLA + FSM
  -> ControlDecision 1.0
  -> 路线适配、PID 与 CARLA VehicleControl
  -> 指标、事件和视频记录
```

## 目录

| 路径 | 作用 |
|---|---|
| `automatic_speech_recognition/` | 语音识别、降噪、翻译与语音数据处理 |
| `structured_command_parser/` | 将文本指令解析为 `DrivingIntent 1.2` |
| `scene_understanding/` | 实时感知、实体对齐、风险评估和高层动作决策 |
| `lightweight_vla_adapter/` | VLA 特征编码、推理、训练、评测与安全门 |
| `experiment/CARLA/` | 场景、交通流、闭环控制、日志和可视化 |
| `基础赛道提交材料/` | 技术文档、模型说明与代表性仿真记录 |

## 场景入口

| 场景 | 入口 | 配置 |
|---|---|---|
| 基础语音操控 5 km | `experiment/CARLA/run_control_experiment.py` | `experiment/CARLA/configs/basic_voice_urban_5km.json` |
| 复杂避障 8 km | `experiment/CARLA/run_scene2_closed_loop.py` | `experiment/CARLA/configs/scene_2_submission_8_runtime.json` |

代码文件的详细职责见 [CARLA 模块索引](experiment/CARLA/README.md)，代表性运行记录见
[仿真测试报告](基础赛道提交材料/06_仿真测试全量报告/仿真测试报告.md)。
