# 模型与可复现代码

## 源码索引

| 模块 | 目录 | 主要输出 |
|---|---|---|
| 语音处理 | `automatic_speech_recognition/` | 中文指令文本与英文规范文本 |
| 指令解析 | `structured_command_parser/` | `DrivingIntent 1.2` |
| 场景理解 | `scene_understanding/` | 感知帧、语义对齐、风险与高层动作 |
| 轻量 VLA | `lightweight_vla_adapter/` | `VLADecisionProposal` |
| 仿真闭环 | `experiment/CARLA/` | `ControlDecision`、车辆控制量、事件与指标 |

## 决策路径

- **规则 FSM**：`DrivingIntent` 与场景理解结果进入高层动作映射、步骤执行器和安全监督，输出统一 `ControlDecision`。
- **VLA + FSM**：VLA 生成高层动作建议，经安全门和时序监督后输出同一 `ControlDecision` 接口。

两条路径共用 CARLA 路线适配、动作完成判定和 PID 控制器。VLA 模型核心、训练和评测代码保留在
`lightweight_vla_adapter/`，CARLA 目录仅保存闭环接入代码。

模型文件、标识和校验信息见 [模型权重与参数统计](模型权重与参数统计.md)。
