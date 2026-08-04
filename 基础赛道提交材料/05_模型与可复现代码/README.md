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

## 环境与权重检查

在提交包根目录执行：

```bash
source submission_env.sh
sha256sum -c models/SHA256SUMS
```

模型均使用提交包内固定相对路径，不依赖原 AutoDL 主机目录。模型文件、标识、参数量和许可证见
[模型权重与参数统计](模型权重与参数统计.md)及根目录 `models/README.md`。

## 回归测试

结构化指令、场景理解、轻量 VLA 和 CARLA 控制链可统一执行：

```bash
python -m pytest -q \
  structured_command_parser/tests \
  scene_understanding/tests \
  lightweight_vla_adapter/tests \
  experiment/CARLA/tests
```

提交前服务器实测结果为 `455 passed，177 subtests passed`。

ASR 目录中的延时、噪声和方言程序是独立命令行评测入口，按
`automatic_speech_recognition/README.md` 中的命令运行。该目录包含名为 `*_test.py`
的基准程序，但它们不是根目录统一 pytest 集合的一部分。

## 复现边界

- 场景一日志对应 5 km 基础语音操控闭环。
- 场景二日志对应 8 km、8 条安全闭环指令的 VLA + FSM 运行。
- `experiment/CARLA/configs/scene_2_town05_runtime.json` 保留完整复杂事件设计，
  但随包日志不能替代该配置的完整障碍事件复测。
