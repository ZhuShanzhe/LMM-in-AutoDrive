# LMM-in-AutoDrive

面向 XH-202602“智能驾驶场景的大模型应用”基础赛道的协作仓库。

## 目录

- `program/`：项目计划、赛题文件和阶段任务，保存在 `main` 分支。
- `docs/baseline_research/`：已完成的模型 baseline 调研。
- `structured_command_parser/`：驾驶指令结构化解析模块和 `DrivingIntent 1.1.0` 接口。

## 当前指令链路

```text
中文语音 -> ASR -> 中文到规范英文翻译
                    -> ModernBERT 英文指令解析
                    -> DrivingIntent JSON
                    -> 安全校验/决策规划/CARLA 控制
```

当前英文解析基线为微调后的 ModernBERT-base。翻译模块将英文文本交给常驻的 `ModernBertCommandService`：

当前公开权重受 Talk2Car 与 SimLingo 上游非商业条款约束，仅用于离线科研评测和接口联调；在取得相应许可或换用许可允许的数据重新训练前，不用于 CARLA 闭环车辆控制。

```python
from structured_command_parser import ModernBertCommandService

service = ModernBertCommandService(
    "/root/autodl-tmp/models/modernbert-drive-command-base",
    device="cuda",
)
service.warmup()

result = service.handle_message(
    {
        "request_id": "translator-0001",
        "text": "Turn right at the upcoming junction.",
        "language": "en-US",
        "modality": "VOICE",
    }
)
```

下游首先检查 `result["parse_result"]["status"]`。只有状态为 `VALID` 时，才可以在完成场景安全检查后把 `intent.steps` 交给规划与控制模块。语言解析结果不能直接转换成油门、制动或转向控制量。

完整的 RTX 5090/SM120 环境配置、模型目录、运行命令、输入输出契约、历史实验、训练校准和测试方法见 `structured_command_parser/README.md`。机器可校验接口以 `structured_command_parser/schemas/driving_intent.schema.json` 为准。
