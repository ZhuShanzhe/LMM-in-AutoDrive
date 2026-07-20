# LMM-in-AutoDrive

面向 XH-202602“智能驾驶场景的大模型应用”基础赛道的协作仓库。

## 目录

- `program/`：项目计划、赛题文件和阶段任务，保存在 `main` 分支。
- `docs/baseline_research/`：已经完成的模型 baseline 调研结果。
- `structured_command_parser/`：中文驾驶指令翻译、英文意图解析和 DrivingIntent JSON 接口。

## 当前指令链路

```text
中文 ASR 文本
  -> Qwen2.5-3B 中文到英文约束翻译
  -> Qwen2.5-3B 英文指令解析
  -> DrivingIntent JSON
  -> 决策规划/CARLA 执行模块
```

运行代码的推荐入口是：

```python
from structured_command_parser import DrivingCommandService

service = DrivingCommandService.from_shared_model(
    "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
)
service.warmup()

result = service.parse_asr_text(
    "前方路口右转",
    request_id="asr-000001",
)
driving_intent = result["driving_intent"]
```

下游执行前必须检查：

```python
status = driving_intent["parse_result"]["status"]
```

仅当状态为 `VALID` 时，才可以把 `intent.steps` 交给规划与控制模块；语言模型输出不能直接转换成油门、制动或转向控制量。

完整安装方法、输入输出契约、命令行调用、状态处理、模型替换和后续评测流程见 `structured_command_parser/README.md`。机器可校验接口以 `structured_command_parser/schemas/driving_intent.schema.json` 为准。
