# 中文驾驶指令翻译与结构化解析模块

本模块接收语音识别模块输出的中文文本，完成约束翻译和英文指令解析，最终输出符合 `DrivingIntent` Schema 的 JSON。ASR、CARLA 控制器和车辆规划器不在本模块内部。

## 架构位置

```text
麦克风/语音流
  -> ASR 模块
  -> {request_id, text, modality}
  -> DrivingCommandService（本模块）
       -> 中文规范化
       -> Qwen2.5-3B 中文到英文约束翻译
       -> Qwen2.5-3B 英文指令解析
       -> 语义归一化、安全约束、Schema 校验
  -> result["driving_intent"]
  -> 决策规划/CARLA 执行模块
```

翻译和解析默认共享同一个模型实例。应用进程应只创建一个 `DrivingCommandService`，不要为每条 ASR 消息重新加载模型。

## 当前进展

截至 2026-07-20，本模块已经完成：

1. 固定 Python `3.12.13`、PyTorch `2.11.0+cu130` 和 RTX 5090（SM120）运行环境。
2. 下载并验证 `Qwen2.5-3B-Instruct`，保留 `Qwen2.5-1.5B-Instruct` 作为轻量对照。
3. 实现中文规范化、专业术语点对点映射、中文到英文约束翻译和英文到 DrivingIntent JSON 解析。
4. 增加歧义保护、危险/违法指令拒绝、动作顺序恢复、单位换算、空模型输出重试和 Schema 安全兜底。
5. 增加 `DrivingCommandService` 常驻接口，可直接接收 ASR 文本或 JSON 风格消息，并复用同一个模型实例。
6. 建立中文开发/冻结数据、SimLingo 英文扩展集和 Talk2Car 中文人工审核队列。
7. 完成 62 项单元与回归测试、真实权重端到端测试和中间文件清理。

当前数据规模：

| 数据 | 数量 | 状态与用途 |
|---|---:|---|
| 原中文回归集 | 30 | 参与过早期开发，只用于防止能力回退 |
| 中文多样性开发集 | 80 | 用于提示词、术语和归一化规则调试 |
| 中文多样性冻结集 | 40 | 未用于后续调参，用于一次性泛化检查 |
| SimLingo 英文扩展集 | 327 | 外部工程验证，标签由 source mode 映射 |
| Talk2Car 中文候选 | 300 | Qwen2.5-3B 机器翻译，必须人工审核 |
| SimLingo 中文候选 | 327 | Qwen2.5-3B 机器翻译，必须人工审核 |

### 已确认结果

“契约匹配率”表示预期状态、动作、方向和速度等字段子集匹配，不是完整 JSON 字符串逐字段相等。

| 测试 | 样本 | JSON 合法率 | 契约匹配率 | 术语约束通过率 | 平均时延 | P95 时延 |
|---|---:|---:|---:|---:|---:|---:|
| 原中文回归集 | 30 | 100.00% | 100.00% | 100.00% | 1530.402 ms | 3355.416 ms |
| 中文多样性开发集 | 80 | 100.00% | 98.75% | 100.00% | 1350.450 ms | 1882.555 ms |
| 中文多样性冻结集 | 40 | 100.00% | 67.50% | 95.00% | 1453.486 ms | 1941.080 ms |
| SimLingo 英文扩展集 | 327 | 100.00% | 96.64% | 不适用 | 1117.193 ms | 1637.402 ms |

冻结集 `67.50%` 是当前更可信的泛化信号。主要薄弱项是口语化减速、模糊指代、三动作组合、动作顺序和违法语义在翻译中的保留。冻结集结果产生后没有继续用它调参；若要针对这些错误改进，必须建立新的独立测试集。

原 30 条与 80 条开发集的高分不能解释为真实道路场景准确率。327 条 SimLingo 数据经过筛选并使用 source mode 构造标签，也不是官方完整基准。627 条外部中文候选虽然通过结构校验，但抽样发现车道序号误译、少量繁体字和不自然表达，人工确认前不能作为训练金标准或正式测试集。

真实模型集成测试已经确认以下链路可运行：

```text
“前方路口右转”
  -> “turn right at the upcoming junction”
  -> status=VALID
  -> action=TURN, direction=RIGHT
```

模型首次冷启动约 6 秒；表中时延为批量评测时模型常驻后的结果。

## 核心文件

```text
structured_command_parser/
├── configs/
│   ├── translation_glossary.json  # 中文驾驶术语点对点映射
│   ├── translator_prompt.txt      # 中文到英文提示词
│   └── english_parser_prompt.txt  # 英文到 JSON 提示词
├── schemas/
│   └── driving_intent.schema.json # 下游接口 Schema
├── src/
│   ├── service.py                 # 推荐的系统集成入口
│   ├── pipeline.py                # 翻译与解析编排
│   ├── translator.py              # 约束翻译
│   ├── english_parser.py          # 英文意图解析及安全归一化
│   ├── qwen_runtime.py            # 共享 Qwen 推理运行时
│   ├── factory.py                 # DrivingIntent JSON 构造
│   ├── schema_tools.py            # Schema 与语义校验
│   ├── normalizer.py              # 文本与单位规范化
│   └── llm_parser.py              # 动作到步骤的展开逻辑
├── scripts/
│   ├── parse_pipeline.py          # 单条命令调试入口
│   ├── evaluate_pipeline.py       # 中文端到端批量评测
│   ├── evaluate_english_parser.py # 英文解析批量评测
│   └── ...                        # 数据构建、翻译、审核与环境检查
├── tests/                         # 单元与回归测试
├── DRIVING_INTENT_REFERENCE.md    # 字段、枚举和下游解释
└── EXPERIMENT_REPORT.md           # 当前实验结果与限制
```

`scripts/` 中的数据和评测代码不是运行时依赖，但为后续换模型、微调、扩充术语和重新评测所必需，因此保留。

## 环境

- Python：`3.12.13`
- PyTorch：`2.11.0+cu130`
- GPU：RTX 5090（SM120）
- 当前模型：`Qwen2.5-3B-Instruct`
- 当前模型路径：`/root/autodl-tmp/models/Qwen2.5-3B-Instruct`
- Conda 环境：`/root/autodl-tmp/conda_envs/command_parser`

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser
pip install -r structured_command_parser/requirements.txt
pip install -r structured_command_parser/requirements-model.txt
```

## 推荐集成方式

在 ASR 消费进程或独立推理服务启动时创建一次服务：

```python
from structured_command_parser import DrivingCommandService

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"

command_service = DrivingCommandService.from_shared_model(
    MODEL_PATH,
    default_modality="VOICE",
    max_input_chars=512,
)

# 可选：服务启动阶段预加载权重，避免首条指令承担冷启动时间。
command_service.warmup()
```

收到 ASR 中文文本后：

```python
result = command_service.parse_asr_text(
    "看到前方行人，减速避让后向左变道",
    request_id="asr-000001",
)

driving_intent = result["driving_intent"]
status = driving_intent["parse_result"]["status"]
```

也可以直接接收 JSON 风格消息：

```python
result = command_service.handle_message(
    {
        "request_id": "asr-000002",
        "text": "前方路口右转",
        "modality": "VOICE",
    }
)
```

输入消息最小契约：

```json
{
  "request_id": "asr-000002",
  "text": "前方路口右转",
  "modality": "VOICE"
}
```

- `text` 必须是非空中文字符串。
- `request_id` 可省略；省略时模块自动生成。
- `modality` 只能是 `VOICE` 或 `TEXT`，默认 `VOICE`。
- 默认最多 512 个字符，可通过 `max_input_chars` 调整。

## 下游状态处理

下游只消费 `result["driving_intent"]`，并首先检查：

```python
status = result["driving_intent"]["parse_result"]["status"]
```

| 状态 | 下游处理 |
|---|---|
| `VALID` | 将 `intent.steps` 交给规划/控制模块执行 |
| `NEEDS_CLARIFICATION` | 不执行动作，将 `clarification_question` 返回交互模块 |
| `UNSUPPORTED` | 拒绝危险、违法或系统不支持的指令 |
| `INVALID` | 进入安全兜底，不执行模型输出 |

执行模块仍必须独立检查道路条件、交通规则、目标是否存在以及轨迹是否安全。语言模型输出不能直接转换成油门、制动或转向控制量。

## 命令行测试

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser

python -m structured_command_parser.scripts.parse_pipeline \
  "看到前方行人，减速避让后向左变道" \
  --model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --modality VOICE
```

若翻译和解析要使用不同模型：

```bash
python -m structured_command_parser.scripts.parse_pipeline \
  "前方路口右转" \
  --translator-model /path/to/translator \
  --parser-model /path/to/parser
```

## 测试

```bash
python -m unittest discover -s structured_command_parser/tests -v
```

当前共 62 项测试，覆盖 Schema、单位换算、歧义、危险指令、术语映射、复杂动作、空模型输出和服务输入契约。

## 后续更换或改进模型

### 只更换本地权重

不需要修改业务代码，只替换服务启动参数：

```python
command_service = DrivingCommandService.from_shared_model("/path/to/new-model")
```

新模型必须兼容 Hugging Face Transformers 的 `AutoTokenizer` 和 `AutoModelForCausalLM` 接口。

### 分别使用翻译模型和解析模型

```python
from structured_command_parser import CommandParserConfig, DrivingCommandService

config = CommandParserConfig(
    translator_model_path="/path/to/translation-model",
    parser_model_path="/path/to/parser-model",
)
command_service = DrivingCommandService(config)
```

路径不同会加载两个模型，需要重新评估显存占用。

### 修改术语或提示词

- 中文术语点对点映射：`configs/translation_glossary.json`
- 中文翻译提示词：`configs/translator_prompt.txt`
- 英文解析提示词：`configs/english_parser_prompt.txt`

每次修改后至少运行 62 项单元测试、原 30 条回归集、中文开发集和新的独立冻结集。不要使用现有 40 条冻结集继续调参。

### 数据与评测代码

- `build_diverse_chinese_commands.py`：生成中文开发/冻结样本。
- `prepare_external_commands.py`：筛选 SimLingo 和 Talk2Car。
- `translate_external_commands.py`：生成中文人工审核候选。
- `validate_external_commands.py`：校验候选并生成审核表。
- `evaluate_pipeline.py`：端到端评测并按语义切片统计。
- `evaluate_english_parser.py`：英文解析评测。

训练或微调前，应先完成人工审核，并重新划分 `train/dev/test`。机器翻译候选不能直接当作金标准。

## 当前结果

- 最终/对照结果仅保留以下文件：

```text
structured_command_parser/results/qwen_1_5b_verified.jsonl
structured_command_parser/results/zh_en_pipeline_30_expanded_final.jsonl
structured_command_parser/results/zh_diverse_dev_final.jsonl
structured_command_parser/results/zh_diverse_holdout_final.jsonl
structured_command_parser/results/simlingo_english_327_final.jsonl
```

上述 `results/`、模型权重、原始数据和处理数据均由 `.gitignore` 排除，不会上传 GitHub。用于后续换模型、扩数据和重新评测的源代码、提示词、术语表、Schema 与测试均已保留。

详细实验条件、逐项分析和复现命令见 `EXPERIMENT_REPORT.md`。

## 下一步

1. 人工审核 627 条中文候选，优先修正方向、车道序号、动作顺序、繁体字和危险语义。
2. 建立 300-500 条双人复核的原生中文独立测试集，冻结后禁止用于调参。
3. 接入真实 ASR，增加至少 200 条同音词、错字、漏字、口音和噪声样本。
4. 将当前冻结集失败模式转入下一版开发集，同时另建新的测试集。
5. 将 `DrivingCommandService` 封装为主控需要的 HTTP、消息队列或 ROS 接口，并进行 CARLA 联调。
