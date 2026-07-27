# ModernBERT 英文驾驶指令结构化解析模块

本模块是 XH-202602 项目中的英文指令解析基线。它接收上游翻译模块输出的英文驾驶指令，通过“语义规范化、原子意图分解、ModernBERT 语义识别、场景检索、规则验证”生成符合 `DrivingIntent 1.2.0` Schema 的 JSON，再交给场景理解、规划与控制模块。

```text
中文语音 -> ASR -> 中文到规范英文翻译（上游模块）
                    -> ModernBertCommandService（本模块）
                    -> DrivingIntent JSON
                    -> 安全校验/决策规划/CARLA 控制（下游模块）
```

本模块只负责语言指令的结构化表达，不直接产生油门、制动或方向盘控制量。下游必须再次检查道路条件、交通法规、目标可见性与轨迹安全。

## 当前基线

- Backbone：`ModernBERT-base`
- Python：`3.12.13`
- PyTorch：`2.11.0+cu130`
- Transformers：`4.57.6`
- 训练/测试 GPU：RTX 5090，计算能力 `SM120`
- 当前组合语义模型路径：`/root/autodl-tmp/models/modernbert-drive-command-compositional`
- 输入：翻译完成的英文文本
- 输出：`DrivingIntent 1.2.0` JSON
- 动作空间：25 类
- 默认推理设备：CUDA，BF16

常见且边界明确的中文指令可以在系统编排层继续走规则短路；未命中规则时再翻译为英文并进入本模块。ModernBERT 是英文解析的唯一默认模型。

## 当前架构

```text
ASR 原文 + 英文翻译
        |
        v
语义规范化
  - 同义改写、ASR 同音候选、否定范围、指代检查
        |
        v
原子意图分解
  - 动作顺序、局部方向、条件和约束
        |
        v
ModernBERT 多任务头 + 实体/关系 token 头
        |
        v
DrivingIntent 1.2.0 可组合意图图
  - entities[]、steps[]、target_ref、goal_conditions[]
        |
        v
scene_understanding 场景实体检索与歧义拒绝
        |
        v
Schema + 语义规则 + 安全能力边界验证
```

动作、方向、目标、条件、顺序和约束分别表示，不通过持续增加整句分类标签来覆盖长尾。确定性分解器只接管有明确词法证据的动作；其余输入仍由 ModernBERT 分类头回退。场景检索不重新解析 `raw_text`，而是使用共享 `target_ref`、规范属性和开放描述匹配 `WorldState`。

设计参考了 [Talk2Car](https://arxiv.org/abs/1909.10838) 的对象指代驾驶命令、[DriveLM](https://arxiv.org/abs/2312.14150) 的图式驾驶语言表示，以及 [CFQ](https://arxiv.org/abs/1912.09713) 和 [COGS](https://arxiv.org/abs/2010.05465) 的组合泛化评测思想。这里是面向本项目接口和时延约束的工程实现，不声称复现这些论文。

## 与上一版不同

上一版输出 `DrivingIntent 1.1.0`，以整句动作分类和每步独立 `target` 为主。当前
`1.2.0` 版本完成以下变更：

- 增加规范化审计，显式记录同义改写、否定/修正、ASR 同音候选和未解决指代；
- 增加共享 `entities[]`、稳定 `entity_id` 和步骤 `target_ref`，多步骤不再重复构造
  同一个语言目标；
- 增加 `goal_conditions[]`，把 `before the red truck`、`after the second
  junction`、`keep a safe distance` 等限定条件保留为可执行关系；
- 采用“规范化 + 原子分解 + ModernBERT 分类/token 头 + 规则验证”，支持未见同义
  表达、多动作顺序、条件指令、否定修正、相对描述和语音噪声拒绝；
- 场景模块只读取结构化实体和关系，不再重新解析 `raw_text`；歧义、能力不足和未解决
  指代均明确失败或请求澄清。

## 许可边界

基础 ModernBERT 使用 Apache-2.0；Talk2Car 数据使用 CC BY-NC-SA 4.0；SimLingo 使用 Wayve 自定义非商业许可。由于最终权重使用了后两者的文本数据，Hugging Face 模型仓库必须标记为 `license: other`，只允许免费的非商业学术、科研、教学和个人实验，并附带完整上游许可与署名。

SimLingo 条款禁止将其数据或训练模型用于车辆/机器人运行及高风险用途，且没有明确给 CARLA 仿真闭环豁免。因此在取得 Wayve 书面许可前，本权重仅用于离线语言解析与接口联调，不进入 CARLA 闭环车辆控制。另一条可行路线是使用许可允许闭环仿真的数据重新训练模型。

## 仓库结构

```text
structured_command_parser/
├── configs/
│   ├── english_terminology.json       # 已审核英文术语
│   └── english_parsing_rules.json     # 已审核解析规则
├── examples/                          # 可直接校验的输出示例
├── schemas/
│   └── driving_intent.schema.json     # 输出接口 Schema
├── scripts/
│   ├── parse_english.py               # ModernBERT 单条推理入口
│   ├── build_full_english_pseudolabels.py
│   ├── train_modernbert_parser.py
│   ├── calibrate_modernbert_thresholds.py
│   ├── evaluate_modernbert_classifier.py
│   ├── evaluate_modernbert_service.py
│   ├── build_compositional_generalization_data.py
│   ├── train_semantic_token_head.py
│   ├── evaluate_compositional_pipeline.py
│   ├── evaluate_parser_alignment.py
│   ├── validate_examples.py
│   └── validate_curated_english_knowledge.py
├── src/
│   ├── modernbert_service.py          # 推荐的系统集成入口
│   ├── modernbert_parser.py           # 推理、阈值和 JSON 构造
│   ├── modernbert_model.py            # Backbone、六个分类头和语义 token 头
│   ├── semantic_normalizer.py         # 同义、否定、ASR 和指代规范化
│   ├── semantic_decomposer.py         # 高精度原子动作分解
│   ├── compositional_frame.py         # 实体、关系与条件图
│   ├── modernbert_labels.py           # 标签顺序
│   └── schema_tools.py                # Schema 与语义校验
├── tests/                             # 单元与回归测试
├── DRIVING_INTENT_REFERENCE.md        # 字段和下游解释
├── ENGLISH_KNOWLEDGE_MINING.md        # 数据与规则归纳过程
└── requirements-modernbert.txt        # 可复现环境依赖
```

数据构建、训练和审核脚本不是运行时依赖；仓库保留它们用于复现实验、重做伪标签和微调模型。

## 配置运行环境

以下步骤在 AutoDL RTX 5090 容器中验证通过。Conda 环境必须放在数据盘，避免占用系统盘。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
source /etc/network_turbo

conda create -p /root/autodl-tmp/conda_envs/command_parser \
  python=3.12.13 -y
conda activate /root/autodl-tmp/conda_envs/command_parser

cd /root/autodl-tmp/LMM-in-AutoDrive
python -m pip install --upgrade pip
pip install -r structured_command_parser/requirements-modernbert.txt
```

验证 CUDA 与 SM120：

```bash
python -c "import torch, transformers; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0)); print('transformers:', transformers.__version__)"
```

预期关键输出：

```text
torch: 2.11.0+cu130
cuda: 13.0
gpu: NVIDIA GeForce RTX 5090
capability: (12, 0)
transformers: 4.57.6
```

## 准备模型权重

GitHub 不上传模型权重。最终权重发布在 Hugging Face：

- 模型页面：<https://huggingface.co/UNIC0RN-Zhu/modernbert-drive-command-base>
- 仓库 ID：`UNIC0RN-Zhu/modernbert-drive-command-base`
- License：`research-only-non-commercial`（Hugging Face 类型为 `other`）

AutoDL 上启用网络加速并下载完整模型：

```bash
source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/command_parser

hf download \
  UNIC0RN-Zhu/modernbert-drive-command-base \
  --repo-type model \
  --local-dir /root/autodl-tmp/models/modernbert-drive-command-compositional
```

公开仓库不要求登录；私有仓库需要先运行 `hf auth login` 并获得读取权限。下载后的目录结构应为：

```text
/root/autodl-tmp/models/modernbert-drive-command-compositional/
├── config.json
├── model.safetensors
├── multitask_heads.pt
├── semantic_token_head.pt
├── semantic_token_head_metrics.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── label_schema.json
├── inference_config.json
├── training_summary.json
├── test_metrics_calibrated.json
├── LICENSE
├── NOTICE
├── SHA256SUMS
├── LICENSE_CHECKSUMS
├── README.md
└── licenses/
```

其中 `model.safetensors` 和 `multitask_heads.pt` 必须来自本项目微调结果，不能只下载未经微调的 ModernBERT-base 代替。`semantic_token_head.pt` 是 1.2.0 组合语义扩展；缺少时程序会使用确定性实体/关系回退，但不会得到本文记录的完整模型结果。该扩展权重、验证指标和校验和已与基础权重一同发布在上述 Hugging Face 仓库。

验证下载完整性：

```bash
cd /root/autodl-tmp/models/modernbert-drive-command-compositional
sha256sum -c SHA256SUMS
sha256sum -c LICENSE_CHECKSUMS
export MODERNBERT_MODEL_PATH=$PWD
```

```bash
export MODERNBERT_MODEL_PATH=/root/autodl-tmp/models/modernbert-drive-command-compositional

test -s "$MODERNBERT_MODEL_PATH/model.safetensors"
test -s "$MODERNBERT_MODEL_PATH/multitask_heads.pt"
test -s "$MODERNBERT_MODEL_PATH/semantic_token_head.pt"
test -s "$MODERNBERT_MODEL_PATH/inference_config.json"
```

只有重新训练时才需要下载原始 Backbone：

```bash
hf download answerdotai/ModernBERT-base \
  --local-dir /root/autodl-tmp/models/ModernBERT-base
```

## 命令行运行

命令行用于单条调试，会先预热模型，再统计实际解析时延：

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser
export MODERNBERT_MODEL_PATH=/root/autodl-tmp/models/modernbert-drive-command-compositional

python -m structured_command_parser.scripts.parse_english \
  "Slow down and stop before the red truck." \
  --request-id demo-0001 \
  --modality TEXT
```

CPU 调试可以增加 `--device cpu`，但正式时延测试使用 CUDA。

## Python 集成

服务应在进程启动时创建一次并预热。不要为每条请求重新加载权重。

RTX 5090 上一次性预热不计入在线请求时延。组合语义版本在 175 条挑战集上的常驻推理平均时延为 `45.63ms`、P95 为 `87.02ms`、最大值为 `94.89ms`，满足本项目 100ms 单模块预算；该数字不包含 ASR、翻译、视觉感知和控制。

```python
from structured_command_parser import ModernBertCommandService

command_parser = ModernBertCommandService(
    "/root/autodl-tmp/models/modernbert-drive-command-compositional",
    device="cuda",
    max_input_chars=512,
)
command_parser.warmup()
```

直接解析英文文本：

```python
result = command_parser.parse_text(
    "Slow down and stop before the red truck.",
    request_id="translator-0001",
    modality="TEXT",
)
```

接收上游翻译模块的 JSON 风格消息：

```python
result = command_parser.handle_message(
    {
        "request_id": "translator-0002",
        "text": "Follow the vehicle ahead, but keep a safe distance.",
        "language": "en-US",
        "modality": "VOICE",
        "source_text": "跟上前面那辆车，但别太近",
        "source_language": "zh-CN",
    }
)
```

`modality` 表示原始输入来自语音还是文本；传给本模块的 `text` 始终必须是英文。

## 输入接口

```json
{
  "request_id": "translator-0002",
  "text": "Change to the left lane after the blue car passes.",
  "language": "en-US",
  "modality": "VOICE",
  "source_text": "跟上前面那辆车，但别太近",
  "source_language": "zh-CN"
}
```

| 字段 | 必需 | 约束 |
|---|---|---|
| `text` | 是 | 非空英文字符串，默认不超过 512 字符 |
| `request_id` | 否 | 字符串；建议沿用 ASR 请求 ID |
| `language` | 否 | `en`、`en-US` 或 `en-GB`，默认 `en-US` |
| `modality` | 否 | `VOICE` 或 `TEXT`，默认 `TEXT` |
| `source_text` | 否 | ASR 原始文本；提供后原样写入 `input.raw_text`，用于审计和噪声确认 |
| `source_language` | 否 | `source_text` 的 BCP 47 语言标签，例如 `zh-CN` |

无效类型、空文本、非英文语言标记和过长输入会在模型推理前抛出异常。

## 输出接口

服务直接返回 `DrivingIntent 1.2.0` 文档，不增加额外包装层：

```json
{
  "schema_version": "1.2.0",
  "request_id": "translator-0001",
  "input": {
    "modality": "TEXT",
    "language": "en-US",
    "raw_text": "Slow down and stop before the red truck.",
    "normalized_text": "Slow down and stop before the red truck."
  },
  "normalization": {
    "edits": [],
    "unresolved_references": []
  },
  "intent": {
    "category": "BASIC_CONTROL",
    "urgency": "NORMAL",
    "entities": [
      {
        "entity_id": "target_1",
        "type": "VEHICLE",
        "relation": "AHEAD",
        "description": "the red truck",
        "canonical_attributes": {
          "color": "RED",
          "vehicle_subtype": "TRUCK"
        },
        "open_descriptors": [],
        "source_span": "the red truck"
      }
    ],
    "suppressed_intents": [],
    "steps": [
      {
        "step_id": "step_1",
        "action": "ADJUST_SPEED",
        "parameters": {"change": "DECREASE"},
        "trigger": {"type": "IMMEDIATE"},
        "depends_on": [],
        "preconditions": [],
        "on_blocked": "SAFE_STOP"
      },
      {
        "step_id": "step_2",
        "action": "STOP",
        "parameters": {},
        "trigger": {"type": "AFTER_STEP", "step_id": "step_1"},
        "depends_on": ["step_1"],
        "preconditions": [],
        "on_blocked": "SAFE_STOP",
        "target_ref": "target_1",
        "goal_conditions": [
          {
            "predicate": "BEFORE",
            "subject": "EGO",
            "object": "target_1",
            "source_span": "before"
          }
        ],
        "completion": {"type": "STOPPED_BEFORE_TARGET"}
      }
    ],
    "constraints": {
      "safety_first": true,
      "obey_traffic_rules": true,
      "driving_style": "NORMAL"
    }
  },
  "parse_result": {
    "status": "VALID",
    "method": "HYBRID",
    "model": "modernbert-drive-command-compositional",
    "confidence": 0.9238,
    "missing_slots": [],
    "warnings": [],
    "latency_ms": 15.448
  }
}
```

该示例来自 RTX 5090 上的实际验证运行；置信度和时延会随输入及硬件变化。完整字段约束见 `schemas/driving_intent.schema.json` 和 `DRIVING_INTENT_REFERENCE.md`。

下游首先检查 `parse_result.status`：

| 状态 | 下游处理 |
|---|---|
| `VALID` | 校验场景安全后执行 `intent.steps` |
| `NEEDS_CLARIFICATION` | 不执行动作，向交互模块请求补充信息 |
| `UNSUPPORTED` | 拒绝危险、违法或系统不支持的指令 |
| `INVALID` | 进入安全兜底，不执行输出 |

## 训练与校准复现

全量语料生成 662,700 条伪标签，规范化文本按 SHA1 分组后再切分，确保相同文本不会跨训练、验证和测试集。

```bash
python -m structured_command_parser.scripts.build_full_english_pseudolabels

# 第一阶段
python -m structured_command_parser.scripts.train_modernbert_parser \
  --base-model /root/autodl-tmp/models/ModernBERT-base \
  --output /root/autodl-tmp/models/modernbert-drive-command-stage1 \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 3e-5

# 第二阶段
python -m structured_command_parser.scripts.train_modernbert_parser \
  --base-model /root/autodl-tmp/models/modernbert-drive-command-stage1 \
  --output /root/autodl-tmp/models/modernbert-drive-command-base \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 1e-5

python -m structured_command_parser.scripts.calibrate_modernbert_thresholds \
  --model /root/autodl-tmp/models/modernbert-drive-command-base

python -m structured_command_parser.scripts.evaluate_modernbert_classifier \
  --model /root/autodl-tmp/models/modernbert-drive-command-base \
  --report /root/autodl-tmp/models/modernbert-drive-command-base/test_metrics_calibrated.json

# 构造组合泛化数据，训练 1.2.0 实体/关系 token 头
python -m structured_command_parser.scripts.build_compositional_generalization_data
python -m structured_command_parser.scripts.train_semantic_token_head \
  --base-model /root/autodl-tmp/models/modernbert-drive-command-base \
  --output /root/autodl-tmp/models/modernbert-drive-command-compositional \
  --epochs 8

# 开发挑战集回归与解析-语义对齐联调
python -m structured_command_parser.scripts.evaluate_compositional_pipeline \
  --model /root/autodl-tmp/models/modernbert-drive-command-compositional
python -m structured_command_parser.scripts.evaluate_parser_alignment \
  --model /root/autodl-tmp/models/modernbert-drive-command-compositional
```

基础切分为 `463,890 / 132,540 / 66,270`；训练时额外加入 108 条稀有动作定向样本，实际训练行数为 `463,998`。

## 已完成实验

| 路线 | 数据/范围 | 主要结果 | 结论 |
|---|---|---|---|
| 纯中文规则 | 40 条旧冻结集 | 契约匹配率 40.00%，P95 1.71ms | 很快，但自然表达覆盖不足 |
| 规则 + BGE-small 语义回退 | 同一旧冻结集 | 契约匹配率 70.00%，P95 9.98ms | 覆盖提高，但仍不够稳定 |
| 双生成模型翻译与解析 | 40 条中文冻结集 | 契约匹配率 67.50%，P95 1,941.08ms | 时延不满足实时预算 |
| Qwen3-0.6B 英文解析对照 | 50 条分层长尾样本 | 动作 exact match 90.00%，micro-F1 97.51%，P95 5,707.26ms | 长尾推理较强，但不能进入实时链路 |
| ModernBERT 第一阶段 | 66,270 条测试集 | 动作 exact match 98.59%，micro-F1 97.57% | 轻量模型已具备较好教师一致率 |
| ModernBERT 第二阶段 + 校准 | 66,270 条测试集 | 动作 exact match 98.70%，micro-F1 97.78% | 作为最终英文解析基线 |
| ModernBERT 端到端解析 | 测试集前 1,000 条 | 动作 exact match 99.30%，P95 12.97ms | 模型常驻后满足 50ms 解析预算 |
| ModernBERT 分层长尾检查 | 50 条 | 动作 exact match 70.00%，micro-F1 88.32%，P95 19.01ms | 复杂多动作仍是主要短板，但时延满足预算 |
| 1.2.0 组合语义首轮 | 175 条平衡开发挑战集 | 图完全匹配 57.71%，状态/动作/方向 97.71%，关系谓词 61.71% | 暴露条件关系、多动作依赖和否定修正缺口 |
| 语义 token 头 | 3,288/1,065 训练/验证样本 | 验证 token F1 99.49%，8 epoch 用时 23.44s | 冻结 Backbone 的实体/关系辅助头 |
| 1.2.0 回归闭环 | 同一 175 条开发挑战集 | 图、状态、动作、方向、关系和实体均 100%；P95 87.02ms | 修复后的开发回归结果，不是独立泛化准确率 |
| 解析-语义对齐联调 | 5 条定向场景 | 5/5 符合预期；最终复测解析平均 38.22ms、最大 74.43ms；对齐平均 0.59ms、最大 2.15ms | 唯一目标成功，歧义/路线能力不足/ASR 疑义/未解析指代均被拒绝或澄清 |

最终完整测试集还包括：状态准确率 `99.39%`、类别准确率 `99.15%`、紧急度准确率 `99.91%`、方向 exact match `99.75%`、速度变化准确率 `99.82%`。

基础大测试集准确率是模型与伪标签教师的一致率，不等同于人工金标准准确率或真实道路准确率。175 条组合挑战集用于发现问题后又参与规则修复，因此其最终 100% 只能证明这些已知现象完成回归闭环，不能作为未见组合泛化结果。当前没有可用于声称真实泛化准确率的双人复核、完全隔离英文金标准。

联调覆盖五类关键行为：

- `the red truck` 唯一匹配时，解析为共享实体和 `BEFORE` 目标条件，对齐状态为 `COMPLETE`；
- 两辆同属性红色卡车同时出现时，对齐拒绝猜测并返回歧义；
- `the second junction` 需要路线级序数定位，当前场景能力不足时明确失败；
- 中文 ASR “前方路口又转”即使译文给出右转，也因同音修正需要确认而返回 `NEEDS_CLARIFICATION`；
- `Follow it...` 没有可解析先行指代时返回 `NEEDS_CLARIFICATION`，不会把未知对象交给控制模块。

## 测试

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser

python -m unittest discover -s structured_command_parser/tests -v
python -m structured_command_parser.scripts.validate_examples
python -m structured_command_parser.scripts.validate_curated_english_knowledge
python -m structured_command_parser.scripts.evaluate_parser_alignment \
  --model /root/autodl-tmp/models/modernbert-drive-command-compositional
git diff --check
```

当前指令解析单元与回归测试共 113 项，4 个 Schema 示例全部通过；联调所依赖的
场景理解模块 186 项测试也全部通过。覆盖输入消息契约、Schema、动作边界、歧义与
危险指令、否定和修正、ASR 疑义、共享实体引用、条件关系、多动作顺序、
ModernBERT 默认后端、伪标签防泄漏切分、规则短路、模型服务预热和线程安全入口。

## Git 上传边界

应上传：

- `src/` 运行时代码
- `configs/` 术语和规则
- `schemas/` 输出契约
- `examples/` 接口示例
- `scripts/` 训练、校准、评测和命令行入口
- `tests/` 测试
- `requirements-modernbert.txt`
- README 与接口参考文档

不上传：

- 原始数据和伪标签 JSONL
- ModernBERT 权重与检查点
- 训练日志、缓存和评测输出目录
- Conda 环境

`.gitignore` 已按以上边界配置。模型权重通过 Hugging Face 分发，代码通过 GitHub 协作。
