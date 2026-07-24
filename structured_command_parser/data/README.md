# 数据集说明

当前阶段只使用英文数据集归纳驾驶术语和解析规则。中文语音识别文本到英文的翻译将在英文解析器稳定后单独处理。

## 统一入口

```text
/root/autodl-tmp/LMM-in-AutoDrive/structured_command_parser/data/corpus/
```

## 英文数据规模

| 数据 | 规模 | 当前用途 |
|---|---:|---|
| Talk2Car | 11,959条英文命令 | 自然驾驶命令、导航目标和口语表达 |
| SimLingo Dreamer | 23,381,458次出现，去重后587,005条 | 动作、槽位、长尾句式、歧义和不支持表达 |
| SimLingo Commentary | 2,085,459次出现，去重后63,736条 | 场景对象和驾驶上下文术语，不作为乘客命令 |

SimLingo语言归档共77个、约11.18GB。没有下载约1.17TB的视觉数据，因为当前工作只需要语言标注。

## 当前目录

```text
corpus/
├── raw/                                # 官方原始英文标注，只读
│   ├── talk2car/
│   └── simlingo/
├── processed/                          # 全量英文统一JSONL
│   ├── talk2car_all.jsonl
│   ├── simlingo_dreamer_unique.jsonl
│   └── simlingo_commentary_unique.jsonl
├── knowledge_mining/
│   ├── representative_samples.jsonl    # 1,000条分层代表样本
│   ├── gpt_inputs/                     # 20个GPT输入批次，每批50条
│   ├── gpt_outputs/
│   │   ├── raw/                        # GPT原始候选结果
│   │   ├── validated/                  # 格式校验通过
│   │   ├── errors/                     # 校验失败
│   │   └── merged/                     # 合并去重后的候选
│   ├── artifacts/
│   │   ├── candidates/                 # 待人工审核术语和规则
│   │   └── approved/                   # 人工批准的术语和规则
│   └── manifests/selection_summary.json
├── deferred_translation/               # 旧双语种子和历史中文回归样本
├── verified/                           # 后续英文解析金标准
├── model_ready/                        # 后续英文小模型train/dev/test
└── manifests/                          # 下载、处理和审计清单
```

## GPT输入

从以下文件开始：

```text
data/corpus/knowledge_mining/gpt_inputs/batch_001.input.jsonl
```

对应Prompt：

```text
structured_command_parser/prompts/ENGLISH_TERMINOLOGY_RULE_MINING_PROMPT.md
```

每批输出一个JSON文件，保存为：

```text
data/corpus/knowledge_mining/gpt_outputs/raw/batch_001.output.json
```

GPT只归纳英文术语候选和解析规则候选，不翻译中文、不逐条生成最终训练标签，也不判断当前道路环境能否执行指令。

## 数据边界

- `weak_hints`来自数据源元信息或启发式映射，不是金标准。
- Commentary只提供上下文术语和语义边界。
- GPT候选必须人工审核后才能进入`artifacts/approved/`。
- 不直接执行GPT生成的代码或正则表达式。
- 中文翻译数据当前保存在`deferred_translation/`，不参与本阶段处理。
- 后续英文金标准必须按来源、场景和模板分组切分，不能随机逐句切分。

详细流程见`structured_command_parser/ENGLISH_KNOWLEDGE_MINING.md`。

## ModernBERT伪标签数据

全量生成目录位于：

```text
data/processed/english_pseudolabels/
├── all.jsonl                         # 662,700条完整伪标签
├── train.jsonl                       # 463,890条，70%
├── validation.jsonl                  # 132,540条，20%
├── test.jsonl                        # 66,270条，10%
├── train_sparse_augmentation.jsonl   # 108条稀有动作增强，仅训练使用
├── label_schema.json
└── manifest.json
```

切分使用规范化英文文本的SHA1分组，相同文本不会跨越训练、验证和测试集。每条记录保留`source`、`text_en`、完整`expected`、伪标签来源、训练权重和原始关键元数据。

`SOURCE_MODE_AND_RULE`权重最高；纯规则标签按来源降权；无法可靠解析的文本标为`NEEDS_CLARIFICATION`，不编造动作；Commentary仅以低权重参与。`train_sparse_augmentation.jsonl`单独补充`EMERGENCY_BRAKE`和`CANCEL`，不进入验证或测试。

这些文件是训练用伪标签，不是人工金标准。当前验证和测试指标只能解释为教师一致率，不能用来声明真实解析准确率。

重新生成：

```bash
python -m structured_command_parser.scripts.build_full_english_pseudolabels
```
