# 英文术语与解析规则归纳

## 当前阶段

当前只使用英文数据完成两项离线工作：

1. 归纳英文驾驶术语、同义表达和容易混淆的概念。
2. 归纳动作识别、槽位抽取、否定、顺序、歧义和不支持表达的解析规则。

中文语音识别文本到英文的翻译属于后续阶段。本阶段不批量生成中文翻译，也不审核中文翻译质量。

当前已完成机器归纳、原始证据核查和Schema一致性审核：31项英文术语、22条解析规则、26类完整语料表达清单。DrivingIntent接口包含25类可执行动作。配置状态为`REVIEWED_APPROVED_WITH_CORRECTIONS`，可用于生成待抽检伪标签，但不能替代独立人工金标准。

## 数据职责

| 来源 | 归纳用途 | 限制 |
|---|---|---|
| Talk2Car | 自然乘客命令、导航目标、口语表达 | 自动动作提示不是金标准 |
| SimLingo Dreamer | 动作表达、数值槽位、歧义与不支持表达 | mode和自动标签只能作为弱提示 |
| SimLingo Commentary | 场景对象、因果描述和驾驶上下文术语 | 不是乘客命令，不直接生成解析标签 |

## 代表样本

运行以下命令生成确定性的分层代表集：

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser

python -m structured_command_parser.scripts.build_english_knowledge_mining_set
```

默认生成1,000条：

- Talk2Car：500条。
- SimLingo Dreamer：300条。
- SimLingo Commentary：200条。
- 每个GPT输入批次：50条，共20批。

抽样会先把数字替换为占位符以减少坐标和速度变化造成的近重复，再按弱动作提示、Dreamer mode、场景、句子长度和出现频率进行轮转采样。固定哈希排序保证重复运行结果一致。

## GPT处理

Prompt位于：

```text
structured_command_parser/prompts/ENGLISH_TERMINOLOGY_RULE_MINING_PROMPT.md
```

输入批次位于：

```text
data/corpus/knowledge_mining/gpt_inputs/batch_001.input.jsonl
...
data/corpus/knowledge_mining/gpt_inputs/batch_020.input.jsonl
```

把每批模型返回的单个JSON对象原样保存到：

```text
data/corpus/knowledge_mining/gpt_outputs/raw/batch_001.output.json
```

模型输出只是候选知识，不是批准后的运行时配置。

## 已审核配置与覆盖清单

```text
configs/english_terminology.json
configs/english_parsing_rules.json
data/corpus/knowledge_mining/manifests/batch_evidence_analysis.json
data/corpus/knowledge_mining/manifests/full_corpus_action_inventory.json
data/corpus/knowledge_mining/manifests/curated_knowledge_validation.json
```

重新生成和校验：

```bash
python -m structured_command_parser.scripts.analyze_english_knowledge_batches
python -m structured_command_parser.scripts.analyze_full_action_inventory
python -m structured_command_parser.scripts.validate_curated_english_knowledge
```

全量盘点覆盖662,700条去重英文记录。模式计数允许重叠，只表示术语证据规模，不是分类准确率。安全动作已经纳入DrivingIntent `1.1.0`的25类动作；明确撞击、闯红灯等请求继续返回`UNSUPPORTED`。

## 审核记录

2026-07-21已对归纳结果完成AI辅助证据与一致性审核，范围不是逐条标注几十万条原始数据：

1. 同义表达是否确实属于同一动作或槽位。
2. `TURN`、`CHANGE_LANE`、`AVOID`、`OVERTAKE`、`YIELD`等边界是否清楚。
3. 规则是否会被否定词或近义反例误触发。
4. 必需槽位缺失时是否应请求澄清。
5. 多动作顺序和数值单位是否保留。
6. Commentary证据是否被错误当成乘客指令。
7. 每个候选是否能追溯到真实样本ID。

已审核并修正的文件为：

```text
configs/english_terminology.json
configs/english_parsing_rules.json
```

审核修正包括动作边界、语义执行顺序、距离槽位归属、速度单位范围、倒车入库语义，以及违法指令拒绝。配置中已记录审核时间、审核方式、修正项和限制。仍不要直接执行大模型生成的正则表达式或代码；伪标签必须抽检，金标准必须由人独立复核。

## 制动标签边界

- 明确紧急/猛烈制动，或危险上下文中的立即制动/停车：`EMERGENCY_BRAKE + EMERGENCY`。
- 尽快停车但没有危险或猛烈制动语义：`STOP + URGENT`。
- 普通完整停车：`STOP + NORMAL`。
- 普通减速或刹车：`ADJUST_SPEED(DECREASE) + NORMAL`。
- 立即刹车但没有完整停车或危险语义：`ADJUST_SPEED(DECREASE) + URGENT`。

全量语料扫描没有发现明确的紧急/猛烈制动正例。Talk2Car与Dreamer只有11条去重命令格式的紧迫停车候选；它们必须按上述边界标为`STOP + URGENT`或结合危险上下文判断，不能仅凭`now/immediately`升级为紧急制动。Dreamer中的`Brake now`和`Hit the brakes`对应`slower/slower_factor`，是`EMERGENCY_BRAKE`硬负例。

## 后续阶段

1. 用已审核术语和规则给全量英文命令生成伪标签，并分层抽检。
2. 抽取300-500条隔离英文指令，由两人复核建立独立金标准。
3. 为`EMERGENCY_BRAKE`和`CANCEL`补充定向样本。
4. 训练英文小模型，并与规则解析器进行组合评测。
5. 英文解析稳定后，再构建中文到规范英文的翻译模块和中文ASR测试集。
