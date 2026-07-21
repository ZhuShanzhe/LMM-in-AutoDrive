# 中文驾驶指令翻译与结构化解析实验报告

## 实验日期

2026-07-20

## 目标与架构

```text
中文 ASR 文本 -> Qwen2.5-3B 约束翻译 -> Qwen2.5-3B 英文指令解析
             -> 通用归一化与安全约束 -> DrivingIntent JSON
```

翻译和解析共享一个模型运行时。翻译阶段不单独计算 BLEU/COMET，只检查术语约束、中文残留和端到端意图契约。

## 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32 GB，SM120 |
| Driver / CUDA | 580.105.08 / 13.0 |
| Python / PyTorch | 3.12.13 / 2.11.0+cu130 |
| 模型 | Qwen2.5-3B-Instruct |
| 模型路径 | `/root/autodl-tmp/models/Qwen2.5-3B-Instruct` |
| Conda 环境 | `/root/autodl-tmp/conda_envs/command_parser` |

## 扩样结果

| 数据 | 扩样前 | 扩样后 | 说明 |
|---|---:|---:|---|
| 中文既有回归 | 30 | 30 | 保留用于防止能力回退 |
| 中文多样性数据 | 0 | 120 | 80 条开发集 + 40 条冻结集 |
| SimLingo 英文样本 | 157 | 327 | 依据 source mode 筛选映射 |
| Talk2Car 审核队列 | 100 | 300 | 启发式标签，待人工审核 |
| 外部中文机器翻译候选 | 257 | 627 | Qwen2.5-3B 生成，待人工审核 |

中文多样性集覆盖速度、车道与导航、道路参与者、停止与元控制、复合动作、模糊指代、违法/危险指令。开发集与冻结集文本完全不重复，均明确标记为 `CURATED_SYNTHETIC` 和 `REQUIRES_HUMAN_REVIEW`。

## 最终指标

“契约匹配率”表示预期字段子集匹配，并非完整 JSON 字符串逐字段相等。

| 测试 | 样本 | JSON 合法率 | 契约匹配率 | 术语约束 | 平均时延 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| 原中文回归集 | 30 | 100.00% | 100.00% | 100.00% | 1530.402 ms | 3355.416 ms |
| 中文多样性开发集 | 80 | 100.00% | 98.75% | 100.00% | 1350.450 ms | 1882.555 ms |
| 中文多样性冻结集 | 40 | 100.00% | 67.50% | 95.00% | 1453.486 ms | 1941.080 ms |
| SimLingo 英文扩展集 | 327 | 100.00% | 96.64% | 不适用 | 1117.193 ms | 1637.402 ms |

冻结集切片结果：

| 切片 | 匹配数 | 匹配率 |
|---|---:|---:|
| speed | 5/6 | 83.33% |
| lane_navigation | 7/8 | 87.50% |
| road_user | 6/8 | 75.00% |
| complex | 4/8 | 50.00% |
| ambiguity | 2/5 | 40.00% |
| unsafe | 3/5 | 60.00% |

## 分析

扩样前 30 条开发回归达到 100%，但新增 80 条开发集首轮只有 55.00%，说明原样本范围确实过窄。通过通用术语映射、歧义保护、动作类型归一化、违法意图保留和空输出兜底，开发集提升到 98.75%，同时原 30 条回归恢复为 100%。

冻结集仅为 67.50%，应作为当前更可信的泛化信号。主要薄弱项是口语化减速、隐式方向/目标指代、三动作组合、动作顺序以及违法意图在翻译中的语义保留。冻结集结果产生后没有继续据此调参，避免把测试集变成新的开发集。

SimLingo 扩展到 327 条后为 96.64%，低于原 157 条上的 100%，新增失败主要来自不常见变道表述、模型生成空参数和少量目标速度表达。这同样说明扩大样本比维持小集合满分更有信息量。

627 条中文外部候选通过结构校验，但抽样发现车道序号误译、少量繁体字和不自然表达。它们只能作为人工审核队列，不能用于报告正式中文准确率。

## 结果文件

```text
structured_command_parser/results/zh_en_pipeline_30_expanded_final.jsonl
structured_command_parser/results/zh_diverse_dev_final.jsonl
structured_command_parser/results/zh_diverse_holdout_final.jsonl
structured_command_parser/results/simlingo_english_327_final.jsonl
```

数据文件保存在 `structured_command_parser/data/processed/`，模型、数据和结果均由 `.gitignore` 排除。

## 复现命令

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser

python -m unittest discover -s structured_command_parser/tests -v
python -m structured_command_parser.scripts.build_diverse_chinese_commands
python -m structured_command_parser.scripts.prepare_external_commands
python -m structured_command_parser.scripts.validate_external_commands

python -m structured_command_parser.scripts.evaluate_pipeline \
  --translator-model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --parser-model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --dataset structured_command_parser/data/processed/chinese_diverse_holdout.jsonl \
  --report structured_command_parser/results/zh_diverse_holdout_final.jsonl

python -m structured_command_parser.scripts.evaluate_english_parser \
  --model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --dataset structured_command_parser/data/processed/simlingo_candidates_en.jsonl \
  --report structured_command_parser/results/simlingo_english_327_final.jsonl
```

## 下一步

1. 人工审核现有 627 条中文候选，优先修正方向、车道序号、动作顺序和危险意图。
2. 新建 300-500 条原生中文人工测试集；在冻结前完成双人复核，冻结后禁止调参。
3. 接入真实 ASR，建立至少 200 条错字、同音词、漏字、口音和噪声测试集。
4. 将本轮冻结集失败模式转入下一版开发集，但必须另建新的独立测试集。
5. 使用常驻推理服务、量化和批处理降低 P95 延迟。
