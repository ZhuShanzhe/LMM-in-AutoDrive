# ASR 测试模块（tests）

本目录包含可直接运行的 ASR 测试脚本；数据加载、指标计算、对照报告等**评测辅助方法**统一放在 `utils/` 下。

+ 评测能力（指标 + 对照报告）全部收录在本目录；`src/` 中不包含评测代码。
+ **指标逻辑的唯一实现位于 `src/`**：文本指标在 `src/asr/text_metrics.py`，语义槽位在 `src/asr/slots.py`；本目录的 `utils/metrics.py` 只做转发与聚合汇总，避免重复代码并保持 `src/` 可独立部署。
+ **槽位指标为可选诊断项，默认关闭**：默认指标只涉及 CER/SER，因此 `summary.json` 默认仅含 `cer` / `ser` / `latency_*` / `empty_hypotheses`；需要额外的驾驶语义诊断时加 `--enable-slots`。

## 1. 目录结构

```text
tests/
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # 数据集加载（mapping JSON 或 目录+命令文件）
│   ├── metrics.py          # 指标聚合与汇总（转发 src/asr 的实现）
│   ├── evaluator.py        # 批量转写、指标汇总、结果保存
│   └── report.py           # 优化前后对照报告（JSON + Markdown）
├── asr_test.py             # 标准普通话数据集测试
├── noise_test.py           # 噪声数据集测试（降噪前后对比）
├── dialect_test.py         # 方言数据集测试（方言归一前后对比）
└── README.md
```

## 2. 运行方式

+ 均以**仓库根目录**为工作目录运行：

```shell
python tests/asr_test.py --dataset data/wav_files/standard/mapping.json --limit 20
python tests/noise_test.py --dataset data/wav_files/noise/standard_noise/mapping.json --limit 20
python tests/dialect_test.py --dataset data/wav_files/dialect/mapping.json --dialect sichuan --limit 20
```

+ `--dataset` 可以是：
  + mapping JSON（列表，元素含 `audio_file` 与 `text`）；
  + 音频目录（此时需额外提供 `--commands-file`，按顺序与命令配对）。

+ 加载时会保留 mapping 的附加字段（`dialect` / `speaker` / `noise_type` / `snr_db` / `dataset` 等），供筛选与分组统计使用。
+ `dialect_test.py` 的方言筛选：`--only-dialect <name>` 只保留 `dialect` 匹配的记录，`--dialect <name>` 只决定使用哪本方言词典；两者配合才能保证“词典与音频同源”。
  + 不指定 `--only-dialect` 而在混合方言集上运行时，会打印提醒告警；
  + 筛选发生在 `--limit` 截断**之前**，因此 `--only-dialect sichuan --limit 20` 得到的是 20 条四川话。

+ 三个脚本采用**统一的实时策略**（同一套参数、同一套输出行为）：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--progress-every` | `10` | 每 N 条打印一行进度（含当前 CER / SER / 空识别数 / 最近识别文本） |
| `--save-every` | `50` | 每 N 条刷新一次 `summary.json` |
| `--no-progress` | 关闭 | 静默模式，不打印进度 |
| `--log-file` | `logs/tests/<script>.log` | 本次运行的日志文件；传空字符串则不落盘 |
| `--enable-slots` | 关闭 | 额外输出可选的槽位诊断指标 |

+ 示例：

```shell
# 小样本调试：每条都打印、每 5 条刷新汇总
python tests/asr_test.py --dataset data/wav_files/standard/mapping.json --limit 20 --progress-every 1 --save-every 5
# 静默运行
python tests/asr_test.py --dataset data/wav_files/standard/mapping.json --no-progress
```

## 3. 指标说明

+ **所有指标均为标点不敏感**：计算前先剥离标点（Unicode 类别 `P*`，中英文标点统一处理），仅保留数字之间的小数点（如 `20.5`），避免标点差异干扰 `cer` / `ser`；`details.json` 的 `correct` 与 `empty_hypotheses` 采用同一规则。
+ 指标实现统一收敛到 `src/asr/text_metrics.py`，`tests/utils/metrics.py`、`training/utils.py`、`src/asr/compression/quantize.py` 均复用同一实现。

| 指标 | 含义 |
|:---|:---|
| `cer` | 字符错误率（语料级）：`Σ编辑距离 / Σ参考字符数` |
| `ser` | 句子错误率：存在任一字符错误的句子占比 |
| `direction_match` | *可选*：方向槽位是否完全一致（左/右/直行/掉头/变道） |
| `negation_match` | *可选*：否定语义是否一致（漏掉“不”会反转指令） |
| `quantity_recall` | *可选*：数字与单位槽位的召回率（速度/距离/车道等） |
| `critical_score` | *可选*：关键槽位综合分 `(方向 + 否定 + 数字) / 3` |
| `latency_mean_ms` / `latency_p95_ms` | 平均 / P95 单条耗时（毫秒） |
| `empty_hypotheses` | 去除标点后仍为空的样本数 |

## 4. 输出

+ 每个脚本在 `--save-dir` 下生成：
  + `summary.json`：指标汇总（**每 `--save-every` 条实时刷新**，中断也能看到已完成部分）；
  + `details.jsonl`：逐样本结果，**每条追加并立即 flush**，可在运行过程中实时查看；
  + `details.json`：全部样本结果，结束时一次性写出；存在 `dialect` 字段时会随结果一并记录，便于按方言分组统计；
+ `noise_test.py` 与 `dialect_test.py` 会依次跑两个阶段（raw / 优化后），进度行带阶段标签便于区分（如 `[noise_test:raw]`、`[dialect_test:normalized]`），并额外生成 `comparison.json`（优化前后对比与增益）。

+ `utils/report.py` 提供通用对照能力，可生成 JSON + Markdown 报告：
  + `compare_backends(references, baseline_texts, optimized_texts, budget=0.03, include_slots=False)`：同源指标对比，校验 CER 预算；`include_slots=True` 时额外校验关键槽位；
  + `to_markdown(result, latency=None)`：渲染为 Markdown 表格；
  + `save_report(result, path, latency=None)`：同时写出 `.json` 与 `.md`。

## 5. 说明

+ 测试脚本读取 `src/asr/` 的真实接口（`Qwen3ASRService`、`build_optimizer`、`DialectNormalizer`、`ASROutputGuard`），不依赖未实现的模块。