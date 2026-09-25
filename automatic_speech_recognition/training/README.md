# ASR 微调模块（training）

+ 面向方言与噪声鲁棒性的 Qwen3-ASR 微调模块，提供数据准备、数据增强、LoRA 微调、结构适配器（瓶颈/软前缀）与字符级 CER 评估。
+ 与推理期优化（`src/asr/optimization/`）分工清晰：本目录负责训练期（造数据、改模型），optimization 负责推理期（免训练的降噪与方言归一）。

## 1. 文件结构

```text
training/
├── __init__.py
├── train.py                 # 训练入口（method: lora | adapter）
├── evaluate.py
├── config.py 
├── utils.py
├── dataset/
│   ├── __init__.py
│   ├── translation.py       # 英 -> 中 指令构建（调用 Qwen-MT）
│   ├── build_dataset.py     # Qwen3-TTS 生成语音集（standard / dialect / noise）
│   ├── augmentation.py      # 加噪/变速/调音量 -> 微调清单（含 CLI）
│   └── manifest.py
└── models/
    ├── __init__.py
    ├── lora.py       
    └── adapters.py  
```

## 2. 环境要求

| 项目 | 推荐配置 |
|:-:|:-:|
| Python | 3.10 ~ 3.12 |
| GPU | NVIDIA GPU；微调 1.7B 建议显存 ≥ 24 GB |
| 依赖 | `qwen-tts`、`torch`、`transformers`、`peft`、`accelerate`、`librosa`、`soundfile`、`pyyaml` |

+ 依赖安装方法见 `src/tts/README.md`

## 3. 使用流程

+ 统一以**仓库根目录**为工作目录：

```shell
# 1) 构建中文指令（可选，若已有 Chinese-commands.json 可跳过）
export DASHSCOPE_API_KEY=sk-ws-...      # Windows: $env:DASHSCOPE_API_KEY="sk-ws-..."
python training/dataset/translation.py --config configs/translation/qwen_mt.yaml


# 2) 用 Qwen3-TTS 生成三类语音集（standard / dialect / noise）
python training/dataset/build_dataset.py --kind standard --config configs/tts/standard.yaml
python training/dataset/build_dataset.py --kind dialect  --config configs/tts/dialect.yaml
python training/dataset/build_dataset.py --kind noise    --config configs/tts/noise.yaml

### 可以使用以下命令对数据集（前 5 条音频）进行测试
python tests/asr_test.py --dataset data/wav_files/standard/mapping.json --limit 5

# 3) 数据增强，产出 train/clean/eval 三份清单
python -m training.dataset.augmentation --config configs/training/augmentation.yaml

# 4) 微调（LoRA 或结构适配器）
python -m training.train --config configs/training/finetune.yaml

# 5) 评估
python -m training.evaluate --config configs/training/finetune.yaml --checkpoint outputs/asr_finetune
```

### 3.1 数据集构建（dataset/build_dataset.py）

+ 统一入口：`python training/dataset/build_dataset.py --kind <standard|dialect|noise> --config <yaml>`，输出均为 16 kHz 单声道 WAV + `mapping.json`。

| `--kind` | 配置文件 | 输入 | 输出 |
|:---|:---|:---|:---|
| `standard` | `configs/tts/standard.yaml` | `data/commands/Chinese-commands.json` | `data/wav_files/standard/` |
| `dialect` | `configs/tts/dialect.yaml` | 同上 | `data/wav_files/dialect/` |
| `noise` | `configs/tts/noise.yaml` | 上两者的 `mapping.json` | `data/wav_files/noise/standard_noise/`、`data/wav_files/noise/dialect_noise/` |

+ 执行顺序：先 `standard`、`dialect`（合成干净语音），再 `noise`（对上述清单逐条加噪）。
+ `mapping.json` 中的 `audio_file` 统一记录**相对仓库根**的路径（如 `data/wav_files/standard/std_00001.wav`），由 `src/tts/utils.load_mapping` 在读取时自动解析为绝对路径。
+ 关键字段：
  + `standard`：`speakers`（普通话音色轮换）、`batch_size`、`target_sample_rate`；
  + `dialect`：`profiles`（`name` 必填；`speaker`/`instruct` 可省略，缺省时自动回落到 `Qwen3TTSService.DIALECT_SPEAKERS` 内置预设）、`per_profile_all`（`true` 逐方言生成全量、`false` 每条指令轮换一种方言）；
  + `noise`：`noise_types`（white/pink/brown）、`snr_db_list`、`real_noise_dir`（可选真实噪声目录）、`seed`（可复现）。
+ 其它参数：`--log-file`（默认取配置中的 `log_file`）、`--log-level`；`--config` 相对仓库根解析。

## 4. 数据增强（dataset/augmentation.py）

+ 由干净音频派生“多噪声 × 多 SNR × 多语速”样本，并保留干净回放集，输出 JSONL 清单：`{"audio", "text", "condition"}`。
+ 清单中的 `audio` 统一记录为**相对仓库根**的路径（与 `mapping.json` 口径一致，增广样本与干净回放样本不再混用绝对/相对路径），读取时由加载侧解析回绝对路径。
+ 命令行参数：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--config` | `configs/training/augmentation.yaml` | 增强配置 |
| `--input-manifest` | `null` | 覆盖配置中的输入清单路径 |

+ 核心接口：
  + `NoiseAugmenter(...).augment_one(audio_path, text, output_dir, sr)`：为单条音频生成增强变体
  + `build_augmented_manifests(records, output_dir, train_manifest, clean_manifest, eval_manifest=None, eval_ratio=0.1, augmenter=None, sr=16000)`：生成三份清单并返回计数

+ 配置字段（`configs/training/augmentation.yaml`）：

| 字段 | 默认值 | 说明 |
|:---|:---|:---|
| `sample_rate` | `16000` | 采样率 |
| `noise_types` | `["white","pink","brown"]` | 合成噪声类型 |
| `snr_db_list` | `[20.0,10.0,5.0]` | 信噪比档位（dB） |
| `speed_rates` | `[1.0,0.9,1.1]` | 语速扰动 |
| `volume_db_range` | `3.0` | 音量扰动范围（dB） |
| `real_noise_dir` | `""` | 真实噪声 WAV 目录（可选） |
| `input_manifest` | `data/wav_files/standard/mapping.json` | 输入（干净音频+文本） |
| `output_dir` | `data/optimization/augmented` | 增强音频输出目录 |
| `manifest_path` | `data/optimization/train_manifest.jsonl` | 训练清单 |
| `clean_manifest_path` | `data/optimization/clean_manifest.jsonl` | 干净回放清单 |
| `eval_manifest_path` | `data/optimization/eval_manifest.jsonl` | 评估清单 |
| `eval_ratio` | `0.1` | 留出评估的干净样本比例 |
| `seed` | `42` | 随机种子 |

## 5. 微调（train.py）

+ 入口：`python -m training.train --config configs/training/finetune.yaml`；参数：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--config` | `configs/training/finetune.yaml` | 微调配置 |
| `--log-file` | `logs/training/train.log` | 日志文件路径 |

+ 配置字段（`configs/training/finetune.yaml`）：

| 字段 | 默认值 | 说明 |
|:---|:---|:---|
| `model_id_or_path` | `models/Qwen3-ASR-1.7B` | 基础模型 id 或本地路径 |
| `device_map` | `cuda:0` | 运行设备 |
| `dtype` | `bfloat16` | 精度 |
| `attn_implementation` | `null` | 注意力实现（如 `flash_attention_2`） |
| `sample_rate` | `16000` | 音频采样率 |
| `prompt` | `Transcribe the audio into Chinese text.` | 系统提示 |
| `max_length` | `2048` | 最大序列长度 |
| `method` | `lora` | `lora` 或 `adapter` |
| `manifest` | `data/optimization/train_manifest.jsonl` | 训练清单 |
| `eval_manifest` | `data/optimization/eval_manifest.jsonl` | 评估清单 |
| `replay_manifest` | `data/optimization/clean_manifest.jsonl` | 干净回放清单 |
| `replay_ratio` | `0.3` | 回放比例（防遗忘） |
| `lora.r` / `lora.alpha` / `lora.dropout` | `16 / 32 / 0.05` | LoRA 秩、缩放与丢弃率 |
| `lora.target_modules` | `["q_proj","k_proj","v_proj","o_proj"]` | LoRA 注入的模块 |
| `adapter.target` | `audio` | 适配器注入位置：`audio`（编码器）或 `decoder`（LM） |
| `adapter.bottleneck` / `adapter.dropout` / `adapter.last_n` | `64 / 0.0 / 6` | 瓶颈维度、丢弃率、注入末 N 层 |
| `output_dir` | `outputs/asr_finetune` | 权重输出目录 |
| `batch_size` / `grad_accum` | `2 / 8` | 批大小与梯度累积 |
| `learning_rate` / `epochs` | `1e-4 / 3` | 学习率与训练轮数 |
| `warmup_ratio` / `logging_steps` / `save_steps` | `0.03 / 10 / 200` | 调度与日志频率 |
| `bf16` / `gradient_checkpointing` | `true / true` | 精度与显存优化 |

+ 两种微调方式：
  + **LoRA**（`method: lora`）：仅训练低秩增量，显存占用小、可插拔；产物为 PEFT 适配器目录，可用 `PeftModel.from_pretrained(base, checkpoint)` 加载。
  + **结构适配器**（`method: adapter`）：冻结主干，向变压器层注入 `BottleneckAdapter`（Houlsby 残差瓶颈，初始近恒等）或 `SoftPrefix`；当前完成“注入 + 冻结 + 保存”，训练循环留有明确接点。
+ **防遗忘**：`replay_manifest` + `replay_ratio` 将干净样本按比例混入训练集，避免微调后通用识别能力退化。

## 6. 评估（evaluate.py）

+ 入口：`python -m training.evaluate --config configs/training/finetune.yaml [--checkpoint <lora_dir>]`；参数：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--config` | `configs/training/finetune.yaml` | 微调配置 |
| `--checkpoint` | `null` | 已保存的 LoRA 适配器目录（为空则评估基础权重） |

+ 指标：语料级字符错误率 CER（**标点不敏感**，复用 `src/asr/text_metrics.py`，与 `tests/` 评测口径一致）

```
CER = (S + D + I) / N
```

+ 输出：`outputs/asr_finetune/eval_cer.json`

```json
{
  "eval_manifest": {"samples": 800, "cer": 0.0621}
}
```

## 7. 注意事项

+ 统一从**仓库根目录**运行，模块内相对导入（`..utils`）才会正确解析；`train.py` / `evaluate.py` / `augmentation.py` 已含 `sys.path`。
+ 配置中的相对路径按**仓库根**解析（`config.py: resolve_path`），与启动目录无关。

## 8. 参考资料

+ https://github.com/QwenLM/Qwen3-ASR
+ https://github.com/huggingface/peft