# 自动驾驶语音识别与指令处理（automatic_speech_recognition）

+ 车载语音指令处理链路：**语音识别 → 推理期优化 → 中英翻译 → 指令解析对接**，并提供轻量化、J6P 部署适配与统一评测。

## 1. 目录结构

```text
automatic_speech_recognition/
├── src/                      # 核心源码
│   ├── pipeline.py            # ASRPipeline：对外统一入口
│   ├── asr/                   # 语音识别（服务 + 优化 + 轻量化 + 部署）
│   ├── translator/            # 中英翻译（Qwen3-1.7B）
│   └── tts/                   # 语音合成（Qwen3-TTS）
├── training/                 # 微调：数据准备 / LoRA / 适配器 / 评估
├── tests/                    # 可运行测试 + 评测工具（utils/）
├── configs/                  # 全部配置（asr / training / translation / tts）
├── resources/dialect/        # 方言词典资源（JSON）
├── examples/                 # 最小可运行示例
├── recoder/                  # 麦克风录音（独立组件）
├── scripts/                  # 端到端任务链脚本
├── models/                   # 模型权重（不入 Git）
└── requirements.txt
```

## 2. 安装

```shell
pip install -r requirements.txt
# GPU 版 PyTorch（可选）
pip3 install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118
```

+ 注意：若环境中已有 `torch`，请确保版本与 `torchvision`、`torchaudio` 一致：
```shell
pip3 show torch torchvision torchaudio       # 确保一致
```

+ 注意：`onnxruntime` 版本需与 `torch` 版本一致。

## 3. 模型准备

| 模型 | 用途 | 默认路径 |
|:---|:---|:---|
| Qwen3-ASR-1.7B | 语音识别 | `models/Qwen3-ASR-1.7B` |
| Qwen3-1.7B | 中英翻译 | `models/Qwen3-1.7B` |
| Qwen3-TTS-12Hz-1.7B | 语音合成（造数据） | `models/Qwen3-TTS-12Hz-1.7B` |

## 4. 快速开始

```python
from src.pipeline import ASRPipeline

pipe = ASRPipeline(
    asr_model_path="models/Qwen3-ASR-1.7B",
    enable_optimization=True,          # 降噪 + 方言归一
    optimization_config="configs/asr/optimization.yaml",
    enable_translation=True,           # 中 -> 英
    output_language="english",         # chinese | english | both
    output_dir="outputs",
)
result = pipe.process("audio.wav", output_json="outputs/result.json")
print(result["text"], "|", result["translation"], "|", result["output_text"])
```

## 5. 完整流程

```shell
# 0) 构建数据集（可选，若已有 Chinese-commands.json 与语音集可跳过）
# Linux/MacOS: export DASHSCOPE_API_KEY=sk-ws-...      Windows: $env:DASHSCOPE_API_KEY="sk-ws-..."
python training/dataset/translation.py   --config configs/translation/qwen_mt.yaml
python training/dataset/build_dataset.py --kind standard --config configs/tts/standard.yaml
python training/dataset/build_dataset.py --kind dialect  --config configs/tts/dialect.yaml
python training/dataset/build_dataset.py --kind noise    --config configs/tts/noise.yaml

# 1) 基线评测（标准 / 噪声 / 方言）
python tests/asr_test.py     --dataset data/wav_files/standard/mapping.json       # --limit 20
python tests/noise_test.py   --dataset data/wav_files/noise/standard_noise/mapping.json # --limit 20
python tests/dialect_test.py --dataset data/wav_files/dialect/mapping.json # --dialect sichuan --limit 20

# 2) 轻量化（ONNX 导出 + INT8 量化）
python -m src.asr.compression.quantize --config configs/asr/compression.yaml

# 3) 微调（可选：LoRA / 结构适配器）
python -m training.dataset.augmentation --config configs/training/augmentation.yaml
python -m training.train    --config configs/training/finetune.yaml
python -m training.evaluate --config configs/training/finetune.yaml --checkpoint outputs/asr_finetune

# 4) 一键串起评测与轻量化
bash scripts/run_asr_optimization.sh
```

## 6. 模块文档

| 模块 | 文档 |
|:---|:---|
| 统一入口 ASRPipeline | [src/README.md](src/README.md) |
| 语音识别 | [src/asr/README.md](src/asr/README.md) |
| 推理期优化 | [src/asr/optimization/README.md](src/asr/optimization/README.md) |
| 中英翻译 | [src/translator/README.md](src/translator/README.md) |
| 语音合成 | [src/tts/README.md](src/tts/README.md) |
| 微调 | [training/README.md](training/README.md) |
| 测试与评测 | [tests/README.md](tests/README.md) |
| 示例 | [examples/README.md](examples/README.md) |

## 7. 说明

+ 权重、数据集与生成语料不入 Git；`data/` 下音频需自行生成或替换。
+ 日志由 `src/utils.py` 统一提供（`setup_logging` / `log_and_print`），各任务不再自带副本；每次运行会**清空并重写**自己的日志文件，默认落在 `logs/` 下（如 `logs/tests/asr_test.log`、`logs/compression.log`、`logs/tts/build_standard.log`），可用 `--log-file ""` 关闭落盘。
+ 推理期优化免训练；模型级微调在 `training/`；评测代码统一在 `tests/`。
+ x86 仿真结果不等同于 J6P 板端性能；功耗与利用率仅在板端测量有效。

## 8. 参考资料

+ https://github.com/QwenLM/Qwen3-ASR
+ https://github.com/QwenLM/Qwen3-TTS
