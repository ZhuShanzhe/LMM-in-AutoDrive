# 自动驾驶语音指令处理流水线

+ 一个面向自动驾驶场景的模块化语音处理工具包，提供从语音指令采集、识别、翻译到合成的全链路解决方案。项目涵盖 ASR（语音识别）、机器翻译、语音合成（TTS）、指令解析以及噪声/方言优化等核心模块，所有组件均支持离线运行与 GPU 加速，便于集成到车载系统中。

## 1. 项目概述
+ 本项目旨在构建面向自动驾驶的语音识别系统，涵盖以下功能：

  + 语音识别（ASR）：将驾驶员的语音指令（中文）转录为文本，提供 FunASR（轻量级）和 Qwen3‑ASR（高精度多语言）两种后端。

  + 机器翻译：将中文指令翻译为英文（可选），便于多语言场景或后续处理。

  + 语音合成（TTS）：将文本指令合成为自然语音，支持方言/情感控制，并可添加背景噪声以模拟真实环境。

  + 指令解析：从自然语言中提取结构化驾驶意图（如“减速”、“左转”），输出 JSON 格式。

  + 优化模块：针对方言/噪声语音的优化算法，提升 ASR 在复杂声学环境下的识别准确率。

  + 端到端流水线：将 ASR + 翻译 + 指令解析串联，实现单次调用完成从音频到结构化意图的全流程。

+ 所有组件均设计为离线优先，支持 GPU 加速，适合车载边缘部署。

## 2. 项目结构
```text
automatic_speech_recognition/
├── example/                    # 示例脚本（各模块的用法演示）
│   ├── asr_example.py
│   ├── asr2_example.py
│   ├── translation_example.py
│   ├── tts_example.py
│   └── pipeline_example.py
│
├── src/                        # 核心源代码
│   ├── asr/                    # FunASR 语音识别
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── funasr_model.py
│   │   ├── service.py
│   │   ├── utils.py
│   │   ├── example.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── asr2/                   # Qwen3-ASR 语音识别
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── asr_model.py
│   │   ├── service.py
│   │   ├── utils.py
│   │   ├── example.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── translation/            # 中英翻译
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── translator.py
│   │   ├── service.py
│   │   ├── example.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── tts/                    # 语音合成 (ChatTTS)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── service.py
│   │   ├── noise_utils.py
│   │   ├── example.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── pipeline.py             # ASR + 翻译（FunASR 后端）
│   ├── pipeline2.py            # ASR + 翻译（Qwen3-ASR 后端，推荐）
│   ├── config.py               # 顶层配置
│   ├── __init__.py
│   └── README.md
|
├── optimization/               # 噪声与方言优化模块
│   ├── __init__.py
│   ├── config.py
│   ├── optimizer.py
│   ├── audio_augmenter.py
│   ├── utils.py
│   └── README.md
|
├── tests/                      # 测试与评估脚本
│   ├── utils/                  # 测试工具
│   │   ├── data_loader.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── asr_test.py             # FunASR 测试
│   ├── qwen_test.py            # Qwen3-ASR 测试
│   ├── commands.py             # 翻译文本指令
│   ├── wav_commands.py         # 生成语音指令
│   ├── noise_commands.py       # 生成带噪语音指令
│   └── README.md
│
├── data/                       # 数据集与输出
│   ├── commands.json           # 原始英文指令集
│   ├── translated_commands.json# 中文翻译指令集
│   ├── wav_files/              # 无噪声语音指令（8349条）
│   │   ├── file_mapping.json
│   │   └── command_0001.wav ~ command_8349.wav
│   ├── wav_files_noise/        # 噪声语音指令（500条）
│   │   ├── wav_files_with_noise/
│   │   │   ├── file_mapping_noise.json
│   │   │   └── command_1_noise.wav ~ command_500_noise.wav
│   │   └── wav_files_without_noise/
│   │       └── file_mapping_without_noise.json
│   ├── test_results/           # FunASR 测试结果
│   │   ├── ASR_result.json
│   │   └── test_summary.json
│   ├── test_results_qwen3/     # Qwen3-ASR 测试结果
│   │   ├── ASR_result_qwen3.json
│   │   └── test_summary_qwen3.json
│   ├── test_results_noise/     # 噪声对比测试结果
│   │   ├── normal/             # 无噪声
│   │   │   └── test_summary_qwen3.json
│   │   └── noise/              # 含噪声
│   │       └── test_summary_qwen3.json
│   └── logging/                # 日志文件
│
├── requirements.txt
└── README.md 
```

## 3. 安装与配置
+ 环境要求：
  + Python 3.10 ~ 3.12（推荐 3.12）

  + NVIDIA GPU（推荐）或 CPU（内存 ≥ 16 GB）

  + CUDA 11.8 或以上及对应驱动程序

+ 安装总依赖：
```shell
pip install -r requirements.txt
```

+ 安装子模块依赖：
```shell
# FunASR
cd src/asr && pip install -r requirements.txt
# Qwen3-ASR
cd src/asr2 && pip install -r requirements.txt
# translation
cd src/translation && pip install -r requirements.txt
# TTS
cd src/tts && pip install -r requirements.txt
```

### 模型权重
+ 本项目模块依赖多个预训练模型完成语音识别、翻译与合成任务。
+ 模型权重下载：
+ 下载后的目录结构应为：

```text
└── models/                     # 预训练模型（需自行下载）
    ├── rvcmd_linux_amd64       # ChatTTS 模型
    ├── Qwen2.5-3B-Instruct     # Qwen2.5 翻译模型
    └── Qwen3-ASR-1.7B          # Qwen3-ASR 模型
```

## 4. 输入接口

+ 本项目的各模块提供了统一的输入接口，支持命令行调用与Python API两种方式，方便灵活集成。

### 语音识别测试

```shell
python tests/qwen_test.py \
    --dataset data/wav_files/file_mapping.json \   # 数据集 JSON 路径
    --output_dir data/test_results_qwen3 \         # 结果输出目录
    --asr_device cuda:0 \                          # 设备（cuda:0 / cpu）
    --load_type local \                            # 加载类型（local / custom）
    --model_path models/Qwen3-ASR-1.7B             # 本地模型路径
```

### 语音指令生成

```shell
python tests/wav_commands.py \
    --dataset data/translated_commands.json \       # 中文指令集
    --output_dir data/wav_files \                   # 输出 WAV 文件目录
    --load_type local \                             # TTS 模型加载方式
    --model_path models/rvcmd_linux_amd64           # ChatTTS 模型路径
    --noise_enabled False                           # 是否添加噪声
```

### `pipeline` 接口

+ 由于两种 `pipeline` 的接口形式基本一致，这里仅展示基于后端 `Qwen3-ASR-1.7B` 的接口说明。
+ 参数设置可以通过两种方法实现：
  + 直接传参（最常用）：
  ```python
  pipeline = ASR2(
    asr_load_type="local",                     # 加载方式：local / custom
    asr_model_path="./models/Qwen3-ASR-1.7B",  # 本地模型路径（若 load_type=local）
    asr_device="cuda:0",                       # 设备
    asr_language="Chinese",                    # 识别语言（必须为完整名称）
    trans_load_type="custom",                  # 翻译模型加载方式
    trans_model_name="Qwen/Qwen2.5-3B-Instruct", # 翻译模型名称
    output_dir="outputs",                      # 结果输出目录
    raise_on_error=False,                      # 是否抛出异常
  )
  ```
  + 使用配置类（适合参数较多或需要复用）：
  ```python
  config = Qwen3PipelineConfig(
    load_type="local",
    model_path="./models/Qwen3-ASR-1.7B",
    language="Chinese",
    trans_load_type="custom",
    trans_model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="outputs",
  )
  pipeline = ASR2(config=config)
  ```
  
+ 核心方法 `process()`：
```python
def process(
    self,
    audio_path: str,                           # 输入音频路径（WAV，16kHz单声道）
    output_json: Optional[str] = None,         # 可选，保存结果JSON路径
    translate: bool = True,                    # 是否翻译为英文
    language: Optional[str] = None,            # 覆盖ASR语言（如 "Chinese"）
    enable_enhancement: Optional[bool] = None, # 覆盖降噪开关
    enable_dialect_mapping: Optional[bool] = None, # 覆盖方言映射
    **kwargs                                   # 其他参数透传给ASR服务
) -> Dict[str, Any]:
    ...
```

### 数据格式
+ 音频输入：本项目的 ASR 模块接受 16kHz 采样率、单声道 WAV 格式 的音频文件。其他格式需先转换为 WAV（可使用 ffmpeg 或 librosa 预处理）。

+ 数据集 JSON：测试脚本使用的 JSON 文件需包含以下字段：
```json
{
  "index": 1,
  "original": "turn left at the intersection",
  "translation": "在交叉路口左转",
  "audio_file": "data/wav_files/command_0001.wav"
}
```

## 5. 输出接口
+ 所有核心模块均支持将结果保存为 JSON 文件，格式统一如下：

```json
{
  "audio_file": "audio.wav",
  "text": "识别出的中文文本",
  "processing_time_seconds": 1.234
}
```

+ 流水线输出包含额外的翻译和耗时字段：

```json
{
  "audio_file": "audio.wav",
  "chinese_text": "请减速至40公里每小时",
  "english_translation": "Please decelerate to 40 km/h",
  "asr_processing_time_seconds": 1.234,
  "translation_time_seconds": 0.567,
  "total_time_seconds": 1.801
}
```

## 6. 实验

### 6.1 已完成的实验
+ 标准语音指令数据集构建与基准测试

  + 实验目标：构建面向自动驾驶场景的标准中文语音指令数据集，并建立 ASR 性能基准。

  + 数据集来源：基于 Talk2Car 数据集。该数据集包含 11,959 条自然语言指令，对应 9,217 张城市道路场景图像，其中训练集包含 8,349 条指令。原始指令为英文，描述了自动驾驶车辆应执行的操作（如“turn left to pick up the pedestrian at the corner”）。

  + 两种模型在字符级识别上均表现优异（>95%），说明关键字词识别能力较强。

+ 噪声鲁棒性测试

  + 实验目标：评估 ASR 模型在噪声环境下的性能下降程度。
  + 白噪声导致字符准确率下降约 7.5% ，句子准确率下降近 10%。

+ 模型对比实验
  + 实验目标：对比不同 ASR 模型在自动驾驶语音指令识别任务上的性能差异。
  + 对比模型： 
    + `FunASR`：paraformer-zh 模型，轻量级中文 ASR 
    + `Qwen3-ASR-1.7B`：1.7B 参数多语言模型，支持 30 种语言和 22 种中文方言

  + Qwen3-ASR-1.7B 在所有指标上均优于 FunASR，尤其在词级和句子级指标上提升更明显

### 6.2 待完成的实验

+ 方言与多语言扩展实验：
  + 实验目标：评估 ASR 模型在多种中文方言及多语言场景下的识别性能，构建方言语音指令数据集。
+ 优化策略实验：
  + 实验目标：验证前端语音增强（降噪）和后端方言映射对 ASR 性能的提升效果。
+ 真实场景部署测试：
  + 实验目标：在实际车载环境下验证 ASR 系统的端到端性能。
+ 多模态融合实验：
  + 实验目标：探索将视觉信息（如摄像头画面）与语音指令融合，提升复杂场景下的指令理解准确率。

## 7. 许可边界

| 模型 | 许可证 | 来源 | 说明 |
|:----:|:----:|:----:|:----:|
| Qwen3-ASR-1.7B | Apache-2.0 | Qwen 团队 / Hugging Face | 高性能多语言语音识别模型 |
| ChatTTS 模型（rvcmd_linux_amd64） | Apache-2.0 | FunAudioLLM / ModelScope | 文本转语音模型，含方言/情感控制 |
| Paraformer-zh（FunASR） | MIT | ModelScope | 中文语音识别模型 |
| FSMN-VAD | MIT | ModelScope | 语音活动检测模型 |
| CT-PUNC | MIT | ModelScope | 标点恢复模型 |
| CAM++（说话人识别） | MIT | ModelScope | 说话人分离模型 |
| Qwen2.5-3B-Instruct | Apache-2.0 | Qwen 团队 | 中英翻译模型 |
