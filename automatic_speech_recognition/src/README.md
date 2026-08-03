# 语音处理流水线
+ 一个面向自动驾驶场景的模块化语音处理工具包。本仓库提供了端到端的流水线，涵盖：

  + ASR（自动语音识别） – 将中文语音转录为文本

  + 翻译 – 将中文文本翻译为英文

  + TTS（语音合成） – 生成带方言/情感控制的自然语音，并可添加背景噪声

  + 指令解析 – 从自然语言指令中提取结构化驾驶意图

+ 流水线 – 将 “ASR + 翻译 + 指令解析” 组合为单次调用

+ 所有组件均设计为离线优先、支持 GPU 加速，且易于集成到更大的系统中

## 1. 项目结构

```text
src/
├── asr/                      # 自动语音识别
│   ├── __init__.py
│   ├── config.py 
│   ├── funasr_model.py
|   ├── service.py
|   ├── utils.py
|   ├── example.py
|   ├── requirements.txt
|   └── README.md
│
├── asr2/                    # Qwen3-ASR
|   ├── __init__.py           
|   ├── config.py            
|   ├── asr_model.py             
|   ├── service.py          
|   ├── utils.py            
|   ├── example.py
|   ├── requirements.txt       
|   └── README.md        
|
├── translation/              # 机器翻译（中 ↔ 英）
|   ├── __init__.py
|   ├── config.py          
|   ├── translator.py      
|   ├── service.py         
|   ├── example.py         
|   ├── requirements.txt
|   └── README.md
│
├── tts/                      # 语音合成
|   ├── __init__.py
|   ├── config.py          
|   ├── model.py      
|   ├── service.py
|   ├── noise_utils.py        
|   ├── example.py        
|   ├── requirements.txt   
|   └── README.md
|
├── __init__.py                 
├── config.py  
├── pipeline.py
├── pipeline2.py
└── README.md
```

## 2. 模块概述

### FunASR 语音识别
+ 基于 `FunASR` 的中文语音识别服务，支持单句和说话人分离模式，主要特性：

  + 支持 `paraformer-zh` 等模型

  + 单条/批量转录，JSON 输出

+ 快速开始：

```python
from asr import FunASRService, FunASRConfig

config = FunASRConfig(mode="single", device="cuda:0")
service = FunASRService(config)
result = service.transcribe("audio.wav", output_json="result.json")
print(result["text"])
```

### Qwen3-ASR 语音识别
+ 基于 `Qwen3-ASR-1.7B` 的多语言语音识别服务，支持 30 种语言和 22 种中文方言，识别精度高，支持本地模型加载。

+ 快速开始：

```python
from asr2 import Qwen3ASRService, Qwen3ASRConfig

config = Qwen3ASRConfig(
    load_type="local",
    model_path="./models/Qwen3-ASR-1.7B",
    language="Chinese"
)
service = Qwen3ASRService(config)
result = service.transcribe("audio.wav", output_json="result.json")
print(result["text"])
```

### Qwen2.5-3B-Instruct 中英文翻译
+ 基于 `Qwen2.5-3B-Instruct` 的翻译服务，支持中英双向翻译，支持单条/批量/文件输入，输出 JSON 含时间和结果。

+ 快速开始：

```python
from translation import Translation

service = Translation(src_lang="zho_Hans", tgt_lang="eng_Latn")
result = service.translate("请减速至40km/h")
print(result)  # "Please decelerate to 40 km/h"
```

### ChatTTS 语音合成
+ 基于 `ChatTTS` 的语音合成服务，支持单条/批量合成，可添加白噪声/粉噪声/自定义环境噪声，输出为 `.wav` 文件。

+ 快速开始：
```python
from tts import ChatTTSService

service = ChatTTSService(model_path="./models/rvcmd_linux_amd64")
filepath = service.synthesize("今天天气真好")
print(filepath)  # outputs/command_0001.wav
```

### `Pipeline` 语音识别模块整合
+ `pipeline.py`：ASR + Translation 组合，将 FunASR 识别和翻译串联，输入音频，输出中文转录和英文翻译
```python
from pipeline import ASR

pipeline = ASR(asr_mode="single", trans_num_beams=4)
result = pipeline.process("audio.wav", output_json="result.json")
print(result["chinese_text"], result["english_translation"])
```

+ `pipeline2.py`（推荐）：基于高性能 Qwen3-ASR 的流水线，同样支持可选翻译
```python
from pipeline2 import ASR2

pipeline = ASR2(
    asr_load_type="local",
    asr_model_path="./models/Qwen3-ASR-1.7B",
    trans_load_type="custom",
    trans_model_name="Qwen/Qwen2.5-3B-Instruct",
)
result = pipeline.process("audio.wav", translate=False)

result = pipeline.process("audio.wav", translate=True)
```

## 3. 数据集构建

### 标准语音指令数据集
+ 使用 Talk2Car 文本指令数据集 + TTS 构建语音指令数据集：

```
  Talk2Car
      │
      ▼
 command.json
      │
      ▼
提取驾驶指令文本
      │
      ▼
 TTS(多说话人)
      │
      ▼
 command.wav
      │
      ▼
（添加噪声/方言）
      │
      ▼
 语音指令数据集
      │
      ▼
 ASR 语音识别
      │
      ▼
  测试准确率
```

+ 结果：生成 `command_0001.wav` ~ `command_8349.wav` 共 `8349` 条语音指令，每条语音指令都包含一项或多项基础操控（部分存在无关指令），用于测试基础语音指令识别的准确率。

## 4. 测试与评估
+ 测试模块位于 `tests/` 目录，提供批量测试、指标计算和结果汇总功能。

## 5. 日志
+ 所有模块均使用 `Python` 的 `logging`，每个模块有独立的 `logger`。您可以这样配置：

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```
