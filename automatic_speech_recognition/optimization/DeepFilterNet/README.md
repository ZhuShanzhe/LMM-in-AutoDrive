# DeepFilterNet3 噪声抑制模块

+ 基于 DeepFilterNet3 的实时语音增强模块，集成于 `optimization` 工具包中，提供高效、可插拔的音频降噪功能，支持单文件、批量处理及 JSON 结果输出，便于集成到 ASR 流水线或语音处理系统中。

## 1. 概述

+ DeepFilterNet3 是一款轻量级全频段语音增强模型，专为实时降噪设计。其核心特点：

    + 参数高效：仅 210 万参数量，模型大小约 8.3 MB（float32）

    + 实时处理：支持 48 kHz 采样率，RT 因子低至 0.1~0.2

    + 多版本可选：DeepFilterNet、DeepFilterNet2、DeepFilterNet3

    + 易于集成：提供 Python API 和命令行两种使用方式

+ 本模块基于官方 `df` 库封装，提供统一的配置接口和服务层，可无缝接入现有项目。

+ 结构：

```text
DeepFilterNet/
├── __init__.py
├── config.py
├── denoiser.py
├── service.py
└── README.md
```

## 2. 环境要求

+ 安装：
  + Python 3.8+

  + PyTorch >= 2.0.0

  + CUDA（可选，GPU 加速）

+ 模型与依赖库安装：

```shell
pip install deepfilternet
pip install librosa soundfile numpy
```

+ 首次运行时会自动下载模型权重（约 8.3 MB），默认缓存于 `~/.cache/DeepFilterNet/`。

## 3. 快速开始

+ 单文件快速处理：
```python
from optimization import DenoiseService, DenoiserConfig

config = DenoiserConfig(
    model_name="DeepFilterNet3",
    output_sr=16000,
    output_dir="outputs"
)

service = DenoiseService(config)

output_path = service.denoise("noisy_speech.wav")
print(f"Denoised file：{output_path}")
```

+ 带 JSON 结果输出：
```python
result = service.denoise(
    audio="noisy_speech.wav",
    output_path="clean_speech.wav",
    output_json="result.json"
)
print(f"Time：{result['processing_time_seconds']:.3f}s")
```

+ 批量处理：
```python
files = ["noisy1.wav", "noisy2.wav", "noisy3.wav"]
outputs = service.denoise(
    audio=files,
    output_json="batch_result.json"
)
for out in outputs:
    print(out)
```
## 4. 接口文档

+ 配置说明 `DenoiserConfig`：

| 参数 | 类型 | 默认值 | 描述 |
|:----:|:----:|:----:|:----:|
| `model_name` | `str` | `"DeepFilterNet3"` | 模型版本，可选 `"DeepFilterNet"`, `"DeepFilterNet2"`, `"DeepFilterNet3"` |
| `device` | `Optional[str]` | `None` | 运算设备：`"cpu"` 或 `"cuda"`，留空则自动检测 |
| `output_sr` | `int` | `16000` | 输出音频采样率（Hz），内部处理使用 48 kHz |
| `output_dir` | `str` | `"outputs"` | 默认输出目录（服务层使用） |

+ 底层降噪引擎 `DeepFilterNetDenoiser`，直接与 `df` 库交互：
```python
denoise_audio(audio: np.ndarray, sr: int, output_sr: Optional[int] = None) -> Tuple[np.ndarray, int]
```

+ 提供计时、JSON 输出和对外统一接口 `DenoiseService`：

```python
denoise(audio: Union[str, List[str]], output_path: Optional[str] = None, output_json: Optional[str] = None, output_sr: Optional[int] = None) -> Union[str, List[str], Dict]
```

+ DeepFilterNet 官方提供了命令行工具，可直接调用：
```shell
python -m df.enhance -m DeepFilterNet3 noisy_audio.wav -o data/

deepFilter --model DeepFilterNet3 data/example_noisy.wav -o data/example_denoisy.wav
```
