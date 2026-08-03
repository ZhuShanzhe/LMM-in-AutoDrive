# Optimization 优化模块

+ `optimization` 模块提供了一套完整的音频增强与降噪工具，适用于语音数据处理、ASR 鲁棒性测试等场景。该模块包含两大核心功能：

  + 噪声增强：为干净的音频添加多种类型的噪声（白噪声、粉红噪声、车辆噪声、自定义噪声），支持 SNR 控制，可用于构建带噪语音数据集。

  + 语音增强：基于 DeepFilterNet3 的实时降噪引擎，可有效抑制背景噪声，提升语音质量，适用于 ASR 前端预处理。

+ 所有组件均支持单文件、批量处理以及 JSON 映射输出，方便集成到数据流水线中。

## 1. 目录结构

```text
optimization/
├── __init__.py
├── config.py
├── audio_processor.py
├── noise_generator.py
├── utils.py
├── DeepFilterNet/
│   ├── __init__.py
│   ├── config.py
│   ├── denoiser.py
│   ├── service.py
│   └── README.md
├── build_noisy_subset.py
├── example.py
├── requirements.txt
└── README.md
```

## 2. 安装与依赖

+ 环境要求：
  + Python 3.8+

  + PyTorch >= 2.0.0（DeepFilterNet 需要）

  + CUDA（可选，用于 GPU 加速降噪）

+ 安装命令：

```shell
pip install -r requirements.txt

pip install librosa soundfile numpy

pip install deepfilternet
```

+ 首次使用降噪功能时会自动下载 DeepFilterNet3 模型（约 8.3 MB），默认缓存于 `~/.cache/DeepFilterNet/`。

## 3. 接口与脚本使用

### 3.1 噪声增强（加噪）

+ 核心类 `AudioAugmenter`
+ 主要方法：
  + `process_file(input_path, output_path=None, snr_db=None, noise_type=None, noise_file=None) -> str`
  处理单个音频文件，返回输出路径。

  + `process_dataset(dataset_json, audio_key="audio_file", output_json_clean=None, output_json_noisy=None, ...) -> Tuple[List[Dict], List[Dict]]`
处理 JSON 格式的数据集，为每个音频生成带噪版本，并返回两个映射列表（干净和带噪），同时可保存 JSON 文件。

  + `process_directory(input_dir, output_dir=None, ...) -> List[str]`
处理目录下所有 WAV 文件，返回输出路径列表。

+ 构建噪声数据集：

```shell
python build_noisy_subset.py --dataset data/wav_files/file_mapping.json \
                             --output_dir data/wav_files_noise \
                             --num_samples 500 \
                             --seed 42
```

+ 该脚本会生成 `noisy_mapping.json`，其中包含原始信息、噪声类型和 SNR 值，便于后续分析。

### 3.2 语音增强（降噪）
+ 基于 `DeepFilterNet3` 的实时语音增强模块，提供高效、可插拔的降噪功能。
+ 使用示例：
```python
from optimization import DenoiseService, DenoiserConfig

config = DenoiserConfig(model_name="DeepFilterNet3", output_sr=16000)
service = DenoiseService(config)
output = service.denoise("noisy.wav", output_path="clean.wav", output_json="result.json")
print(f"Time: {output['processing_time_seconds']:.3f}s")

files = ["noisy1.wav", "noisy2.wav"]
outputs = service.denoise(files, output_json="batch.json")
```

+ 或者使用命令行调用（`DeepFilterNet` 官方工具）：
```shell
python -m df.enhance -m DeepFilterNet3 noisy_audio.wav -o output_dir/
deepFilter --model DeepFilterNet3 noisy_audio.wav -o output_dir/
```
