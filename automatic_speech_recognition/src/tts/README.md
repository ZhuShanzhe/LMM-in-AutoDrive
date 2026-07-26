# ChatTTS 语音合成模块
+ 基于 ChatTTS 的本地离线文本转语音（TTS）服务，支持中英文语音合成、批量处理、自定义噪声注入等功能。
+ 模块提供函数式接口和面向对象接口两种使用方式。

## 1. 目录结构

```text
tts/
├── __init__.py
├── config.py          
├── model.py               
├── service.py
├── noise_utils.py        
├── example.py
├── requirements.txt
└── README.md         
```

## 2. 环境要求
+ Python 版本：Python 版本应在 3.9 以上，测试时使用 3.11
+ NVIDIA GPU（推荐）或 CPU
+ 操作系统：Linux / Windows / macOS

## 3. 模型下载
+ 克隆官方仓库：

```shell
https://github.com/2noise/ChatTTS.git
```

+ 安装依赖库：
```shell
conda create -n ChatTTS python=3.11
conda activate ChatTTS
pip install -r requirements.txt
pip install safetensors vllm==0.2.7 torchaudio
```

+ 简单本地部署：
```shell
pip install ChatTTS
pip install git+https://github.com/2noise/ChatTTS
pip install -e .
```

+ 模型本地下载与加载：
  1. 下载压缩包：https://github.com/fumiama/RVC-Models-Downloader/releases/download/v0.2.11/rvcmd_linux_amd64.tar.gz
  2. 解压并运行：
  ```shell
  chmod +x rvcmd
  ./rvcmd -notui -w 0 assets/chtts
  ```
  3. 镜像下载：
  ```shell
  wget -P asset/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/Decoder.safetensors
  wget -P asset/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/DVAE.safetensors
  wget -P asset/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/Embed.safetensors
  wget -P asset/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/Vocos.safetensors
  wget -P asset/gpt/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/gpt/config.json
  wget -P asset/gpt/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/gpt/model.safetensors
  wget -P asset/tokenizer/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/tokenizer/special_tokens_map.json
  wget -P asset/tokenizer/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/tokenizer/tokenizer_config.json
  wget -P asset/tokenizer/ https://hf-mirror.com/2Noise/ChatTTS/resolve/main/asset/tokenizer/tokenizer.json
  ```

## 4. Quick Start

+ 面向对象接口 `ChatTTSService` 配置：
  + `model_path`: 模型目录路径

  + `compile_model`: 是否启用 `Torch` 编译优化（默认 `True`）

  + `batch_size`: 默认批量大小（默认 `8`）

  + `output_dir`: 输出目录（默认 `data/wav_files`）

  + `noise_enabled`: 是否默认启用噪声（默认 `False`）

  + `noise_type`: 噪声类型（`"white"/"pink"/"brown"/"from_file"`）

  + `noise_level`: 噪声幅度（默认 `0.005`）

  + `noise_file`: 自定义噪声文件路径（当 `noise_type="from_file"` 时使用）

### 噪声启用

+ 启用噪声（模拟真实环境）

```python
service = ChatTTSService(
    model_path="./models/rvcmd_linux_amd64",
    noise_enabled=True,
    noise_type="white",
    noise_level=0.01
)
file = service.synthesize("前方有行人，请减速慢行")
```

+ 在运行时动态添加噪声

```python
file = service.synthesize(
    "注意安全",
    add_noise=True,
    noise_type="brown",
    noise_level=0.005
)
```

+ 加载自定义环境噪声文件

```python
file = service.synthesize(
    "车辆已启动",
    add_noise=True,
    noise_type="from_file",
    noise_level=0.02,
    noise_file="./traffic_noise.wav"
)
```


