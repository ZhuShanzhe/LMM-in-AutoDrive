# Qwen3-ASR-1.7B 语音识别模块

+ 基于 Qwen3-ASR-1.7B 的本地离线语音识别服务，支持 30 种语言 和 22 种中文方言，模块提供简洁统一的接口，支持单条/批量转录、JSON 输出、性能计时，并集成了噪声抑制和方言词汇映射两种优化策略，可通过配置灵活启用。

## 1. 文件结构

```text
asr2/
├── __init__.py           
├── config.py            
├── asr_model.py             
├── service.py          
├── utils.py            
├── example.py
├── requirements.txt       
└── README.md        
```

## 2. 安装与环境配置
+ 环境要求：

| 项目  |                      推荐配置                 |
|:-:|:-----------------------------------------:|
| Python  |                3.10 ~ 3.12                |
| GPU (推荐)  |        NVIDIA GPU + CUDA 11.8 或以上         |
| CPU 运行  |              支持（内存 ≥ 16 GB）               |
|    硬盘空间     |            模型文件约 4.7 GB（FP16）             |

## 3. 模型下载

+ 根据官方文档，在 `qwen-asr` 包或 `vLLM` 中加载模型时，会根据模型名称自动下载模型权重。但若您的运行环境不允许在执行过程中下载权重，可使用以下命令手动将模型权重下载至本地目录：

```shell
# Download through ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
# Download through Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```

## 4. Quick Start

### 4.1 环境配置

+ 使用 `Qwen3-ASR` 最简单的方法是从 `PyPI` 安装 `qwen-asr` Python 包，这将自动安装所需的运行时依赖项，并允许您加载任意已发布的 `Qwen3-ASR` 模型。
  + 如果希望进一步简化环境配置，也可以使用官方 Docker 镜像。

+ `qwen-asr` 包提供两种后端：`transformers` 后端和 `vLLM` 后端。可以按如下方式创建 Python 3.12 环境：

```shell
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

+ 运行以下命令以最小化安装并启用 `transformers` 后端支持：

```shell
pip install -U qwen-asr
```

+ 若要启用 `vLLM` 后端以获得更快的推理速度和流式支持，请运行：

```shell
pip install -U qwen-asr[vllm]
```

### 4.2 使用方法

+ 快速推理：
  + `qwen-asr` 包提供了两个后端——`transformers` 后端 和 `vLLM` 后端
  + 可以将音频输入作为本地路径、URL、base64 数据或 `(np.ndarray, sr)` 元组传入，并执行批量推理
  + 若要快速尝试 Qwen3-ASR，可以使用以下代码通过 `transformers` 后端调用 `Qwen3ASRModel.from_pretrained(...)`

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None, # set "English" to force the language
)

print(results[0].language)
print(results[0].text)
```

## 5. 参考资料
+ Qwen3-ASR 官方文档：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B
	
	
	
	