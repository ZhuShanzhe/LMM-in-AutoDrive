# Qwen3-TTS 语音合成模块

+ 基于 Qwen3-TTS（12Hz 系列）的本地离线文本转语音服务，支持中英文等 10 种语言、内置方言音色与指令式口音控制。模块提供简洁统一的接口，可对文本进行单条/批量合成，并在合成后按需叠加噪声或方言配置，最终统一输出 16 kHz 单声道 WAV 文件。
+ 与数据集构建脚本配合，可快速生成普通语音、方言语音与噪声语音三类评测数据。

## 1. 文件结构

```text
tts/
├── __init__.py             
├── qwen3_tts_service.py        # 包含类 Qwen3TTSService 的实现
├── utils.py
└── README.md
```

## 2. 环境要求

| 项目 | 推荐配置 |
|:-:|:-:|
| Python | 3.12（建议使用独立 conda 环境） |
| GPU（推荐） | NVIDIA GPU + CUDA 11.8 或以上 |
| 显存 | 加载 1.7B 模型建议 ≥ 24 GB |
| 依赖 | `qwen-tts`、`torch`、`soundfile`、`librosa`、`numpy`、`pyyaml` |

+ 安装核心依赖：

```shell
pip install qwen-tts==0.1.1 transformers==4.57.3
```

+ 推荐安装 FlashAttention-2 以降低显存占用（需与模型 `dtype=float16/bfloat16` 配合）：

```shell
pip install -U flash-attn --no-build-isolation
```

## 3. 模型下载

+ 加载时若无法在线下载权重，可先手动下载到本地（以 `CustomVoice` 为例，国内推荐 ModelScope）：

```shell
# Hugging Face 镜像
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B
```

```shell
# ModelScope
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir models/Qwen3-TTS-12Hz-1.7B
```

## 4. 快速开始

+ 面向对象接口位于 `Qwen3TTSService` 类，构造函数参数如下：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `model_id_or_path` | `str` | - | 模型 id 或本地模型目录路径 |
| `device_map` | `str` | `cuda:0` | 运行设备，如 `cuda:0`、`cpu` |
| `dtype` | `str` | `bfloat16` | 模型精度（`float32/float16/bfloat16`） |
| `attn_implementation` | `str` | `""` | 注意力实现（如 `flash_attention_2`，留空用默认） |
| `gen_kwargs` | `Dict` | `{}` | 透传给 `generate` 的采样参数（如 `max_new_tokens`） |
| `output_dir` | `str` | `outputs` | 默认输出目录 |
| `target_sample_rate` | `int` | `16000` | 输出 WAV 采样率（统一 16 kHz） |

### 4.1 单条合成（可带方言与噪声）

```python
from src.tts import Qwen3TTSService

service = Qwen3TTSService(
    model_id_or_path="models/Qwen3-TTS-12Hz-1.7B",
    device_map="cuda:0",
    gen_kwargs={"max_new_tokens": 1024},
)

# 标准普通话音色
rec = service.generate("前方路口请左转", output_file="outputs/a.wav", speaker="Vivian")

# 内置方言音色（Eric=四川 / Dylan=北京）
rec = service.generate(
    "前面路口左拐一下",
    output_file="outputs/sichuan.wav",
    dialect="sichuan",
)

# 合成时直接叠加噪声
rec = service.generate(
    "请减速至每小时 40 公里",
    output_file="outputs/noisy.wav",
    speaker="Uncle_Fu",
    add_noise=True,
    noise_type="pink",
    snr_db=15.0,
)
```

### 4.2 批量合成

```python
records = service.generate_batch(
    texts=["前方请减速", "请靠边停车"],
    output_dir="outputs/batch",
    file_names=["cmd_00001.wav", "cmd_00002.wav"],
    speakers=["Vivian", "Serena"],
)
```

### 4.3 对已有音频叠加噪声

```python
rec = service.add_noise_to_file(
    audio_path="outputs/a.wav",
    output_path="outputs/a_white_10db.wav",
    noise_type="white",
    snr_db=10.0,
)
```

## 5. 方言配置说明

+ 内置 `DIALECT_SPEAKERS` 预设（仅使用 CustomVoice 自带音色 + 口音指令，不额外加载模型）：

| 名称 | 音色 | 说明 |
|:-:|:-:|:-:|
| `sichuan` | `Eric` | 四川口音男声（原生方言音色，推荐） |
| `beijing` | `Dylan` | 北京口音男声，儿化音自然（原生方言音色，推荐） |
| `dongbei` | `Vivian` | 东北口音（指令渲染，建议采样验证） |
| `shaanxi` | `Uncle_Fu` | 陕西方言口音（指令渲染，建议采样验证） |
| `shandong` | `Uncle_Fu` | 山东口音（指令渲染，建议采样验证） |
| `henan` | `Serena` | 河南口音（指令渲染，建议采样验证） |
| `shanghai` | `Vivian` | 上海腔普通话（指令渲染，建议采样验证） |
| `cantonese` | `Uncle_Fu` | 广东腔普通话（指令渲染，建议采样验证） |
| `hunan` | `Serena` | 湖南腔普通话（指令渲染，建议采样验证） |
| `tianjin` | `Dylan` | 天津口音（指令渲染，建议采样验证） |

+ 说明：CustomVoice 官方仅内置 `Eric`（四川）、`Dylan`（北京）两个中文方言音色；其余为**通过口音指令（instruct）在普通话音色上模拟**，效果并非原生方言，批量生成前建议用方言 ASR 评测确认。
+ `generate()` 的 `dialect` 参数支持三种写法：
  + 预设名（字符串）：`dialect="sichuan"`
  + 自定义字典：`dialect={"speaker": "Vivian", "instruct": "请用东北口音朗读"}`
  + 查询全部预设：`Qwen3TTSService.supported_dialects()`

## 6. 噪声参数说明

+ `noise_type`：`white`（白噪声）/ `pink`（粉噪声）/ `brown`（棕噪声）
+ `snr_db`：信噪比（dB），数值越小噪声越强
+ `real_noise_file`：自定义真实噪声 WAV 路径（非空时优先使用真实噪声）

## 7. 输出格式

+ 所有接口最终输出统一为 **16 kHz、单声道、float32 WAV**，兼容 ASR 输入规格。
+ 单条/批量接口返回 record 字典：

```json
{
  "text": "前方路口请左转",
  "audio_file": "outputs/a.wav",
  "sample_rate": 16000,
  "speaker": "Vivian",
  "instruct": null,
  "noise_type": null,
  "snr_db": null
}
```

## 8. 数据集构建

+ 三类语音集由 `training/dataset/build_dataset.py` 统一构建（均以仓库根为工作目录）：

```shell
python training/dataset/build_dataset.py --kind standard --config configs/tts/standard.yaml
python training/dataset/build_dataset.py --kind dialect  --config configs/tts/dialect.yaml
python training/dataset/build_dataset.py --kind noise    --config configs/tts/noise.yaml
```

| `--kind` | 说明 | 输出 |
|:---|:---|:---|
| `standard` | 按 `speakers` 轮换普通话音色逐条合成 | `data/wav_files/standard/` |
| `dialect` | 按 `profiles` 生成各方言语音（`per_profile_all` 控制全量或轮换） | `data/wav_files/dialect/` |
| `noise` | 对标准/方言语音集按 `noise_types × snr_db_list` 加噪 | `data/wav_files/noise/standard_noise/`、`data/wav_files/noise/dialect_noise/` |

+ 每个数据集均输出 16 kHz 单声道 WAV 与 `mapping.json`（字段：`index`、`text`、`audio_file`、`sample_rate`、`speaker`/`dialect`、`noise_type`/`snr_db`），可直接用于 ASR 评测与微调。
+ 其中 `audio_file` 记录**相对仓库根**的路径（如 `data/wav_files/standard/std_00001.wav`），便于跨机器与容器迁移；读取时由 `load_mapping` 自动解析回绝对路径。
+ 执行顺序：先 `standard`、`dialect`，再 `noise`（其输入为前两者的 `mapping.json`）。

## 9. 参考资料
+ https://github.com/QwenLM/Qwen3-TTS
