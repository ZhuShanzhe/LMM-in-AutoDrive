# Qwen3-ASR 语音识别模块

+ 基于 Qwen3-ASR-1.7B 的本地离线语音识别服务，支持 30 种语言与 22 种中文方言。模块提供统一接口，可对音频单条/批量转写、计时并输出 JSON 结果。
+ 内置可插拔的推理期优化与安全处理：噪声前端（降噪/归一）、方言文本归一、异常输出安全守卫，分别通过 `frontend`、`dialect_normalizer`、`guard` 参数接入；均可不启用，保持核心链路纯净。

## 1. 文件结构

```text
asr/
├── __init__.py
├── qwen3_asr_service.py     # 类 Qwen3ASRService 的定义
├── slots.py                 # 驾驶语义槽位（方向/动作/否定/数字）
├── text_metrics.py          # 文本指标（标点不敏感：编辑距离/CER/SER）
├── guard.py                 # 异常输入处理 ASROutputGuard
├── utils.py
├── README.md
├── optimization/            # 推理期优化（免训练）
│   ├── __init__.py
│   ├── optimizer.py
│   ├── frontend.py
│   ├── dialect.py
│   ├── lexicons.py
│   └── README.md
├── compression/             # 轻量化与量化
│   ├── __init__.py
│   ├── export_onnx.py       # ONNX 导出
│   └── quantize.py          # PTQ 量化 + 精度回归
└── deployment/              # J6P 部署适配
    ├── __init__.py
    ├── hb_convert.py        # 地平线工具链转换
    ├── runtime.py           # 统一推理适配器（pytorch/onnx/j6p）
    └── resource_probe.py    # 时延/内存/功耗测量
```

## 2. 环境要求

| 项目 | 推荐配置 |
|:-:|:-:|
| Python | 3.10 ~ 3.12 |
| GPU（推荐） | NVIDIA GPU + CUDA 11.8 或以上 |
| 显存 | 加载 1.7B 模型约 4 GB（BF16） |
| 依赖 | `qwen-asr`、`torch`、`transformers`、`librosa`、`soundfile`、`pyyaml` |

+ 安装核心依赖：

```shell
pip install qwen-asr==0.0.6 transformers==4.57.6
```

## 3. 模型下载

+ 加载时若无法在线下载权重，可先下载到本地（默认路径 `models/Qwen3-ASR-1.7B`）：

```shell
# ModelScope（国内推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir models/Qwen3-ASR-1.7B
```

```shell
# Hugging Face 镜像
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
```

## 4. 快速开始

+ 核心服务位于 `Qwen3ASRService` 类，构造函数参数如下：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `model_id_or_path` | `str` | `models/Qwen3-ASR-1.7B` | 模型 id 或本地目录路径 |
| `device` | `str` | `cuda:0` | 运行设备，如 `cuda:0`、`cpu` |
| `dtype` | `str` | `bfloat16` | 模型精度（`float32/float16/bfloat16`） |
| `attn_implementation` | `Optional[str]` | `None` | 注意力实现（如 `flash_attention_2`） |
| `max_inference_batch_size` | `int` | `32` | 推理批大小上限 |
| `max_new_tokens` | `int` | `256` | 最大生成 token 数 |
| `language` | `Optional[str]` | `None` | 语言；`None` 为自动检测 |
| `target_sample_rate` | `int` | `16000` | 目标采样率 |
| `frontend` | `Optional[Callable]` | `None` | 噪声优化接口 `(audio, sr) -> audio` |
| `dialect_normalizer` | `Optional[Callable]` | `None` | 方言归一接口 `text -> text` |
| `guard` | `Optional[ASROutputGuard]` | `None` | 异常输出安全守卫（空/超长/重复幻觉等） |
| `output_dir` | `str` | `outputs` | 默认输出目录 |
| `raise_on_error` | `bool` | `False` | 出错时是否抛出异常 |

### 4.1 基础转写

```python
from src.asr import Qwen3ASRService

svc = Qwen3ASRService(model_id_or_path="models/Qwen3-ASR-1.7B")
res = svc.transcribe_file("audio.wav", output_json="outputs/asr.json")
print(res["text"], res["processing_time_seconds"])
```

### 4.2 带优化（降噪 + 方言归一）

```python
import yaml
from src.asr import Qwen3ASRService
from src.asr.optimization import build_optimizer

opt = build_optimizer(yaml.safe_load(open("configs/asr/optimization.yaml", encoding="utf-8")))
frontend, dialect_normalizer = opt.as_hooks()

svc = Qwen3ASRService.from_yaml(
    "configs/asr/qwen3_asr.yaml",
    frontend=frontend,
    dialect_normalizer=dialect_normalizer,
)
res = svc.transcribe_file("noisy_sichuan.wav", save_enhanced="outputs/enhanced.wav")
```

### 4.3 带 `guard`（防止异常输出）

```python
from src.asr import Qwen3ASRService
from src.asr.guard import ASROutputGuard

svc = Qwen3ASRService(model_id_or_path="models/Qwen3-ASR-1.7B", guard=ASROutputGuard(min_chars=2, max_chars=120))
res = svc.transcribe_file("audio.wav")
print(res["text"], "| safe:", res["guard_safe"], "| reasons:", res["guard_reasons"])
```

+ 也可在 `from_yaml(...)` 时传入 `guard=`，或从配置构建：`ASROutputGuard.from_yaml("configs/asr/pipeline.yaml")`（读取顶层 `guard` 段）。

## 5. 对外接口说明

+ `transcribe_file(audio_path, output_json=None, language=None, use_frontend=None, use_dialect=None, save_enhanced=None) -> dict`：单文件转写
+ `transcribe(audio, output_json=None, **kwargs)`：`str` 返回 `dict`，`List[str]` 返回 `list[dict]`
+ `transcribe_batch(audio_paths, output_json=None, **kwargs) -> list[dict]`：批量转写
+ `transcribe_dir(input_dir, output_json=None, **kwargs) -> list[dict]`：目录内所有音频转写
+ `from_yaml(path, frontend=None, dialect_normalizer=None, guard=None)`：从 YAML 构建服务

## 6. 输出格式

+ 单条转写输出 JSON：

```json
{
  "audio_file": "audio.wav",
  "text": "前方路口请左转",
  "language": "Chinese",
  "success": true,
  "processing_time_seconds": 0.42,
  "frontend_applied": true,
  "dialect_normalized": true,
  "guard_safe": true,
  "guard_reasons": [],
  "slots": {"direction": ["left"], "action": [], "negation": false, "quantities": []}
}
```

+ 批量输出为 `{"count": N, "records": [...]}`。
+ 仅当传入 `guard` 时，结果才包含 `guard_safe` / `guard_reasons` / `slots` 字段；`guard_safe=False` 时下游应按“无有效指令”处理。

### 6.1 驾驶语义槽位（slots.py）

+ **唯一实现**位于本模块 `slots.py`；`tests/utils/metrics.py` 仅做转发，避免逻辑重复，并让 `src/` 可独立部署。
+ 槽位抽取前统一调用 `text_metrics.strip_punctuation` 剥离标点，因此 `20，公里` 之类的标点不会阻断数量槽位匹配。
+ `extract_slots(text) -> dict`：抽取 `direction`（左/右/直行/掉头/变道）、`action`（停车/制动/加速/跟随等）、`negation`（否定）、`quantities`（数字 + 归一单位）。
+ `slot_preservation(ref, hyp) -> dict`：对比参考与识别的槽位一致度，输出 `direction_match` / `negation_match` / `quantity_recall` / `critical_score`。
+ 用途：供下游指令解析与安全守卫复用，并可作优化前后的驾驶语义诊断。
+ 槽位指标为**可选诊断项，默认关闭**：仅在 `tests/` 脚本加 `--enable-slots` 时参与聚合与报告，默认不写入 `summary.json`。

### 6.2 安全守卫（guard.py）

+ `ASROutputGuard.check(text, language) -> {"safe", "reasons", "slots", "text"}`。
+ 判定的异常类型：`too_short` / `too_long` / `repetition_hallucination` / `low_diversity` / `unexpected_language` / `missing_critical_slot` / `unsupported_direction`。
+ 可通过 `ASROutputGuard.from_yaml(path)` 从配置构建（顶层 `guard` 段）。

## 7. 轻量化与部署（compression/、deployment/）

+ `compression/export_onnx.py`：导出 ONNX（含 opset 与体积报告）；`compression/quantize.py`：构建**与评测集不重叠**的校准集、PTQ INT8（dynamic/static）、量化前后 CER 回归（`budget=0.03`）。
+ `deployment/hb_convert.py`：生成 `hb_mapper` 配置并调用地平线工具链（无工具链时明确报错）；`deployment/runtime.py`：`ASRRuntime` 统一后端（`pytorch` / `onnx` / `j6p`）；`deployment/resource_probe.py`：`ResourceProbe` 采集时延分位、显存/内存、功耗。
+ 说明：**x86 仿真结果不等同于 J6P 板端性能**，功耗与利用率仅在板端测量有效。

+ 链路：
```text
PyTorch 模型
   │  export_onnx.py        ← （格式转换，不改精度）
   ▼
  ONNX（.onnx，可在 x86 上用 ONNX Runtime 跑仿真）
   │  quantize.py           ← （INT8 量化，减小体积/加速）
   ▼
量化 ONNX
   │  deployment/hb_convert.py  ← （地平线工具链）
   ▼
J6P 可执行的 .bin
```

+ 安装依赖库：
```bash
pip install onnx onnxruntime-gpu
```

+ 运行示例：
```bash
bash scripts/run_asr_optimization.sh
```

## 8. 配置文件

+ 服务配置：`configs/asr/qwen3_asr.yaml`；优化配置：`configs/asr/optimization.yaml`（详见 `optimization/README.md`）。
+ 轻量化配置：`configs/asr/compression.yaml`（导出/量化/校准/精度预算）；部署配置：`configs/asr/deployment.yaml`（后端、J6P 参数、资源预算）。

## 9. 注意事项

+ 模型权重在进程内只加载一次（模块级缓存）；相同路径重复构造实例不会二次占用显存。
+ 本模块只做语音转文本；模型级微调属于 `training/` 目录。
+ 评测与对照报告统一放在 `tests/` 目录（指标、加载、报告均在 `tests/utils/`）；`src/` 不包含评测代码。

## 10. 参考资料

+ https://github.com/QwenLM/Qwen3-ASR