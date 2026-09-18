# ASRPipeline 对外统一入口

+ `src/pipeline.py` 中的 `ASRPipeline` 是 `automatic_speech_recognition` 对外推荐的单一入口，将语音识别、（可选）推理期优化与（可选）中英翻译串联为一次调用。
+ 可设定：ASR 模型路径、是否启用优化、输出中文还是英文、输出文件地址等；翻译器仅在需要英文时才加载。
+ 子模块（`asr/`、`translator/`、`tts/`）的细节见各自目录下的 `README.md`，本文件只介绍 `ASRPipeline`。

## 1. 目录结构

```text
src/
├── __init__.py                 # 惰性导出（按需加载子模块）
├── utils.py                    # 全项目共享工具：日志（setup_logging / log_and_print）
├── pipeline.py                 # ASRPipeline：对外统一入口（ASR + 可选优化 + 可选翻译）
├── asr/  
│   ├── __init__.py
│   ├── qwen3_asr_service.py    # Qwen3ASRService
│   ├── slots.py                # 驾驶语义槽位
│   ├── guard.py                # 异常输出安全守卫
│   ├── utils.py              
│   ├── optimization/           # 推理期优化（免训练）
│   │   ├── __init__.py
│   │   ├── optimizer.py      
│   │   ├── frontend.py        
│   │   ├── dialect.py        
│   │   ├── lexicons.py       
│   │   └── README.md
│   ├── compression/            # ONNX 导出 + PTQ 量化
│   ├── deployment/             # J6P 转换 / 统一运行时 / 资源测量
│   └── README.md
├── translator/               
├── tts/    
└── README.md
```

## 2. 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `asr_model_path` | `str` | `models/Qwen3-ASR-1.7B` | ASR 模型路径 |
| `asr_device` | `str` | `cuda:0` | ASR 运行设备 |
| `asr_dtype` | `str` | `bfloat16` | ASR 精度 |
| `asr_attn_implementation` | `Optional[str]` | `None` | 注意力实现（如 `flash_attention_2`） |
| `language` | `Optional[str]` | `None` | ASR 语言；`None` 自动检测 |
| `enable_optimization` | `bool` | `False` | 是否启用降噪 + 方言归一 |
| `optimization_config` | `Union[str, dict]` | `None` | 优化配置（yaml 路径或 dict） |
| `enable_translation` | `bool` | `False` | 是否中译英 |
| `translator_model_path` | `str` | `models/Qwen3-1.7B` | 翻译模型路径 |
| `translator_device` | `str` | `cuda:0` | 翻译设备 |
| `translator_dtype` | `str` | `bfloat16` | 翻译精度 |
| `output_language` | `str` | `chinese` | `chinese` / `english` / `both`（别名 `zh/cn/en/eng`） |
| `output_dir` | `str` | `outputs` | 默认输出目录 |
| `enable_guard` | `bool` | `False` | 是否启用异常输出安全守卫 |
| `raise_on_error` | `bool` | `False` | 出错时是否抛出异常 |

## 3. 接口介绍

| 方法 | 说明 |
|:---|:---|
| `from_yaml(path)` | 从 YAML 构建 pipeline |
| `process(audio, output_json=None, output_language=None, use_frontend=None, use_dialect=None, save_enhanced=None, translate=None)` | 处理单个音频 |
| `process_batch(audio_paths, output_json=None, output_language=None, **kwargs)` | 批量处理 |
| `process_dir(input_dir, output_json=None, output_language=None, **kwargs)` | 处理目录内音频 |

+ 单次调用可覆盖开关：`output_language`、`use_frontend`、`use_dialect`、`translate`、`save_enhanced`、`output_json`（相对路径按仓库根解析）。

## 4. 返回字段

```json
{
  "audio_file": "audio.wav",
  "text": "前方路口请左转",
  "translation": "Turn left at the intersection ahead.",
  "output_text": "Turn left at the intersection ahead.",
  "output_language": "english",
  "language": "Chinese",
  "success": true,
  "frontend_applied": true,
  "dialect_normalized": true,
  "translation_applied": true,
  "processing_time_seconds": 0.42,
  "guard_safe": true,
  "guard_reasons": [],
  "slots": {"direction": ["left"], "action": [], "negation": false, "quantities": []}
}
```

+ `output_text` 由 `output_language` 决定：`chinese` 为中文原文；`english` 为英文译文（无译文时回退中文）；`both` 为 `中文\t英文`。
+ 启用 `enable_guard` 时，才会出现 `guard_safe` / `guard_reasons` / `slots` 字段。
+ `process_batch` / `process_dir` 的 JSON 输出为 `{"count": N, "records": [...]}`。

## 5. 快速开始

```python
from src.pipeline import ASRPipeline

pipe = ASRPipeline(
    asr_model_path="models/Qwen3-ASR-1.7B",
    enable_optimization=True,
    optimization_config="configs/asr/optimization.yaml",
    enable_translation=True,
    output_language="english",
    output_dir="outputs",
)
result = pipe.process("audio.wav", output_json="outputs/result.json")
print(result["text"])          # 中文原文
print(result["translation"])   # 英文译文
print(result["output_text"])   # 按 output_language 选择的结果
```

## 6. 日志（utils.py）

+ 全项目的日志由 `src/utils.py` 统一提供，各任务（数据构建、翻译、微调、评测、测试、轻量化）不再各自实现。
+ 接口：
  + `setup_logging(log_file=None, level=INFO)`：为本次运行准备日志；文件以 **`mode="w"`** 打开，**每次运行先清空**，同时会移除旧 handler，避免同进程内重复输出。传空字符串则只输出到控制台。
  + `log_and_print(message, level=INFO)`：同一行同时写入控制台与日志文件，并逐条 flush。
  + `resolve_log_path(log_file)`：相对路径按仓库根解析。
+ 日志挂在 root logger 上，因此 `src/` 与 `training/` 内部的 `logger.info/warning` 也会一并写入当次日志文件，便于事后排查。

```python
from src.utils import log_and_print, setup_logging

setup_logging("logs/run.log")
log_and_print("task started")
```

## 7. 参考资料

+ https://github.com/QwenLM/Qwen3-ASR

