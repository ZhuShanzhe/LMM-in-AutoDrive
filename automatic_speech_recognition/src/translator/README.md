# Qwen3-1.7B 中英翻译模块

+ 基于 Qwen3-1.7B 的本地离线文本翻译服务，面向自动驾驶指令场景，主要完成中文到英文的翻译（也支持英译中）。模块提供简洁统一的接口，可对文本进行单条/批量翻译、耗时统计，并将结果保存为 JSON 文件。
+ 模型加载与业务逻辑合并在 `Qwen3TranslatorService` 类中，辅助函数位于 `utils.py`；支持直接构造或通过 YAML 配置加载，便于集成到 `pipeline`。

## 1. 文件结构

```text
translator/ 
├── init.py
├── qwen3_translator_service.py         # 包含类 Qwen3TranslatorService 的实现
├── utils.py
└── README.md
```

## 2. 环境要求

| 项目 | 推荐配置 |
|:-:|:-:|
| Python | 3.10 ~ 3.12 |
| GPU（推荐） | NVIDIA GPU + CUDA 11.8 或以上 |
| 显存 | 加载 1.7B 模型约 3.5 GB（BF16） |
| 依赖 | `torch`、`transformers`、`pyyaml` |

+ 安装核心依赖：

```shell
pip install torch transformers pyyaml
```

## 3. 模型下载

+ 加载时若无法在线下载权重，可先手动下载到本地（默认路径 `models/Qwen3-1.7B`，国内推荐 `ModelScope`）：

```shell
# Hugging Face 镜像
pip install -U "huggingface_hub[cli]" 
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```
```shell
# ModelScope
pip install -U modelscope 
export modelscope download --model Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B
```
## 4. 快速开始

+ 面向对象接口位于 `Qwen3TranslatorService` 类，构造函数参数如下：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `model_id_or_path` | `str` | `models/Qwen3-1.7B` | 模型 id 或本地模型目录路径 |
| `device_map` | `str` | `cuda:0` | 运行设备，如 `cuda:0`、`cpu`、`auto` |
| `dtype` | `str` | `bfloat16` | 模型精度（`float32/float16/bfloat16`） |
| `attn_implementation` | `str` | `""` | 注意力实现（如 `flash_attention_2`，留空用默认） |
| `source_lang` | `str` | `Chinese` | 默认源语言 |
| `target_lang` | `str` | `English` | 默认目标语言 |
| `generation_max_length` | `int` | `512` | 生成的最大 token 数 |
| `enable_thinking` | `bool` | `False` | 是否启用思考模式（翻译应保持 `False`） |
| `do_sample` | `bool` | `False` | 是否采样（`False` 为贪心解码，更稳定） |
| `temperature` | `Optional[float]` | `None` | 采样温度（仅 `do_sample=True` 时生效） |
| `top_p` | `Optional[float]` | `None` | 核采样阈值（仅 `do_sample=True` 时生效） |
| `output_dir` | `str` | `outputs` | 默认结果输出目录 |
| `raise_on_error` | `bool` | `False` | 出错时是否抛出异常（`False` 则记日志并返回空串） |
| `gen_kwargs` | `Optional[Dict]` | `None` | 透传给 `model.generate` 的额外参数 |

### 4.1 直接构造

```python
from src.translator import Qwen3TranslatorService

translator = Qwen3TranslatorService(model_id_or_path="models/Qwen3-1.7B")
english = translator.translate_zh_to_en("前方路口请左转")
print(english)
```
### 4.2 通过 YAML 配置加载

```python
from src.translator import Qwen3TranslatorService

translator = Qwen3TranslatorService.from_yaml("configs/translation/qwen3_translator.yaml")
english = translator.translate_zh_to_en("请减速至每小时40公里")
print(english)
```

### 4.3 批量翻译并保存 JSON

```python
results = translator.translate_zh_to_en( ["请靠边停车", "在下一个红绿灯右转"], output_json="outputs/translated.json")
```

### 4.4 翻译指令文件

```python
records = translator.translate_commands(input_file="data/commands/Chinese-commands.json", output_json="outputs/commands_translated.json", text_key="text")
```


## 5. 对外接口说明

+ `translate(text, source_lang=None, target_lang=None, output_json=None, return_meta=False)`：统一翻译入口
  + `text` 为 `str` 时返回 `str`；为 `List[str]` 时返回 `List[str]`
  + `output_json` 不为空时，将结果写入 JSON 文件
  + `return_meta=True` 时额外返回元信息字典
+ `translate_zh_to_en(text, output_json=None, return_meta=False)`：中文 → 英文快捷接口
+ `translate_en_to_zh(text, output_json=None, return_meta=False)`：英文 → 中文快捷接口
+ `translate_commands(input_file, output_json=None, text_key="text", **kwargs)`：翻译 JSON 指令文件，并为每条附加 `translation` 字段
+ `save_result(meta, path)`：将翻译结果写入 JSON 文件

## 6. 输出格式

+ 单条翻译输出 JSON：

```json
{ 
    "model": "models/Qwen3-1.7B", 
    "source": "前方路口请左转", 
    "translation": "Turn left at the intersection ahead.", 
    "success": true, 
    "source_lang": "Chinese", 
    "target_lang": "English", 
    "time_seconds": 0.42 
}
```

+ 批量翻译额外包含 `count` 与 `total_time_seconds`，其中 `source/translation/success/time_seconds` 为等长数组。

## 7. 配置文件

+ 配置文件放在 `configs/translation/` 下（示例见 `qwen3_translator.yaml`），键名与构造函数参数一一对应。

+ 使用 `Qwen3TranslatorService.from_yaml(path)` 加载；配置中未知的键会被忽略并打印告警，便于向前兼容。

## 8. 注意事项

+ `enable_thinking` 建议保持 `False`，确保 Qwen3 直接输出译文，不夹带 `<think>...</think>` 思考链（模块另含 `_strip_think` 兜底）。
+ 同一模型路径的权重在进程内只加载一次（模块级缓存），重复构造实例不会二次占用显存。
+ `raise_on_error=False`（默认）适合批量作业：单条失败不中断整批，失败项 `success=false`、`translation=""`，可据此重试补翻。

## 9. 参考资料
+ https://github.com/QwenLM/Qwen3