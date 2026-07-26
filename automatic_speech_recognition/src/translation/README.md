# 基于 Qwen2.5 的中英文翻译模块

## 1. 项目结构
```text
translation/
├── __init__.py
├── config.py          # 模型配置文件
├── translator.py      # 模型
├── service.py         # 统一对外接口
├── example.py         # 示例代码
├── requirements.txt   # 项目依赖列表
└── README.md
```

## 2. 运行环境

+ Python 版本：Python 3.9 以上，已在 Python 3.10 上进行测试
+ CPU 运行：支持纯 CPU 运行（需内存 ≥ 8GB，推荐 16GB+）
+ 最低显存 (GPU)：约 4 GB（使用 4-bit 量化加载时）；约 6 GB（FP16 半精度时）
+ 硬盘空间：模型文件约 6 GB（缓存目录）
+ 依赖库安装：
```shell
pip install -r requirements.txt
```

## 3. Quick Start
### 3.1 模型下载
+ 可以选择设置 `load_type` 更改模型加载的方式，默认为 `'custom'`（从 Hugging face 下载），也可以设置为 `'local'`（从本地加载）并设置 `model_path` 为正确的模型路径。
+ 如果遇到网络连接问题，可使用镜像网站下载：
```shell
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

### 3.2 接口文档
+ 模型对外接口位于 `service.py` 中的 `Translation` 类，初始化的参数设置说明如下：

|          参数             |         类型          |              默认值              |                         说明                          |
|:-----------------------:|:-------------------:|:-----------------------------:|:---------------------------------------------------:|
|      `model_name`       |        `str`        | `"Qwen/Qwen2.5-3B-Instruct"`  |   Hugging Face 模型标识符（当 `load_type='custom'` 时生效）    |
|       `src_lang`        |        `str`        |         `"eng_Latn"`          |                        源语言类别                        |
|       `tgt_lang`        |        `str`        |         `"zho_Hans"`          |                       目标语言类别                        |
|       `load_type`       |        `str`        |          `"custom"`           | 加载方式：`"custom"`（从 Hugging Face 下载）或 `"local"`（本地加载） |
|      `model_path`       |   `Optional[str]`   |            `None`             |        当 `load_type='local'` 时必须指定，本地模型目录路径         |
|      `max_length`       |        `int`        |             `512`             |                   输入文本最大 Token 长度                   |
| `generation_max_length` |   `Optional[int]`   |            `None`             |         输出最大 Token 长度（默认与 `max_length` 相同）          |
|       `num_beams`       |        `int`        |              `1`              |            束搜索宽度（`1` 为贪心解码，增大可提升质量但增加耗时）            |
|      `temperature`      |       `float`       |             `0.1`             |                 采样温度（越低输出越确定，越高越随机）                 |
|    `raise_on_error`     |        `bool`       |            `False`            |         出错时是否抛出异常（`True`）或返回空字符串并记录日志（`False`）          |

+ 对外接口请使用 `translate()` 函数，可设置的参数说明如下：


|         参数           |            类型            |     默认值      |                               说明                               |
|:--------------------:|:------------------------:|:------------:|:--------------------------------------------------------------:|
|        `text`        | `Union[str, List[str]]`  |      -       | 待翻译的内容。如果传入 `str`，方法返回 `str`；如果传入 `List[str]`，方法返回 `List[str]` |
|      `src_lang`      |          `str`           | `"eng_Latn"` |                             源语言类别                              |
|      `tgt_lang`      |          `str`           | `"zho_Hans"` |                             目标语言类别                             |
|  `json_output_path`  |     `Optional[str]`      |    `None`    |             结构化 JSON 输出文件路径，若为 `None` 则不输出 JSON 文件             |

+ JSON 输出文件格式：

```json
{
  "source": [输入的原文],
  "translation": [翻译后的结果],
  "time_seconds": [该次翻译的耗时（秒）]
}
```

+ 具体使用示例见 `example.py`