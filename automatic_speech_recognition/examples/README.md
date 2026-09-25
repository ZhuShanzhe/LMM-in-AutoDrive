# 示例代码（examples）

本目录提供 `automatic_speech_recognition` 各模块的最小可运行示例。所有脚本均以**仓库根目录**为工作目录运行。

## 运行方式

```shell
cd automatic_speech_recognition
python examples/asr.py
python examples/translator.py
python examples/tts.py
python examples/demo.py
```

## 文件说明

| 文件 | 对应模块 | 示例内容 |
|:---|:---|:---|
| `asr.py` | `src/asr` | 语音识别：单文件转写、批量转写；并演示启用推理期优化（降噪前端 + 方言归一）后的转写 |
| `translator.py` | `src/translator` | 中英翻译：单条翻译、带元信息（耗时）翻译、批量翻译并保存 JSON；以及通过 YAML 配置加载翻译服务 |
| `tts.py` | `src/tts` | 语音合成：单条合成、方言音色（`dialect="sichuan"` 等）、合成时叠加噪声、批量合成 |
| `demo.py` | `src/pipeline` | **端到端流水线**：语音 → (可选)降噪/方言优化 → (可选)中译英，演示 `from_yaml` 构建与显式参数构建、`process` / `process_batch`、`output_language` 切换与 JSON 落盘 |

## 说明

+ `demo.py` 演示的是对外统一接口 `ASRPipeline`，可设置：模型路径、是否优化（及优化配置）、是否翻译、输出语言（`chinese` / `english` / `both`）、输出文件地址等。
+ 各示例默认读取 `data/` 下的音频路径；若数据尚未生成，请先按 `training/dataset/` 的流程构建数据集，或把脚本中的音频路径改为你本地已有的文件。
