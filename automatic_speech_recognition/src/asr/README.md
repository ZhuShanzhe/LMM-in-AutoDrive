# FunASR 语音识别模块

+ 基于 FunASR 工具包封装的本地离线语音识别服务，支持单句识别和说话人识别两种模式。模块提供了简洁统一的接口，可对 WAV 音频文件进行转录，并支持将结果打印或保存为结构化 JSON 文件。
+ 双模式识别:
  + `single` 模式：将整个音频识别为一段文字，自动添加标点
```json
"single": 
{
    "model": "paraformer-zh",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "spk_model": None,
}
```
  + `speaker` 模式：识别不同说话人的语句，并标注说话人 ID（需音频含多人对话）

```json
"speaker": 
{
    "model": "iic/SenseVoiceSmall",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "spk_model": "cam++",
}
```

## 1. 项目结构
```text
asr/
├── __init__.py          
├── config.py            
├── funasr_model.py             
├── service.py
├── utils.py      
├── example.py           # 使用示例
├── requirements.txt     # 项目依赖列表
└── README.md
```

## 2. 运行环境

+ Python 版本：Python 3.8 及以上
+ CPU 运行：支持纯 CPU 运行（需内存 ≥ 8GB，推荐 16GB+）
+ GPU 支持：NVIDIA GPU + CUDA 11.8 及以上
+ 内存占用：首次运行时会自动从 ModelScope 下载模型文件（约 2 GB）
+ 依赖库安装：
```shell
pip install -r requirements.txt
```

## 3. Quick Start

+ 语音识别接口位于 `FunASRService` 类，构造函数的一些重要参数如下：

|      参数       |  类型   |        默认值        |                     说明                     |
|:-------------:|:-----:|:-----------------:|:------------------------------------------:|
|    `mode`     | `str` |    `"signle"`     | 识别模式：`"single"`（单句识别）或 `"speaker"`（说话人识别）  |
|   `device`    | `str` |     `cuda:0`      |         运行设备，如 `"cuda:0"`、`"cpu"`          |

+ 接口函数：`transcribe()` – 单条音频转录

|        参数         |       类型        |  默认值   |                 说明                       |
|:-----------------:|:---------------:|:------:|:----------------------------------------:|
|   `audio_path`    |      `str`      |   -    |    输入音频文件路径（推荐 WAV 格式，16kHz 采样率，单声道）     |
|   `output_json`   | `Optional[str]` | `None` |          若提供，将转录结果保存到该 JSON 文件           |

+ 接口函数：`transcribe_batch()`– 批量音频转录


|        参数         |       类型             |  默认值   |                                       说明                         |
|:-----------------:|:--------------------:|:------:|:----------------------------------------------------------------:|
|   `audio_paths`   |     `List[str]`      |   -    |                             音频文件路径列表                             |
| `output_json_dir` |   `Optional[str]`    | `None` |          若提供，为每个音频生成同名的 JSON 文件（如 `audio1.json`）保存在该目录下          |


