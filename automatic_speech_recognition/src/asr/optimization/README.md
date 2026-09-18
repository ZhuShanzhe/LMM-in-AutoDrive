# ASR 推理期优化模块（optimization）

+ 面向自动驾驶语音指令场景的推理期免训练优化模块，针对噪声与方言两类退化问题，在不修改 Qwen3-ASR 模型权重的前提下提升识别质量。
+ 模块分三层：信号层 `frontend`（降噪/归一）、文本层 `dialect`（方言归一）、装配层 `optimizer`，并通过 `ASROptimizer.as_hooks()` 直接对接 `Qwen3ASRService` 的 `frontend` 与 `dialect_normalizer`。
+ 方言词典以 JSON 资源形式独立存放于 `resources/dialect/`，便于扩展与维护；新增方言无需改动代码。

## 1. 文件结构

```text
optimization/
├── __init__.py 
├── optimizer.py 
├── frontend.py 
├── dialect.py   
├── lexicons.py
└── README.md
```

## 2. 环境要求

| 项目 | 推荐配置 |
|:-:|:-:|
| Python | 3.10 ~ 3.12 |
| 依赖 | `numpy`、`librosa`、`soundfile` |
| 可选（高精度降噪） | `torch`、`deepfilternet` |
| 可选（拼音纠错） | `pypinyin` |
| 可选（高通滤波） | `scipy` |

+ 安装核心依赖：

```shell
pip install numpy librosa soundfile
```

+ 可选依赖（按需安装）：

```shell
pip install scipy pypinyin          # 高通滤波 / 拼音同音纠错
pip install torch deepfilternet     # DeepFilterNet 高精度降噪
```

## 3. 词典资源

+ 方言词典以 JSON 形式存放在项目根目录 `resources/dialect/` 下：

```text
resources/dialect/
├── _aliases.json      # 共享：驾驶领域别名（左拐→左转、刹车→制动 等），对所有方言生效
├── sichuan.json       # 四川话
├── dongbei.json       # 东北话
├── beijing.json       # 北京话
├── tianjin.json       # 天津话
├── shandong.json      # 山东话
├── henan.json         # 河南话
├── shaanxi.json       # 陕西话
├── shanghai.json      # 上海话
├── cantonese.json     # 粤语
├── hunan.json         # 湖南话
├── hubei.json         # 湖北话
└── yunan.json         # 云南话
```

+ 每个词典文件格式如下，`words` 为 `{方言词: 标准词}` 映射：

```json
{
  "name": "sichuan",
  "display_name": "四川话",
  "words": {
    "啥子": "什么",
    "咋个": "怎么",
    "刹一脚": "制动",
    "前头": "前面"
  }
}
```

+ 新增方言只需在 `resources/dialect/` 增加一个同名 JSON 文件，`available_dialects()` 会自动发现，代码无需改动。

## 4. 快速开始

### 4.1 信号层：AudioFrontend

+ 处理链：`VAD 裁剪 → 高通滤波 → 降噪 → 预加重 → RMS 归一 → 峰值限制`。

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `method` | `str` | `wiener` | 降噪方法：`spectral` / `wiener` / `deepfilternet` / `none` |
| `vad` | `bool` | `True` | 是否做静音裁剪 |
| `highpass_filter` | `bool` | `True` | 是否高通滤波（去低频轰鸣） |
| `highpass_cutoff` | `float` | `80.0` | 高通截止频率（Hz） |
| `preemphasis` | `bool` | `True` | 是否预加重 |
| `rms_norm` | `bool` | `True` | 是否 RMS 响度归一 |
| `target_db` | `float` | `-20.0` | 目标响度（dB） |

```python
from src.asr.optimization import AudioFrontend

frontend = AudioFrontend(method="wiener", vad=True)
enhanced = frontend(audio_array, sr=16000)
```

### 4.2 文本层：DialectNormalizer

+ 归一链：`词典替换（长词优先） → 拼音同音纠错 → 热词模糊吸附`。

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `dialect` | `Optional[str]` | `None` | 方言名（见 `available_dialects()`）；`None` 仅用共享别名 |
| `custom_map` | `Optional[Dict]` | `None` | 额外 `{方言词: 标准词}` 覆盖 |
| `lexicon_dir` | `Optional[str]` | `None` | 词典目录，`None` 指向 `resources/dialect` |
| `use_shared_aliases` | `bool` | `True` | 是否合并 `_aliases.json` |
| `fuzzy_threshold` | `float` | `0.85` | 热词模糊匹配阈值 |
| `hotwords` | `Optional[List]` | `None` | 需强制吸附的关键驾驶词 |
| `enable_pinyin` | `bool` | `True` | 是否启用拼音同音纠错（需 `pypinyin`） |

```python
from src.asr.optimization import DialectNormalizer

normalizer = DialectNormalizer(dialect="sichuan", hotwords=["左转", "制动"])
text = normalizer.normalize("前头刹一脚，左拐")   # -> "前面制动，左转"
```

### 4.3 装配层：build_optimizer

+ 从 YAML 配置一次性构建并组合两层，返回可对接 ASR 服务的接口。

```python
import yaml
from src.asr import Qwen3ASRService
from src.asr.optimization import build_optimizer

cfg = yaml.safe_load(open("configs/asr/optimization.yaml", encoding="utf-8"))
optimizer = build_optimizer(cfg)

frontend, dialect_normalizer = optimizer.as_hooks()
service = Qwen3ASRService.from_yaml(
    "configs/asr/qwen3_asr.yaml",
    frontend=frontend,
    dialect_normalizer=dialect_normalizer,
)
result = service.transcribe_file("noisy_sichuan.wav", output_json="outputs/asr.json")
```

## 5. 对外接口说明

+ `ASROptimizer(frontend, dialect, enable_frontend=True, enable_dialect=True)`
  + `process_audio(audio, sr)`：启用时调用前端，否则原样返回
  + `process_text(text)`：启用时做方言归一，否则原样返回
  + `as_hooks()`：返回 `(frontend, dialect_normalizer)`，可直接传给 `Qwen3ASRService`
+ `build_optimizer(cfg)`：从配置字典构建 `ASROptimizer`
+ `AudioFrontend(...)`：可调用对象 `frontend(audio, sr) -> audio`
+ `DialectNormalizer(...).normalize(text) -> text`
+ 词典工具：`available_dialects()`、`load_lexicon(name)`、`build_lexicon(dialect, extra)`

## 6. 配置文件

+ 推理期优化配置位于 `configs/asr/optimization.yaml`，字段与上述构造参数一一对应。

## 7. 注意事项

+ 本模块不需要训练，仅作用于推理期；模型级微调属于 `training/` 目录。
+ `method="deepfilternet"` 时若未安装 `deepfilternet`，会自动回退为维纳滤波并打印告警，不中断流程。
+ `enable_pinyin=True` 但未安装 `pypinyin` 时，拼音纠错自动跳过，词典与热词归一仍生效。
+ 词典目录默认解析为 `项目根/resources/dialect`，与文件所在层级解耦；如自定义路径可用 `lexicon_dir` 覆盖。