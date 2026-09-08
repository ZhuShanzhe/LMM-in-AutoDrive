# ASR 测试评估模块

+ 本模块用于对语音识别（ASR）系统进行批量测试和性能评估，计算多种指标并生成详细报告。

## 1. 目录结构
```text
tests/
├── utils/                     
│   ├── data_loader.py
│   ├── evaluator.py 
│   └── metrics.py
|   
├── asr_test.py
├── qwen_test.py
├── denoise_test.py
├── accent_test.py
├── commands.py
├── wav_commands.py
├── noise_commands.py
└── README.md

data/
├── results/                    # ASR 识别结果 json 报告 
├── test_results/               # 基础数据集测试结果
│   ├── ASR_result.json
│   ├── test_summary.json
|   ├── ASR_result_qwen3.json
|   └── test_summary_qwen3.json
|
├── test_results_noise/         # 噪声对比实验
|   ├── normal/
|   |   ├── ASR_result_qwen3.json
|   |   └── test_summary_qwen3.json
|   └── noise/
|       ├── ASR_result_qwen3.json
|       └── test_summary_qwen3.json
|
├── test_results_denoising/     # 噪声优化测试
|   ├── denoisy_mapping.json
|   ├── ...
|   └── comparison.json
|
├── test_results_accent/        # 方言优化测试
|   ├── Guangzhou_details.json
|   ├── ...
|   └── Sichuan_details.json
|
├── wav_files/                  # 数据集
│   ├── file_mapping.json
│   ├── command_0001.wav
│   ├── ...
│   └── command_8349.wav
|
├── wav_files_noise/            # 噪声数据集
|   ├── wav_files_with_noise/
│   |   ├── file_mapping_noise.json
│   |   ├── command_1_noise.wav
│   |   ├── ...
│   |   └── ...
|   |
|   └── wav_files_without_noise/
|       └──file_mapping_without_noise.json
|
├── wav_files_accent/            # 方言语音数据集
|   ├── Dongbei Dialect Speech Corpus for TTS/
|   ├── ...
|   └── Sichuan Dialect Speech Corpus for TTS/
|
├── commands.json              # 原始文本指令集
├── translated_commands.json   # 中文文本指令集
└── logging/                   # 日志
```

+ 功能概述：
  + 从 JSON 文件中读取测试样本（每个样本包含音频路径和参考文本）；

  + 调用 ASR Pipeline 对每个音频进行识别（不启用翻译，仅做中文语音识别）；

  + 将识别结果与参考文本对比，计算 CER、WER 和句子准确率；

  + 输出总体指标和每个样本的详细结果，保存为 JSON 文件。

## 2. 评估指标详解

### 2.1 CER (Character Error Rate) — 字符错误率
+ CER 是评估语音识别系统性能最核心的指标之一，它衡量识别结果与标准答案在字符级别的差异程度。对于中文等表意文字系统，CER 是最直接、最常用的评价标准。
+ 计算公式：$\text{CER}=\frac{S+D+I}{N}\times 100\%$

    + $S$ (Substitutions)：替换错误。识别结果中的某个字符被错误地替换为另一个字符。
例如：参考文本为 "减速"，识别为 "加速" → 1 个字符被替换（$S=1$）

    + $D$ (Deletions)：删除错误。识别结果漏掉了参考文本中的某个字符。例如：参考文本为 "请减速"，识别为 "减速" → 漏掉“请”（$D=1$）

  + $I$ (Insertions)：插入错误。识别结果中多出了参考文本中没有的字符。例如：参考文本为 "减速"，识别为 "减减速" → 多出一个“减”（$I=1$）

  + $N$ (Total Characters)：参考文本的总字符数。

+ 计算示例：

|          参考文本          |        识别文本         | S | D | I | N  | CER  |
|:--------------------------:|:------------------------:|:-:|:-:|:-:|:--:|:-----:|
| 请减速至40公里每小时 | 请减速至40公里每小时 | 0 | 0 | 0 | 10 | 0% |
| 请减速至40公里每小时 | 减速至40公里每小时 | 0 | 1 | 0 | 10 | 10% |
| 请减速至40公里每小时 | 请加速至40公里每小时 | 1 | 0 | 0 | 10 | 10% |
| 请减速至40公里每小时 | 请减速到40公里每小时 | 1 | 0 | 0 | 10 | 10% |
| 请减速至40公里每小时 | 请减速至40公里 | 0 | 2 | 0 | 10 | 20% |

### 2.2 WER (Word Error Rate) — 词错误率

+ WER 是语音识别中另一个广泛使用的指标，它衡量识别结果与标准答案在词级别的差异程度。英文等使用空格分隔的语言天然适合 WER。
+ 计算公式：$\text{WER}=\frac{S_w+D_w+I_w}{N_w}\times 100\%$
+ 在中文场景下的特殊说明：

  + 中文文本是连续书写的，没有天然的词边界。因此，中文 WER 的计算依赖于分词器的选择。本测试默认使用简单的空格分词（`text.split()`），这会导致：
    + 若参考文本和识别文本均为连续中文（无空格），则整句被视为 1 个词。此时 WER 的计算方式退化为：只要句子不是完全相同，WER 就接近 100%。

+ 改善中文 WER 的方法：
  + 使用 `jieba` 等中文分词器进行分词：
```python
import jieba
evaluator = ASREvaluator(tokenizer=jieba.lcut)
```

### 2.3 SER (Sentence Error Rate)——句子错误率
+ 在 SER 测试时，整句只要存在任意一个字符错误，即判定为错误句子。
+ SER 可以反映语音识别后语句的完整性与可用性，在指令类任务中可作为模型性能评估的指标之一。

## 3. 测试代码运行
+ 按照数据集的构建方法，标准测试部分分三步进行：
1. 将标准 Talk2Car 文本指令 `data/commands.json` 翻译为标准中文指令集，保存在 `data/translated_commands.json`

```shell
python commands.py --dataset data/commands.json --output_dir data/translated_commands.json --load_type local --model_path models/Qwen2.5-3B-Instruct
```

2. 将标准中文指令集中的指令逐条生成对应的语音指令 WAV 文件，保存在 `data/wav_files/` 下

```shell
python wav_commands.py --dataset data/translated_commands.json --output_dir data/wav_files --load_type local --model_path models/rvcmd_linux_amd64 --noise_enabled False
```

3. 执行 `pipeline` 用于对每个 WAV 文件进行语音识别并记录识别结果，最后进行指标计算，结果保存在 `data/test_results/test_summary.json`

```shell
python asr_test.py --dataset data/wav_files/file_mapping.json --output_dir data/test_results --asr_device cuda:0
```

或者执行改进后基于 `Qwen3-ASR` 的 `pipeline2` 用于对每个 WAV 文件进行语音识别并记录识别结果，最后进行指标计算，结果保存在 `data/test_results/test_summary_qwen.json`

```shell
python qwen_test.py --dataset data/wav_files/file_mapping.json --output_dir data/test_results --asr_device cuda:0 --load_type local --model_path models/Qwen3-ASR-1.7B
```

+ 以上三种指标均为错误率，我们在评估模型时使用正确率（$1-e$）以让结果更直观。
+ 在无噪声优化的基础 pipeline 下，模型在 `8349` 条标准普通话语音指令下进行了测试，结果字符准确率高达 **95.19%**，词准确率达到 86.19%，但句子准确率仅有 45.86%，模型识别关键字词的能力较强，但识别的句子与原始句子相比在开头或结尾处容易出现单个字词的错误。

```json
"average_cer": 0.04814557975262654,
"average_wer": 0.1380501254633386,
"sentence_accuracy_rate": 0.4585938435740807
```
```json
"reference": "在那辆灰色汽车到达之前穿过交叉路口。",
"hypothesis": "在那辆灰色汽车到达之前穿过交叉路口。",
"original": "pass the intersection before the grey car arrives. "
```

+ 在改进后的 Qwen3-ASR 模型下，模型在 `8349` 条标准普通话语音指令下进行了测试，结果字符准确率高达 **96.04%**，词准确率达到 89.45%，句子准确率提高到 48.58%，模型的性能得到了一定的提高，但仍受限于生成语音指令数据集的质量欠佳。

```json
"average_cer": 0.03956448642714553,
"average_wer": 0.1054857927638226,
"sentence_accuracy_rate": 0.48578272847047553
```

```json
"reference": "在那辆灰色汽车到达之前穿过交叉路口。",
"hypothesis": "在那辆灰色汽车到达之前，穿过交叉路口。",
"original": "pass the intersection before the grey car arrives. "
```

## 5. 噪声测试
+ 为了验证噪声对于模型性能的影响，我们使用 `TTS` 模块生成了一个简单的带噪声的语音指令数据集：
  + 使用随机种子（`seed=42`）从原始数据集中随机抽取 500 条指令，确保实验可复现。抽取后的噪声子集保存在 `data/wav_files_noise/wav_files_with_noise/file_mapping_noise.json`
  + 通过 `index` 匹配，从原始数据集中提取与噪声子集对应的 500 条无噪声指令，保存在 `data/wav_files_noise/wav_files_without_noise/file_mapping_without_noise.json`，用于对比实验的基准（baseline）
  + 使用 `ChatTTS` 模型对抽取的 500 条中文指令进行语音合成，并添加白噪声（`noise_type="white", noise_level=0.01`），生成带噪 WAV 文件，保存在 `data/wav_files_noise/wav_files_with_noise/` 目录下，文件命名格式为 `commands_{idx}_noise.wav`
```shell
python noise_commands.py
```
+ 执行以下命令进行噪声条件下的 ASR 评估：
```shell
python qwen_test.py \
    --dataset data/wav_files_noise/wav_files_without_noise/file_mapping_without_noise.json \
    --output_dir data/test_results_noise/normal \
    --asr_device cuda:0 \
    --load_type local \
    --model_path models/Qwen3-ASR-1.7B
```
```shell
python qwen_test.py \
    --dataset data/wav_files_noise/wav_files_with_noise/file_mapping_noise.json \
    --output_dir data/test_results_noise/noise \
    --asr_device cuda:0 \
    --load_type local \
    --model_path models/Qwen3-ASR-1.7B
```

+ 简单的测试结果对比如下：
```json
# 无噪声（基准）
"average_cer": 0.04709145151247135,
"average_wer": 0.1597904761904763,
"sentence_accuracy_rate": 0.46
```
```json
# 引入噪声
"average_cer": 0.10391789058209343,
"average_wer": 0.2275714285714285,
"sentence_accuracy_rate": 0.364
```

+ 以上实验结果表明噪声会对 ASR 性能产生一定影响，在评价指标中句子准确率对噪声最为敏感：
```json
# 无噪声（基准）
"reference": "停下，你左边的卡车想变道。",
"hypothesis": "停下！你左边的卡车想变道。",
"original": "stop, the truck on your left side wants to pull out."
```
```json
# 引入噪声
"reference": "停下，你左边的卡车想变道。",
"hypothesis": "停下！你左边的卡车上变道。",
"original": "stop, the truck on your left side wants to pull out."
```

## 6. 噪声优化模块测试

+ 我们需要测试为 `pipeline` 引入 `DeepFilterNet3` 进行降噪预处理对 ASR 性能的提升效果。
+ 为了排除 `TTS` 合成音频质量可能不稳定带来的干扰，我们通过一种特殊的方法构建测试集：
  + 选取原始数据集中经过 `ASR` 识别准确率较高的语音样本，从中随机抽取 1000 个样本并为其添加随机强度的噪声，得到噪声数据集（`data/wav_files_noise/wav_files_with_noise/file_mapping_noise.json`）
+ 基于噪声数据集调用 `DeepFilterNet3` 模型对每条带噪音频进行降噪处理，生成降噪后的 WAV 文件，并建立新的映射文件 `denoisy_mapping.json`.
+ 噪声的相关配置可以在 python 脚本中查看和设置。
+ 运行脚本：
```shell
python denoise_test.py \
    --dataset data/wav_files_noise/wav_files_with_noise/file_mapping_noise.json \
    --original_dataset data/wav_files/file_mapping.json \
    --output_dir data/test_results_denoising \
    --asr_device cuda:0 \
    --load_type local \
    --model_path models/Qwen3-ASR-1.7B \
    --denoiser_model DeepFilterNet3 \
    --denoiser_output_sr 16000
```

+ 相关参数设置：

|          参数          |             说明             |
|:--------------------:|:--------------------------:|
|     `--dataset`      |       带噪数据集的 JSON 路径       |
| `--original_dataset` |  原始干净数据集的 JSON（用于提取干净子集）   |
|    `--output_dir`    |           输出根目录            |
|    `--asr_device`     |          ASR 运行设备          |
|     `--load_type`      | ASR 模型加载方式（`local` / `custom`） |
|     `--model_path`     |      Qwen3‑ASR 本地模型路径      |
|   `--denoiser_model`   | 降噪模型名称（默认 `DeepFilterNet3`）  |
| `--denoiser_output_sr` |    降噪后音频的采样率（默认 16000）     |

+ 分别对干净基线（从原始数据集中提取对应的采样样本）、带噪数据和降噪数据运行 Qwen3‑ASR 测试，得到三组识别结果：

|          模型          |  CER↓  |  WER↓  |   SAR↑   |
|:--------------------:|:------:|:------:|:--------:|
|        Origin        | 0.0003 | 0.0055 |  0.994   |
|         ASR          | 0.0390 | 0.2311 |  0.766   |
| ASR + DeepFilterNet3 | 0.0179 | 0.1204 |  0.877   |

+ 对比 `comparison.json` 中的测试指标，可得出以下结论：
  + 原 Qwen3‑ASR 模型对噪声已经具有较好的鲁棒性，添加噪声后 CER 仅下降了约 4%，但 WER 下降了超过 20%，识别出的句子在中间部分词语上可能出现一定偏差，从而影响后续的指令解析。
  + 降噪处理后各项指标均有所提高，WER 上升超 10%，证明降噪优化可以在有环境噪声的场景下提高语音识别的准确率，但相对应地优化部分需要占用一定的响应时间，导致处理效率略有下降。

## 7. 方言优化模块测试

+ 当前在自动驾驶领域针对方言的语音指令数据集较为缺乏，我们选取了 Magic Data 开源的五种方言 TTS 数据集（MagicData-Dialect-TTS-Lite）。该数据集包含五种方言变体：东北话、河南话、四川话、吴语（江苏话）和粤语，每种方言包含若干语音片段（WAV 文件），由年龄在 30 至 60 岁之间的当地母语者录制，保留了地道的口音和表达习惯。数据集遵循 CC BY-NC-ND 4.0 许可，仅限学术研究使用。
+ 针对方言部分，我们测试基线方法 `ASR`（未引入方言优化）与改进后的基于 Qwen3‑ASR 的 `ASR2` 进行对比测试，测试脚本如下：

```shell
python accent_test.py \
    --data_root data/wav_files_accent \
    --output_dir data/test_results_accent \
    --asr_device cuda:0 \
    --model_path models/Qwen3-ASR-1.7B
```

|      参数       | 说明 |
|:-------------:|:---:|
|  `--data_root`  | 方言数据集根目录 |
| `--output_dir`  | 输出结果目录 |
| `--asr_device`  | ASR 运行设备 |
| `--model_path`  | Qwen3‑ASR 本地模型路径 |
| `--max_samples` | 每种方言最大测试样本数 |

+ 测试结果：


|    方言     |   模型   |       CER↓        |    WER↓    |
|:---------:|:------:|:-----------------:|:----------:|
|  Dongbei  | Origin |      0.0982       |   0.9733   |
|  Dongbei  |  ASR2  |    **0.0756**     | **0.7467** |
|   Henan   | Origin |      0.0868       |   0.9459   |
|   Henan   |  ASR2  |    **0.0634**     | **0.6108** |
|  Sichuan  | Origin |      0.0953       |   0.9221   |
|  Sichuan  |  ASR2  |    **0.0590**     | **0.6442** |
| Guangzhou | Origin |      0.3730       |   0.9835   |
| Guangzhou |  ASR2  |    **0.2039**     | **0.8454** |
|  Jiangsu  | Origin |      0.7543       |   0.9886   |
|  Jiangsu  |  ASR2  |    **0.3177**     | **0.9024** |

+ 简单总结：
  + Qwen3-ASR 在方言识别上全面优于 FunASR：Qwen3-ASR 原生支持 30 种语言和 22 种中文方言，在五种方言上的 CER 均有所降低，尤其在较为困难的粤语和江苏话上提升较为明显。
  + 方言难度差异显著：
    + 粤语和江苏话对两个模型都是最具挑战性的方言，CER 分别高达 20.39% 和 31.77%（Qwen3-ASR）。
    + 河南话和东北话属于官话方言区，与普通话在声调、词汇上的差异相对较小，CER 分别为 6.34% 和 7.56%（Qwen3-ASR），识别效果较好。
+ 未来展望：
  + 模型微调：对于较为困难的目标方言，可使用更大规模、标注更规范的方言数据进行模型微调。
  + 方言词汇映射：针对语音指令中的一些常用词汇，建立方言→普通话的词汇映射表，对 ASR 输出进行后处理修正。
  + 数据增强：利用 TTS 合成更多方言语音数据，或对现有音频进行变速、加噪等处理，扩充训练集多样性。
