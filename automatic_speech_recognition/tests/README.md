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
├── commands.py
├── wav_commands.py
├── noise_commands.py
└── README.md

data/
├── test_results/
│   ├── ASR_result.json
│   ├── test_summary.json
|   ├── ASR_result_qwen3.json
|   └── test_summary_qwen3.json
|
├── test_results_noise/
|   ├── normal/
|   |   ├── ASR_result_qwen3.json
|   |   └── test_summary_qwen3.json
|   └── noise/
|       ├── ASR_result_qwen3.json
|       └── test_summary_qwen3.json
|
├── wav_files/
│   ├── file_mapping.json
│   ├── command_0001.wav
│   ├── ...
│   └── command_8349.wav
|
├── wav_files_noise/
|   ├── wav_files_with_noise/
│   |   ├── file_mapping_noise.json
│   |   ├── command_1_noise.wav
│   |   ├── ...
│   |   └── command_500_noise.wav
|   |
|   └── wav_files_without_noise/
|       └──file_mapping_without_noise.json
|
├── commands.json              # 原始文本指令集
├── translated_commands.json   # 中文文本指令集
└── logging/                   # 日志文件夹
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

## 4. 测试结果
+ 以上三种指标均为错误率，我们在评估模型时使用正确率（$1-e$）以让结果更直观。
+ 在无噪声优化的基础 pipeline 下，模型在 `8349` 条标准普通话语音指令下进行了测试，结果字符准确率高达 **95.2%**，词准确率达到 86.2%，但句子准确率仅有 45.9%，模型识别关键字词的能力较强，但识别的句子与原始句子相比在开头或结尾处容易出现单个字词的错误。

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

+ 在改进后的 Qwen3-ASR 模型下，模型在 `8349` 条标准普通话语音指令下进行了测试，结果字符准确率高达 **96.0%**，词准确率达到 89.5%，句子准确率提高到 48.6%，模型的性能得到了一定的提高，但仍受限于生成语音指令数据集的质量欠佳。

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
