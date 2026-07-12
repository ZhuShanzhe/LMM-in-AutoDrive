# DriveLM-CARLA 基线复现实验记录

## 实验目标

负责人：朱善哲

本阶段目标是在 CARLA 数据上运行 DriveLM 官方 LLaMA-Adapter V2 7B 基线，完成环境配置、数据下载、图文对齐、输入转换、模型推理和结果分析。

本次实验使用当前已完成图文对齐的子集：

```text
Town01 / ControlLoss
```

本次运行范围：

```text
关键帧图片：274
QA 数量：7432
```

需要说明：

- 本次运行的是 `Town01/ControlLoss` 子集的全部数据。
- 7432 是 QA 数量，不是图片数量。
- 该子集不是完整的 DriveLM-CARLA 数据集。
- 官方 DriveLM-CARLA 标注包含 182491 个 VQA JSON。
- 当前只下载了 `Town01/ControlLoss` 对应的 PDM-Lite CARLA 原始图片，因此其他官方标注暂时无法进行图文推理。

## 服务器与环境

项目路径：

```text
/root/autodl-tmp/LMM-in-AutoDrive
```

服务器配置：

```text
CPU：Intel Xeon Platinum 8470Q，25 核
内存：90 GB
GPU：NVIDIA GeForce RTX 5090
驱动版本：580.105.08
系统 CUDA：13.0
PyTorch CUDA Runtime：12.8
GPU 计算能力：sm_120 / (12, 0)
```

Conda 环境位于数据盘：

```text
/root/autodl-tmp/conda_envs/drivelm_carla
```

环境激活方式：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/drivelm_carla
```

核心版本：

```text
Python：3.10.20
PyTorch：2.11.0+cu128
torchvision：0.26.0+cu128
torchaudio：2.11.0+cu128
NumPy：1.26.4
OpenCV：4.6.0
CARLA：0.9.15
pygame：2.6.0
```

GPU 验证结果：

```text
CUDA 可用：是
GPU：NVIDIA GeForce RTX 5090
计算能力：(12, 0)
GPU 矩阵运算：通过
```

RTX 5090 属于 Blackwell 架构，需要 `sm_120` 支持，因此没有安装官方旧依赖中指定的 PyTorch 2.0.0+cu117，而是保留支持 RTX 5090 的 PyTorch 2.11.0+cu128。

已安装的基线依赖：

```text
sentencepiece
fairscale
tensorboard
timm
openai-clip
tenacity
```

## 代码仓库

DriveLM：

```text
experiments/drivelm/external/DriveLM
提交版本：1de72a7
```

DriveLM-CARLA：

```text
experiments/drivelm/external/DriveLM-CARLA
提交版本：0737c33
```

DriveLM 官方仓库提供了挑战基线，但没有直接发布完整的 CARLA 推理入口。因此，本实验在官方 LLaMA-Adapter V2 基线基础上增加了 CARLA 单相机输入适配。

## 模型权重

LLaMA 7B 主干权重：

```text
/root/autodl-tmp/models/llama_model_weights/tokenizer.model
/root/autodl-tmp/models/llama_model_weights/7B/checklist.chk
/root/autodl-tmp/models/llama_model_weights/7B/consolidated.00.pth
/root/autodl-tmp/models/llama_model_weights/7B/params.json
```

LLaMA-Adapter V2 权重：

```text
/root/autodl-tmp/models/llama_adapter_v2/BIAS-7B.pth
```

CLIP ViT-L/14 权重缓存：

```text
/root/autodl-tmp/model_cache/clip/ViT-L-14.pt
```

本次使用的是 LLaMA-Adapter V2 `BIAS-7B` 零样本权重，不是专门在 DriveLM-CARLA 上微调后的权重。因此，本次结果反映的是通用多模态模型直接迁移到 CARLA 驾驶问答任务时的零样本性能。

## RTX 5090 与 CARLA 适配

完成了以下适配：

1. 保留 PyTorch 2.11.0+cu128，避免旧版 CUDA 11.7 PyTorch 覆盖当前环境。
2. 将 CLIP 权重缓存放到数据盘，避免占用系统盘。
3. 将 CARLA 单张前视图片统一转换为以下张量结构：

```text
[批量, 视角数, 通道, 高度, 宽度]
```

单相机的视角数为 1。

4. 官方代码默认按照批量 32 分配 KV Cache，本实验在模型进入推理状态前将缓存批量修改为实际批量大小。
5. 增加每 25 条预测自动保存功能。
6. 增加断点续跑功能。
7. 使用 PIL 以 RGB 模式读取图片，避免 OpenCV 默认 BGR 顺序造成颜色通道错误。

## CARLA 原始数据

当前已下载 PDM-Lite CARLA 子集：

```text
Town01/data/ControlLoss
```

数据路径：

```text
experiments/drivelm/data/PDM_Lite_Carla_LB2
```

数据大小：

```text
约 460 MB
```

原始前视图片：

```text
分辨率：1024 × 512
颜色模式：RGB
```

## 官方 DriveLM-CARLA 标注

官方标注路径：

```text
experiments/drivelm/data/DriveLM/drivelm_carla_keyframes.txt
experiments/drivelm/data/DriveLM/drivelm_carla_vqas
```

标注统计：

```text
drivelm_carla_keyframes.txt：196313 行
drivelm_carla_vqas：182491 个 JSON
解压后大小：约 5.1 GB
```

单个 VQA JSON 的主要字段：

```text
key_object_infos
QA
image_paths
```

QA 类别：

```text
perception：场景与目标感知
prediction：交通参与者行为预测
planning：自车规划与驾驶决策
behavior：高层驾驶行为
```

QA 图关系字段：

```text
con_up
con_down
cluster
layer
object_tags
```

## 图文对齐子集

基于本地已有图片与官方 DriveLM-CARLA 标注，构建了 `Town01/ControlLoss` 图文对齐子集。

对齐结果：

```text
官方 VQA JSON：274
匹配图片：274
缺失图片：0
```

Route 分布：

```text
Route0_Rep0：41
Route1_Rep0：39
Route2_Rep0：60
Route3_Rep0：44
Route4_Rep0：51
Route5_Rep0：39
```

对齐文件：

```text
experiments/drivelm/outputs/town01_controlloss_official_pairs.jsonl
experiments/drivelm/outputs/town01_controlloss_official_pairs_summary.json
```

## 模型输入

274 个关键帧共展开为 7432 条视觉问答记录：

```text
perception：3502
prediction：2346
planning：1584
behavior：0
总计：7432
```

中间 JSONL：

```text
experiments/drivelm/outputs/town01_controlloss_model_inputs.jsonl
experiments/drivelm/outputs/town01_controlloss_model_inputs_30q.jsonl
```

LLaMA-Adapter 输入：

```text
experiments/drivelm/outputs/town01_controlloss_llama_full.json
experiments/drivelm/outputs/town01_controlloss_llama_30q.json
```

30 条输入用于后续快速冒烟测试，完整实验使用 7432 条输入。

## 推理配置

```text
模型：LLaMA-Adapter V2 BIAS-7B
输入数量：7432
批量大小：1
数据加载进程：2
最大生成长度：256
temperature：0.2
top_p：0.1
保存间隔：25 条
断点续跑：开启
```

批量大小 4 可以正常运行，但由于不同问题的回答长度差异较大，同一批次中的短回答需要等待最长回答生成结束，实际吞吐低于批量大小 1，因此完整实验使用批量大小 1。

## 完整推理结果

```text
完成预测：7432 / 7432
唯一 ID：7432
重复 ID：0
空预测：0
覆盖率：100%
完整生成时间：1 小时 24 分 59 秒
平均吞吐：约 1.46 条 QA/秒
峰值 CUDA 显存：14.02 GiB
```

最终预测：

```text
experiments/drivelm/outputs/drivelm_llama_adapter_predictions_full.json
```

最终指标：

```text
experiments/drivelm/outputs/drivelm_llama_adapter_metrics_full.json
```

完整日志：

```text
experiments/drivelm/outputs/llama_full.log
```

## 本地诊断指标

以下指标用于分析当前 CARLA 子集上的模型表现，不是 DriveLM 官方挑战总分。

原因：

1. CARLA 标注不包含 nuScenes 挑战评测使用的 `tag` 字段。
2. 官方总分包含 GPT 评分，需要单独配置 OpenAI API。
3. 本地 BLEU 使用纯 Python 加一平滑实现，不等同于官方 COCO 工具链。
4. 本次只覆盖 `Town01/ControlLoss`，不能代表完整 DriveLM-CARLA 数据集。

总体指标：

| 指标 | 结果 |
|---|---:|
| 样本数 | 7432 |
| 空预测数 | 0 |
| 严格完全匹配 | 0.0054 |
| 归一化完全匹配 | 0.0054 |
| Token F1 | 0.4761 |
| ROUGE-L | 0.4416 |
| BLEU-1 | 0.3258 |
| BLEU-2 | 0.2586 |
| BLEU-3 | 0.2054 |
| BLEU-4 | 0.1590 |

严格完全匹配数量：

```text
40 / 7432
```

分类指标：

| 类别 | 数量 | Token F1 | ROUGE-L | BLEU-1 | BLEU-4 |
|---|---:|---:|---:|---:|---:|
| perception | 3502 | 0.5491 | 0.5129 | 0.4623 | 0.2628 |
| prediction | 2346 | 0.4327 | 0.3969 | 0.2278 | 0.0976 |
| planning | 1584 | 0.3791 | 0.3500 | 0.2766 | 0.1041 |

类别表现排序：

```text
perception > prediction > planning
```

Token F1 分布：

```text
总体平均值：0.4761
总体中位数：0.5000
10% 分位数：0.2353
90% 分位数：0.6667
```

Token F1 不低于 0.5 的样本数：

```text
perception：2703 / 3502
prediction：949 / 2346
planning：330 / 1584
```

Token F1 低于 0.25 的样本数：

```text
perception：132 / 3502
prediction：408 / 2346
planning：306 / 1584
```

planning 类没有样本达到 Token F1 0.75，说明模型在复杂驾驶决策问题上缺少稳定的高质量输出。

## 结果分析

感知类表现最好。模型能够较好识别车辆颜色、车辆相对位置、车道线颜色、交通灯状态和基础道路结构。

较好样例：

```text
问题：
自车左侧是什么车道线？

标准答案：
自车左侧是黄色虚线。

模型回答：
自车左侧有黄色车道线。
```

该回答主体、方位和颜色基本正确，但遗漏了“虚线”属性。

prediction 类表现明显低于 perception。模型对交通灯状态等短答案问题偶尔可以完全正确，但对车辆运动方向、车道风险和潜在冲突关系，容易生成通用驾驶描述。

完全正确样例：

```text
问题：
交通灯当前是什么状态？

标准答案：
交通灯是红色。

模型回答：
交通灯是红色。
```

prediction 类共有 40 条严格完全匹配，占该类别约 1.71%。全部严格完全匹配样本均来自 prediction 类，主要原因是部分问题答案较短且形式固定。

planning 类表现最弱。此类问题要求模型同时识别道路环境、判断交通参与者是否影响自车路径、推断风险并给出驾驶动作。通用 BIAS-7B 缺少针对 DriveLM 图结构和驾驶规则的训练，容易使用一般交通常识替代当前图片证据。

典型错误：

```text
问题：
自车应该如何根据停车标志行动？

标准答案：
当前没有影响自车的停车标志。

模型回答：
自车应该在停车标志前完全停车，然后再继续行驶。
```

模型直接接受问题中的“停车标志”前提，没有先验证图片中是否真的存在影响自车的停车标志。

## Yes/No 肯定偏置

标准答案以 Yes 或 No 开头的问题共有：

```text
1822 条
```

模型在这 1822 条问题中全部以 Yes 回答。

混淆统计：

```text
标准答案 Yes，模型回答 Yes：333
标准答案 No，模型回答 Yes：1489
标准答案 No，模型回答 No：0
```

总体 Yes/No 准确率：

```text
333 / 1822 = 18.28%
```

分类准确率：

```text
perception：66 / 786 = 8.40%
prediction：0 / 274 = 0%
planning：267 / 762 = 35.04%
```

这说明模型没有可靠地依据图片完成二分类判断，而是存在非常强的肯定回答偏置。

## 输出长度问题

平均答案长度：

| 类别 | 模型平均词数 | 标准答案平均词数 | 长度比例 |
|---|---:|---:|---:|
| 总体 | 20.87 | 11.37 | 1.84 |
| perception | 15.90 | 11.28 | 1.41 |
| prediction | 23.64 | 9.74 | 2.43 |
| planning | 27.78 | 13.97 | 1.99 |

最长模型回答达到：

```text
209 个词
```

模型经常在正确或部分正确的短答案后继续生成泛化描述，导致：

- 引入图片中不存在的目标、天气或道路信息。
- 降低 BLEU 和 ROUGE 指标。
- 掩盖原本可能正确的核心答案。
- 增加推理时间。
- 增加视觉幻觉风险。

## 幻觉问题

典型感知幻觉：

```text
问题：
场景中有哪些重要目标？

标准答案：
场景中没有重要目标。

模型回答：
场景中包括一辆汽车、一栋建筑和一棵树，并继续描述这些目标的位置。
```

模型在标准答案明确表示没有重要目标时，仍主动生成多个物体，说明其容易依赖常见街景模板，而不是严格依据当前图片。

典型预测幻觉：

```text
标准答案：
自车需要注意对向车道的来车。

模型回答：
模型描述了左车道、右车道和中央车道，并生成大量通用驾驶安全建议。
```

该回答语言流畅，但没有准确回答具体需要关注的车道。

## 总体结论

本实验成功完成了 DriveLM 官方 LLaMA-Adapter V2 7B 基线在 `Town01/ControlLoss` CARLA 图文子集上的端到端零样本推理。

工程层面：

- RTX 5090 的 `sm_120` 环境适配成功。
- 274 张图片与官方标注全部对齐。
- 7432 条 QA 全部完成推理。
- 无重复、无缺失、无空预测。
- 峰值显存为 14.02 GiB。
- 断点保存和恢复机制工作正常。

模型性能层面：

- 简单视觉感知能力相对较好。
- 行为预测能力明显弱于感知。
- 复杂驾驶规划能力最弱。
- 模型存在严重的 Yes 肯定偏置。
- 回答普遍过长。
- 模型容易生成无关描述。
- 对不存在的物体、标志和道路条件存在明显幻觉。
- 通用 BIAS-7B 零样本权重不足以可靠解决 DriveLM-CARLA 驾驶问答任务。

本次结果适合作为零样本基线，不应被解释为经过 DriveLM-CARLA 微调后的模型性能。