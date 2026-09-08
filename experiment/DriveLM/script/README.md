# DriveLM Experiment Scripts


本目录保存 DriveLM baseline 复现实验过程中使用的辅助脚本。

主要用于：

- 数据集采样
- 实验数据构造
- 训练配置准备
- 模型结果处理

本目录中的脚本均来源于实际实验过程。


---

# 1. Dataset Processing Scripts


## split_50000.py


### 功能

从完整 DriveLM 训练数据中随机采样 50000 条数据，
用于 LLaMA Adapter V2 Multimodal 7B 模型微调实验。


### 输入文件


data/train_llama.json



原始训练数据规模：


377956 samples



### 输出文件


data/train_llama_50000.json



### 主要代码逻辑

```python
random.seed(42)

small = random.sample(data,50000)
说明

使用固定随机种子保证数据采样过程可复现。

split_10000.py
功能

构造 10000 条训练数据子集，
用于快速验证模型训练流程。

输入文件
data/train_llama.json
输出文件
data/train_llama_10000.json
用途

用于测试：

数据读取是否正常
模型训练流程是否正常
小规模训练效果
make_balanced_5000.py
功能

构造 5000 条平衡训练数据。

由于 DriveLM 数据集中不同任务类型数量存在差异，
因此根据任务类别进行采样。

主要任务类型包括：

perception
prediction
planning
behavior
输出文件
train_llama_5000_balance.json
用途

用于测试小规模数据训练效果。

2. Training Configuration
finetune_data_config.yaml
功能

指定 LLaMA Adapter V2 微调过程中使用的数据路径。

示例：

META:
  - '/root/autodl-tmp/drivelm_work/data/train_llama_50000.json'

训练程序：

main_finetune.py

通过该配置文件读取训练数据。

3. Model Training Script
exps/finetune.sh
功能

启动 LLaMA Adapter V2 Multimodal 7B 微调训练。

执行格式
bash exps/finetune.sh \
llama_model_path \
checkpoint_path \
data_config \
output_dir
本实验执行命令
bash exps/finetune.sh \
/root/autodl-tmp/drivelm_work/llama_model_weights \
/root/autodl-tmp/drivelm_work/checkpoints/BIAS-7B.pth \
finetune_data_config.yaml \
output
输出

训练完成后生成：

checkpoint-0.pth

模型checkpoint大小：

约14GB

由于文件较大，没有上传 GitHub。

4. Model Inference Script
inference / demo script
功能

加载训练完成后的 Adapter checkpoint，
对测试数据进行推理。

输入：

测试图像
自动驾驶问题

输出：

模型预测结果：

output.json
5. Evaluation Script
evaluation.py
功能

使用 DriveLM 官方 evaluation 工具，
计算模型性能。

输入

模型预测结果：

output_50000_fixed_v2.json

官方测试答案：

test_eval.json
执行命令
python evaluation.py \
--root_path1 output_50000_fixed_v2.json \
--root_path2 test_eval.json
输出指标

包括：

Accuracy
language score
Final Score
6. Answer Post-processing Script
Answer Fixing
背景

DriveLM evaluation 对部分选择题采用严格答案匹配。

训练模型生成结果可能存在格式差异。

例如：

模型输出：

The ego vehicle is not moving.

官方答案：

D

虽然语义一致，
但字符串不同会导致 evaluation 失败。

处理方法

增加答案后处理步骤：

判断问题是否为选择题；
提取模型回答中的选项信息；
转换为官方答案格式。

最终生成：

output_50000_fixed_v2.json
7. Complete Experiment Pipeline

完整实验流程：

train_llama.json

        |

        ↓

split_50000.py

        |

        ↓

train_llama_50000.json

        |

        ↓

LLaMA Adapter V2 Training

        |

        ↓

checkpoint-0.pth

        |

        ↓

Inference

        |

        ↓

output.json

        |

        ↓

Answer Post-processing

        |

        ↓

output_50000_fixed_v2.json

        |

        ↓

evaluation.py

        |

        ↓

Final Score
