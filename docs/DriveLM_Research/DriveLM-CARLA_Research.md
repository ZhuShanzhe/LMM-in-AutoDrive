# DriveLM（Drive Language Model）模型研究

负责人：黄皓星

当前状态：
DriveLM baseline 调研完成；
DriveLM-nuScenes 数据处理完成；
LLaMA Adapter V2 Multimodal 7B 模型微调实验完成；
50000条训练数据实验完成；
模型推理结果生成完成。


---

# 1. 调研目标


本调研面向“智能驾驶大模型应用”方向，
目标是研究 DriveLM 作为自动驾驶多模态大模型（Multimodal Large Language Model, MLLM）的应用价值。


本阶段重点分析：

- DriveLM 的整体架构设计；
- 自动驾驶场景中的视觉语言理解方法；
- nuScenes 数据集组织形式；
- 大语言模型在驾驶决策中的作用；
- DriveLM 与 CARLA 仿真环境结合的可能性；
- 基于 DriveLM 的智能驾驶语义决策方案。


---

# 2. 基本信息


| 项目 | 内容 |
|----|----|
| 模型名称 | DriveLM |
| 全称 | Drive Language Model |
| 任务类型 | 自动驾驶视觉语言理解 |
| 数据来源 | nuScenes |
| 模型类型 | Vision Language Model |
| Backbone | LLaMA Adapter V2 Multimodal |
| 输入 | 多视角驾驶图像 + 文本问题 |
| 输出 | 驾驶场景理解与决策回答 |
| 主要方向 | 自动驾驶大模型推理 |


官方项目：

https://github.com/OpenDriveLab/DriveLM



---

# 3. DriveLM 背景介绍


传统自动驾驶系统通常采用模块化结构：



Camera/LiDAR

↓

Perception

↓

Prediction

↓

Planning

↓

Control



该方法存在：

- 各模块独立优化；
- 信息传递存在损失；
- 难以理解复杂驾驶语义；
- 缺少高级推理能力。


随着大语言模型的发展，
DriveLM 提出利用视觉语言模型能力，
让模型不仅能够识别环境，
还能够理解驾驶场景并进行推理。


DriveLM 将自动驾驶任务转换为：


驾驶场景

自然语言问题

↓

大模型推理

↓

驾驶决策解释



使自动驾驶系统具备：

- 场景理解能力；
- 逻辑推理能力；
- 语义解释能力。


---

# 4. DriveLM 核心思想


DriveLM 的核心思想：

利用 Large Language Model 作为驾驶场景理解中心。


模型输入包括：


## 视觉信息


例如：

- 摄像头图像；
- 周围车辆；
- 行人；
- 道路环境。


## 语言信息


例如：



What is the ego vehicle's next action?

Why should the vehicle slow down?



模型输出：



The vehicle should slow down because
a pedestrian is crossing the road.



因此 DriveLM 不仅输出结果，
还能给出决策依据。


---

# 5. DriveLM 模型结构分析


整体结构：



Camera Images

  |

Vision Encoder

  |

Visual Feature

  |

LLaMA Adapter

  |

Large Language Model

  |

Text Response



主要包含三个部分：


## 5.1 Vision Encoder


负责：

- 图像特征提取；
- 场景信息编码。


输入：

多视角驾驶图片。


输出：

视觉 embedding。


---

## 5.2 Adapter Module


作用：

连接视觉模型和语言模型。


由于直接训练大型语言模型成本较高，
Adapter 通过参数高效方式完成视觉信息融合。


优势：

- 参数量小；
- 训练成本低；
- 保留 LLaMA语言能力。


---

## 5.3 LLaMA Language Model


负责：

- 场景推理；
- 问题理解；
- 生成驾驶相关回答。


例如：

输入：


Describe the danger in this scene.



输出：


A pedestrian is close to the road,
so the vehicle should slow down.




---

# 6. DriveLM 数据格式


DriveLM 基于 nuScenes 数据集。


主要包含：


## 图像数据


六个摄像头：



CAM_FRONT

CAM_BACK

CAM_LEFT

CAM_RIGHT

CAM_FRONT_LEFT

CAM_FRONT_RIGHT



---

## 标注数据


包括：


- agent 信息；
- 车辆位置；
- 交通参与者；
- 驾驶场景描述；
- 问答数据。


数据形式：

```json
{
"question":
"What should ego vehicle do?",

"answer":
"Slow down because pedestrian ahead."
}
7. DriveLM 与传统自动驾驶模型区别
模型	主要能力
BEVFormer	三维感知
VAD	预测与规划
UniAD	端到端驾驶任务
DriveLM	驾驶场景理解与推理

DriveLM 重点不是替代感知模型，
而是提供：

场景理解

+

语义推理

+

决策解释
8. 本次实验环境
硬件环境
项目	配置
GPU	NVIDIA RTX 4090
显存	24GB
平台	AutoDL 云服务器
软件环境
项目	版本
OS	Ubuntu
Python	3.10
CUDA	12.x
PyTorch	2.x
Transformers	相关版本
9. 实验方案

本实验采用：

LLaMA Adapter V2 Multimodal 7B

训练流程：

DriveLM Dataset

        ↓

50000 samples sampling

        ↓

Model Fine-tuning

        ↓

Checkpoint

        ↓

Inference

        ↓

Evaluation
10. 数据处理实验

原始训练数据：

train_llama.json

由于完整数据规模较大，
采用50000条数据进行实验。

处理脚本：

split_50000.py

生成：

train_llama_50000.json

采用固定随机种子：

seed=42

保证实验可复现。

11. 模型训练

训练模型：

LLaMA Adapter V2 Multimodal 7B

训练方式：

参数高效微调。

训练数据：

50000 samples

输出：

checkpoint

由于模型权重较大，
不上传 GitHub。

12. 推理实验

最终生成：

output_50000_fixed_v2.json

该文件保存模型在测试集上的预测结果。

同时进行了：

答案格式修正；
evaluation测试。
13. 实验结果

最终结果：

Metric	Score
Accuracy	0.444444
Final Score	0.094468

实验说明：

模型能够完成自动驾驶场景问答任务。

14. DriveLM 与 CARLA 结合分析

CARLA 是自动驾驶仿真平台。

DriveLM 可以作为：

CARLA Simulator

↓

Camera Sensor

↓

DriveLM

↓

High-level Decision

↓

Vehicle Controller

其中：

CARLA负责：

环境模拟；
车辆运动；
传感器生成。

DriveLM负责：

场景理解；
高层驾驶决策；
解释原因。
15. 项目应用价值

结合 CARLA 后，可以实现：

场景理解

例如：

前方存在行人
风险分析

例如：

行人可能进入车辆路径
决策生成

例如：

降低速度并等待

形成：

感知

+

语言理解

+

决策

的智能驾驶框架。

16. 当前不足
1. 模型无法直接控制车辆

DriveLM 输出：

文本决策

不是：

steering
throttle
brake

因此需要：

PID controller
MPC controller

完成车辆控制。

2. 训练成本较高

完整训练需要：

大规模GPU；
大规模数据。

当前采用：

50000 samples

进行验证。

3. 仿真迁移问题

nuScenes 与 CARLA 数据存在差异：

需要：

sensor adaptation；
数据格式转换；
prompt设计。
17. 对本项目的启发

DriveLM 可以作为智能驾驶系统中的：

视觉理解模块

+

语义决策模块

结合：

CARLA

自动驾驶感知模型

DriveLM

形成：

环境感知

↓

场景理解

↓

LLM决策

↓

车辆控制
18. 总结

本实验完成了 DriveLM baseline 调研以及最小复现实验。

主要完成：

DriveLM模型分析；
数据处理；
50000样本训练；
模型推理；
evaluation测试。

实验验证：

视觉语言大模型能够应用于自动驾驶场景理解任务。

未来方向：

DriveLM + CARLA；
自动驾驶智能体；
LLM辅助决策；
闭环驾驶控制。
19. 参考资料

DriveLM:

https://github.com/OpenDriveLab/DriveLM

LLaMA Adapter:

https://github.com/OpenGVLab/LLaMA-Adapter

nuScenes:

https://www.nuscenes.org/

CARLA Simulator:

https://carla.org/
