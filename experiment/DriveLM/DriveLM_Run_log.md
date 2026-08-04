当前状态：DriveLM 最小复现实验已完成。

本文件只记录实际执行过的实验过程、命令、输出和问题。
未实际运行的内容不记录为实验结果。


# 2026-07-10

## 目标

完成 DriveLM baseline 调研，并搭建 LLaMA Adapter V2 Multimodal 7B 实验环境。


## 已完成

阅读并整理：

DriveLM论文：

DriveLM: Driving with Multi-modal Large Language Models

官方仓库：

https://github.com/OpenDriveLab/DriveLM


确认 DriveLM 任务方向：

自动驾驶场景理解

+
多模态大模型

+
语言推理


完成基础调研：

- DriveLM 数据格式
- nuScenes 数据集
- LLaMA Adapter V2
- 多模态输入方式
- evaluation流程


---

# 2026-07-11


## 目标

完成 DriveLM baseline 最小复现实验：

- 配置运行环境
- 准备训练数据
- 微调 LLaMA Adapter V2
- 测试 inference
- evaluation


---

# 环境配置


## 主机


平台：

AutoDL


GPU:

NVIDIA vGPU-48GB


显存：

48GB


CUDA:

12.8



## Python


Python:

3.8



Conda Environment:


llama_adapter_v2



## 当前目录


/root/autodl-tmp/drivelm_work



---

# 代码版本


DriveLM:


https://github.com/OpenDriveLab/DriveLM


使用：

challenge/llama_adapter_v2_multimodal7b


---

# 模型准备


## LLaMA模型


模型：

LLaMA 7B


路径：


llama_model_weights/


## Adapter checkpoint


文件：


BIAS-7B.pth



路径：


checkpoints/BIAS-7B.pth



---

# 数据准备


## 原始数据


文件：

train_llama.json


规模：

377956 samples



## 数据采样


由于完整数据训练时间较长，

分别测试：

5000

10000

50000


数据采样脚本：


split_50000.py



代码：


random.seed(42)

random.sample(data,50000)



最终训练数据：


train_llama_50000.json



---

# 遇到的问题1：LLaMA路径错误


## 错误


FileNotFoundError:


params.json



实际错误路径：


llama_model_weights/7B/params.json



## 原因


finetune.sh 会自动添加：

7B



导致路径重复。



## 处理


错误：


/root/autodl-tmp/drivelm_work/llama_model_weights/7B



修改为：


/root/autodl-tmp/drivelm_work/llama_model_weights



解决：

模型正常加载。



---

# 模型训练


## 执行命令


bash exps/finetune.sh \
/root/autodl-tmp/drivelm_work/llama_model_weights \
/root/autodl-tmp/drivelm_work/checkpoints/BIAS-7B.pth \
finetune_data_config.yaml \
output



## 训练参数


Epoch:

1


Batch size:

1


Learning rate:

blr=10e-4


Weight decay:

0.02



## 训练过程


50000数据训练：


时间：

约4小时



最终输出：


checkpoint-0.pth



大小：


14GB



---

# Inference测试


## 执行


python demo.py \
--llama_dir xxx \
--checkpoint checkpoint-0.pth \
--data test_llama.json \
--output output.json



## 初始结果


生成：

output_50000.json



发现问题：


选择题答案格式不统一。


例如：

模型输出：

A. Back up.


或者：

The ego vehicle is going straight.



官方答案：

B


导致：

accuracy较低。


---

# 答案格式修正


## 问题


DriveLM evaluation 对选择题采用严格匹配。


自然语言答案无法直接匹配。


## 处理


增加：

answer post-processing



生成：


output_50000_fixed_v2.json



---

# Evaluation


## 执行命令


python evaluation.py \
--root_path1 ./output_50000_fixed_v2.json \
--root_path2 ./test_eval.json



## 最终结果



|指标|结果|
|-|-:|
|Accuracy|0.444444|
|Final Score|0.094468|



---

# 实验问题与处理


## 问题1：checkpoint没有生成


原因：

第一次训练路径错误。


解决：

修改llama路径。


---

## 问题2：evaluation accuracy为0


原因：

模型回答正确但是格式不一致。


解决：

增加选择题答案转换。


---

## 问题3：完整数据训练成本过高


原因：

377956数据训练时间较长。


解决：

构造5000、10000、50000子集。


最终采用：

50000训练集。


---

# 当前结果总结


|项目|状态|
|-|-|
|环境配置|✅|
|LLaMA Adapter加载|✅|
|数据转换|✅|
|5000训练测试|✅|
|10000训练测试|✅|
|50000训练测试|✅|
|Inference|✅|
|Evaluation|✅|


---

# 当前结论


已完成 DriveLM baseline 最小复现实验。


当前实验验证：

DriveLM数据格式

+

LLaMA Adapter V2训练流程

+

自动驾驶视觉问答推理流程


由于训练资源限制：

未进行：

- 377956完整训练
- LLaMA全参数训练
- nuScenes重新构建


采用：

50000训练数据

+

LLaMA Adapter V2

+

官方evaluation


完成baseline验证。


下一步：

- 可视化DriveLM回答结果
- 分析不同数据规模影响
- 与CARLA场景结合
