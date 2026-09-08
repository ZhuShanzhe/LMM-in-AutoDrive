# DriveLM Dataset

本目录用于说明 DriveLM 实验使用的数据集。

由于 DriveLM-nuScenes 数据集包含大量图像数据以及标注文件，
完整数据集未上传至 GitHub。

## Dataset Information

原始训练数据：

- train_llama.json
- 样本数量：377956

实验过程中，根据 GPU 显存和训练时间限制，
构造不同规模训练子集。

| Dataset | Samples | Purpose |
|---|---:|---|
| train_llama_5000 | 5000 | 快速验证训练流程 |
| train_llama_10000 | 10000 | 小规模实验 |
| train_llama_50000 | 50000 | 最终模型训练 |

## Data Processing

数据处理流程：

1. 下载 DriveLM-nuScenes 数据；
2. 转换为 LLaMA Adapter 输入格式；
3. 根据随机种子进行采样；
4. 使用采样后的数据进行模型微调。

随机采样：

```python
random.seed(42)
random.sample(data, 50000)
