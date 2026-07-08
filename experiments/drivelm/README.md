# DriveLM Reproduction Notes

当前状态：仅完成官方资料整理；复现实验尚未开始。

本目录用于记录 DriveLM / DriveLM-CARLA baseline 的最小复现实验。实验内容应在实际运行后再补充，不提前写结果。

## 1. 实验目标

本实验目录服务于朱善哲负责的 DriveLM / DriveLM-CARLA baseline 调研任务，目标是：

- 记录官方 DriveLM 代码和文档中的复现路径。
- 检查 AutoDL 环境是否满足最小运行条件。
- 优先跑通 demo data 的数据转换、推理或评估流程。
- 判断 DriveLM 是否值得作为本项目正式 baseline 或模块设计参考。

## 2. 官方资料

- 官方仓库：https://github.com/OpenDriveLab/DriveLM
- 论文：https://arxiv.org/abs/2312.14150
- 项目页：https://opendrivelab.com/DriveLM/
- Challenge README：https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md
- GVQA / data details：https://github.com/OpenDriveLab/DriveLM/blob/main/docs/gvqa.md

## 3. 计划实验步骤

以下步骤是根据官方 challenge 文档整理的计划，不代表已经运行成功。

### 3.1 环境检查

待记录：

```bash
nvidia-smi
nvcc --version
python --version
conda --version
df -h
```

### 3.2 获取官方代码

待执行或记录：

```bash
git clone https://github.com/OpenDriveLab/DriveLM.git
```

建议将官方代码放在 `experiments/drivelm/external/DriveLM` 或 AutoDL 的临时实验目录中，避免污染本项目主目录。

### 3.3 创建环境

官方 challenge README 使用 Python 3.8 和 llama-adapter v2 相关依赖。具体 conda / pip 命令需以官方仓库当前文件为准。

待记录：

- conda 环境名：
- Python 版本：
- PyTorch 版本：
- CUDA 版本：
- 安装是否成功：
- 遇到的问题：

### 3.4 数据准备

优先使用官方 demo data，暂不直接下载完整数据集。

待记录：

- demo data 下载地址：
- 数据存放路径：
- 解压后目录结构：
- 数据转换命令：

### 3.5 数据转换与格式检查

官方流程包含数据抽取、格式转换和 LLaMA 训练格式转换。实际命令需以当前官方仓库脚本为准。

待记录：

- 输入数据路径：
- 输出数据路径：
- 转换脚本：
- 是否成功：
- 输出文件示例：

### 3.6 推理 / 评估

如果当前 AutoDL 显存和模型权重满足要求，再尝试运行 inference 或 evaluation。

待记录：

- 使用模型 / checkpoint：
- 输入样例：
- 输出样例：
- 官方评估结果：
- 推理耗时：
- 峰值显存：

## 4. 记录规范

每次运行都应写入 `run_log.md`，包括：

- 时间。
- 当前目录。
- git commit 或代码版本。
- 环境信息。
- 执行命令。
- 关键输出。
- 错误信息。
- 是否解决。
- 下一步动作。

不要只记录“成功 / 失败”，要保留可复查的命令和路径。

## 5. 与主项目的连接点

复现实验完成后，需要回答以下问题：

- DriveLM 输出能否转换成本项目需要的结构化驾驶意图？
- DriveLM 是否能辅助语义对齐和风险判断？
- DriveLM-CARLA 的场景组织方式是否能迁移到本项目 CARLA 基础赛道？
- 复现成本是否适合作为正式 baseline？
- 是否只作为报告中的对比方法和设计启发？

## 6. 当前结论

尚未开始复现实验，暂无实验结论。
