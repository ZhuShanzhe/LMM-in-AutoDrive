# Baseline 调研报告

本目录集中保存组员已经完成的模型调研、最小复现和对照实验。报告用于第一阶段技术选型依据，不参与默认运行时导入。

| 报告 | 负责人 | 数据/环境 | 定位 |
|---|---|---|---|
| [DriveLM-CARLA](DriveLM_DriveLM-CARLA.md) | 朱善哲 | CARLA Town01/ControlLoss | DriveLM 官方 LLaMA-Adapter V2 7B 零样本基线 |
| [DriveLM-nuScenes](DriveLM_nuScenes.md) | 黄皓星 | nuScenes | DriveLM 视觉语言理解与微调调研 |
| [SparseDrive / nuScenes](Sparse_nuScene/README.md) | 刘旭 | nuScenes | 稀疏端到端感知规划基线 |
| [VAD](VAD.md) | 王皓然 | nuScenes-mini | 向量化场景表示与规划基线 |
| [Senna](Senna.md) | 李畅锦 | nuScenes-mini | VLM 高层语义决策 + E2E 轨迹规划 |

## 使用边界

- 报告中的环境、模型许可和数据许可分别以各报告记录为准。
- 不把论文原始指标、组员复现实验指标和本项目当前链路指标混为一类。
- baseline 外部仓库、数据集和模型权重不提交本 Git 仓库。
- 缺少的完整训练日志、原始指标和可视化材料将在最终提交材料阶段继续补齐。
