# LMM-in-AutoDrive

> 2026-08-03：场景 2/3 现实化与 VLA-first 决策协调器更新见
> [docs/SCENE_DECISION_UPDATE_2026-08-03.md](docs/SCENE_DECISION_UPDATE_2026-08-03.md)。

面向 XH-202602“智能驾驶大模型应用场景研究”基础赛道的模块化语音控制与多模态决策系统。本仓库 `main` 保存第一阶段可集成运行代码、稳定接口、环境与模型下载说明，以及已完成的技术调研；训练数据、模型权重、检查点和大规模实验输出不进入 Git。

## 第一阶段目标

系统围绕题目要求的四个核心模块组织：

| 题目模块 | 本仓库实现 | 主要输出 |
|---|---|---|
| 语音解析 | `automatic_speech_recognition` + `structured_command_parser` | ASR 文本、`DrivingIntent 1.2` |
| 视觉理解 | `scene_understanding` | `PerceptionFrame`、`WorldState`、候选实体 |
| 语义对齐 | `scene_understanding` | 实体对齐、TTC、`RiskAssessment` |
| 动作生成 | 原规则链路或 `lightweight_vla_adapter` | `ControlDecision 1.0` |
| 仿真执行 | `experiment/CARLA` | CARLA 控制量、日志与场景结果 |

第一阶段材料索引、当前完成项和待补报告见 [docs/phase1_submission/README.md](docs/phase1_submission/README.md)。

## 双链路架构

仓库同时保留原规则链路和轻量 VLA 新链路。两条链路共享结构化指令、场景理解、风险判断、FSM 和 CARLA 控制协议。

```text
语音 / 文本
    |
    v
ASR / 翻译
    |
    v
structured_command_parser -> DrivingIntent 1.2
    |
    v
scene_understanding -> WorldState + semantic alignment + RiskAssessment
    |
    +---------------------- 原规则链路 ----------------------+
    |  canonical high-level action -> ControlPlan FSM         |
    |                                                         |
    +---------------------- VLA 新链路 -----------------------+
       camera BEV + LiDAR BEV + entities + ego + intent token
                         |
                         v
       lightweight_vla_adapter -> VLADecisionProposal 1.0
                         |
                         v
       deterministic safety gate + canonical fallback
                         |
                         v
                    ControlPlan FSM
    |
    v
ControlDecision 1.0 -> CARLA protocol -> controller
```

原规则链路是默认兜底：VLA 权重未安装、输入不完整、推理失败、置信度不足或安全门拒绝时，系统继续使用 canonical rule decision。VLA 不能绕过紧急制动、目标车道安全、语义对齐失败或 FSM 阻塞处理。

## 目录

| 路径 | 内容 |
|---|---|
| `automatic_speech_recognition/` | 中文语音识别、降噪和可选翻译 |
| `structured_command_parser/` | 英文指令规范化、组合意图解析和 JSON Schema |
| `scene_understanding/` | 实时感知、视觉语义、实体对齐、风险和原规则决策 |
| `lightweight_vla_adapter/` | 可选多模态高层决策适配器 |
| `experiment/CARLA/` | Linux CARLA 0.9.16 场景、控制和评估 |
| `docs/baseline_research/` | DriveLM、SparseDrive、VAD、Senna 调研报告 |
| `docs/phase1_submission/` | 第一阶段提交材料索引与缺口 |
| `program/` | 题目、计划和任务文档 |

## 运行环境

统一集成环境：

```text
Linux / Ubuntu 22.04
Python 3.12.13
CARLA 0.9.16
NVIDIA Driver 580.105.08
CUDA 13.0
RTX 5090 / sm_120
```

不同模块的依赖存在差异，不建议在仓库根目录一次性安装所有依赖。按照各模块 README 创建或复用环境：

- [语音模块](automatic_speech_recognition/README.md)
- [指令解析](structured_command_parser/README.md)
- [场景理解](scene_understanding/README.md)
- [轻量 VLA](lightweight_vla_adapter/README.md)
- [CARLA 仿真](experiment/CARLA/README.md)

## 模型权重

权重托管在 Hugging Face 或模块 README 指定的上游仓库，不提交 Git。

| 模型 | 仓库 | 推荐本地位置 |
|---|---|---|
| ModernBERT 指令解析 | `UNIC0RN-Zhu/modernbert-drive-command-base` | `/root/autodl-tmp/models/modernbert-drive-command-compositional` |
| YOLO11s 场景检测 | `UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1` | `/root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt` |
| 轻量 VLA v10 | `UNIC0RN-Zhu/lightweight-vla-drive-decision-adapter-v10` | `/root/autodl-tmp/models/lightweight_vla_adapter/v10/model.pt` |

VLA 仓库采用自动批准访问门控，首次下载前需要 `hf auth login` 并在模型页面申请访问。完整命令、校验和与许可证见各模块 README。

## main 提交边界

`main` 包含：

- 运行时源代码和稳定接口；
- 配置、Schema、最小示例与必要的回归入口；
- 环境、下载、运行和接入说明；
- 已完成 baseline 调研报告；
- 当前阶段结果摘要和已知边界。

`main` 不包含：

- 数据集原文件和生成语料；
- 模型权重、检查点和 Hugging Face 缓存；
- VLA 训练、蒸馏、教师、数据构建和批量评测脚本；
- 大规模逐样本预测、图片帧、视频、日志和临时输出。

VLA 完整训练与实验代码继续保留在 `zsz` 分支，待最终复现包审核后再按题目要求单独整理。其他模块已有的复现代码暂不重写；新生成的数据和输出统一受 `.gitignore` 管理。

## 当前边界

- 当前仓库是第一阶段集成版本，不等同于最终比赛压缩包。
- 完整技术方案、三类场景全量时序报告、原始指标数据和演示视频仍需继续补齐。
- CARLA 和离线指标只代表对应测试范围，不能解释为真实道路安全认证。
- 各模型必须遵守 Hugging Face 模型卡和上游数据集许可证。
