# LMM-in-AutoDrive

面向 XH-202602“智能驾驶大模型应用场景研究”的模块化语音控制与多模态决策系统。`main` 保存稳定运行链路、模块接口、环境说明、既有测试结果和baseline调研；挑战赛道优化在独立分支开展。权重、数据集和大规模运行输出不进入 Git。

## 挑战赛道规划与分支

挑战赛道沿用现有模块化链路，面向地平线 J6P 进行模型压缩、量化部署和性能验证。
挑战赛道五人职责及达标要求见 [0911 职责与达标要求 PDF](program/task_0911.pdf)。

- `main`：保留基础赛道稳定链路，并收录共享规划文档；不直接进行挑战赛道实验开发。
- `challenge-track`：挑战赛道集成分支，汇总验证后的模块优化。
- `zsz`：朱善哲个人开发分支，负责指令解析、轻量 VLA 和系统集成；通过评审后合并到 `challenge-track`。

其他成员同样将挑战赛道变更合并到 `challenge-track`；基础赛道必要修复可单独同步到 `main`。

## 系统模块

系统围绕题目要求的四个核心模块组织：

| 题目模块 | 本仓库实现 | 主要输出 |
|---|---|---|
| 语音解析 | `automatic_speech_recognition` + `structured_command_parser` | ASR 文本、`DrivingIntent 1.2` |
| 视觉理解 | `scene_understanding` | `PerceptionFrame`、`WorldState`、候选实体 |
| 语义对齐 | `scene_understanding` | 实体对齐、TTC、`RiskAssessment` |
| 动作生成 | 原规则链路或 `lightweight_vla_adapter` | `ControlDecision 1.0` |
| 仿真执行 | `experiment/CARLA` | CARLA 控制量、日志与场景结果 |

模块说明、调研及系统测试记录统一从 [文档索引](docs/README.md) 查阅。

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
| `docs/` | 调研与系统测试文档索引 |
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

权重不提交Git。目录、挂载与校验方式统一见 [模型说明](models/README.md)，下载入口见各模块README。VLA固定哈希、数据来源、指标和边界见 [三场景通用 VLA 模型卡](lightweight_vla_adapter/UNIVERSAL_THREE_SCENE_MODEL.md)，避免在多个入口重复维护。

## 三场景统一运行

CARLA 服务端启动后，三个场景共用同一模型权重和决策接口：

```bash
source submission_env.sh
bash experiment/CARLA/scripts/run_universal_vla.sh scene1
bash experiment/CARLA/scripts/run_universal_vla.sh scene2
bash experiment/CARLA/scripts/run_universal_vla.sh scene3
```

脚本只使用仓库相对路径，并允许用 `MODEL_ROOT`、`PYTHON_BIN`、`CARLA_HOST`、`CARLA_PORT` 和输出目录参数适配 Docker。测试范围和结果见 [三场景测试报告](program/UNIVERSAL_VLA_THREE_SCENE_TEST_REPORT_20260806.md)。

## 仓库边界

`main` 包含：

- 运行时源代码和稳定接口；
- 配置、Schema、最小示例与必要的回归入口；
- 环境、下载、运行和接入说明；
- 已完成 baseline 调研报告；
- 当前阶段结果摘要和已知边界。

`main` 不包含：

- 数据集原文件和生成语料；
- 模型权重、检查点和 Hugging Face 缓存；
- 大规模逐样本预测、图片帧、视频、日志和临时输出。

VLA训练、数据构建、离线评测和三场景运行代码按实际目录保留；数据集原文件和权重独立存储，新生成的数据和输出统一受 `.gitignore` 管理。

## 当前边界

- main保留稳定运行代码与既有研究结果；旧基础赛道提交清单和待补提交事项已移除。
- [三场景历史测试结果](program/FINAL_SUBMISSION_TEST_REPORT_20260809.md)保留原文件名与实测指标，用于后续对照，不是待完成的提交清单。
- `submission_env.sh`沿用历史名称，但仍被三场景运行入口调用，保留作为运行环境脚本。
- 既有服务器环境与历史指标不代表已完成J6P适配或压缩后性能验证。
- CARLA 和离线指标只代表对应测试范围，不能解释为真实道路安全认证。
- 各模型必须遵守 Hugging Face 模型卡和上游数据集许可证。
