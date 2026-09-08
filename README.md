# LMM-in-AutoDrive

面向 XH-202602“智能驾驶大模型应用场景研究”的模块化语音控制与多模态决策系统。当前分支复用基础赛道稳定链路，作为挑战赛道模型压缩、J6P适配与联调的开发基线；训练数据、模型权重、检查点和大规模实验输出不进入 Git。

## 挑战赛道规划与分支

挑战赛道沿用现有模块化链路，面向地平线 J6P 进行模型压缩、量化部署和性能验证。
路线规划及五人分工见 [0908 任务规划 PDF](program/task_0908.pdf)。

- `main`：保留基础赛道稳定链路，并收录共享规划文档；不直接进行挑战赛道实验开发。
- `challenge-track`：挑战赛道集成分支，汇总验证后的模块优化。
- `zsz`：朱善哲个人开发分支，负责指令解析、轻量 VLA 和系统集成；通过评审后合并到 `challenge-track`。

其他成员同样将挑战赛道变更合并到 `challenge-track`；基础赛道必要修复可单独同步到 `main`。

服务器工作目录、现有环境与权重复用、基线回归及分支合并方式见 [挑战赛道开发入口](CHALLENGE_DEVELOPMENT.md)。

## 当前实现

系统围绕题目要求的四个核心模块组织：

| 题目模块 | 本仓库实现 | 主要输出 |
|---|---|---|
| 语音解析 | `automatic_speech_recognition` + `structured_command_parser` | ASR 文本、`DrivingIntent 1.2` |
| 视觉理解 | `scene_understanding` | `PerceptionFrame`、`WorldState`、候选实体 |
| 语义对齐 | `scene_understanding` | 实体对齐、TTC、`RiskAssessment` |
| 动作生成 | 原规则链路或 `lightweight_vla_adapter` | `ControlDecision 1.0` |
| 仿真执行 | `experiment/CARLA` | CARLA 控制量、日志与场景结果 |

模块说明及压缩前对照结果见 [文档索引](docs/README.md)。

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
| `docs/` | 挑战赛道文档与现有基线结果索引 |
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

| 模型 | 来源 | 提交包内相对位置 |
|---|---|---|
| ModernBERT 指令解析 | 组合指令微调权重 | `models/modernbert-drive-command-compositional/` |
| YOLO11s 场景检测（可选审核模块） | 驾驶场景检测权重 | `models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt` |
| 三场景通用 VLA V6 sensor policy | 本项目 CARLA + nuScenes 训练；策略端禁用 CARLA actor 真值 | `models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt` |

权重可随 Docker 镜像交付，也可只读挂载到 `models/`。固定哈希、训练数据来源、指标和已知边界见 [三场景通用 VLA 模型卡](lightweight_vla_adapter/UNIVERSAL_THREE_SCENE_MODEL.md)。

## 三场景统一运行

CARLA 服务端启动后，三个场景共用同一模型权重和决策接口：

```bash
source submission_env.sh
bash experiment/CARLA/scripts/run_universal_vla.sh scene1
bash experiment/CARLA/scripts/run_universal_vla.sh scene2
bash experiment/CARLA/scripts/run_universal_vla.sh scene3
```

脚本只使用仓库相对路径，并允许用 `MODEL_ROOT`、`PYTHON_BIN`、`CARLA_HOST`、`CARLA_PORT` 和输出目录参数适配 Docker。测试范围和结果见 [三场景测试报告](program/UNIVERSAL_VLA_THREE_SCENE_TEST_REPORT_20260806.md)。

## 版本与数据边界

当前开发分支保留：

- 运行时源代码和稳定接口；
- 配置、Schema、最小示例与必要的回归入口；
- 环境、下载、运行和接入说明；
- 挑战赛道可复用的训练、数据处理和评测工具；
- 当前阶段结果摘要和已知边界。

Git 不包含：

- 数据集原文件和生成语料；
- 模型权重、检查点和 Hugging Face 缓存；
- 大规模逐样本预测、图片帧、视频、日志和临时输出。

已结束的独立baseline调研、DriveLM实验和旧第一阶段材料索引已从挑战赛道开发分支移除，仍可在 `main` 和Git历史中查阅。既有路线规划PDF、模块代码、训练工具和本系统基线测试结果保持保留。新生成的数据和输出统一受 `.gitignore` 管理。

## 当前边界

- 当前仅完成挑战赛道开发基线同步和目录整理，尚未完成挑战赛道模型压缩或J6P适配。
- 基础赛道三场景测试结果作为历史对照保留，不能当作J6P部署或压缩后性能。
- CARLA 和离线指标只代表对应测试范围，不能解释为真实道路安全认证。
- 各模型必须遵守 Hugging Face 模型卡和上游数据集许可证。
