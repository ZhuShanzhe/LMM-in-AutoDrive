# XH-202602 智能驾驶多模态闭环系统

本目录是基础赛道两场景审核版，包含基础语音操控 5 km 和多模态闭环 8 km 的代码、权重、配置、代表性日志与提交文档。场景三代码和设计配置暂时保留，但不纳入本版结果或达标结论。

## 系统链路

```text
语音指令
  -> ASR、降噪与术语约束翻译
  -> ModernBERT / DrivingIntent 1.2
  -> 实时感知、场景语义融合与风险评估
  -> 规则 FSM 或 VLA + FSM
  -> ControlDecision 1.0
  -> 路线适配、PID 与 CARLA VehicleControl
  -> 事件、指标与视频记录
```

## 提交目录

| 路径 | 作用 |
|---|---|
| `automatic_speech_recognition/` | 语音识别、降噪、翻译和测试入口 |
| `structured_command_parser/` | 将文本指令解析为 `DrivingIntent 1.2` |
| `scene_understanding/` | 实时感知、实体对齐、风险评估和高层动作 |
| `lightweight_vla_adapter/` | 轻量 VLA 训练、推理、评测与安全门 |
| `experiment/CARLA/` | 场景、交通流、闭环控制、日志与可视化 |
| `models/` | 上游预训练权重、团队最终权重、许可证与校验清单 |
| `基础赛道提交材料/` | 题目要求的六类文档、报名表和两场景佐证 |

内部计划、题目原文件、基线调研、Git 历史、缓存和原始训练数据不进入本审核目录。

## 当前场景

| 场景 | 入口 | 配置 | 实测材料 |
|---|---|---|---|
| 场景一：基础语音操控 5 km | `experiment/CARLA/run_control_experiment.py` | `experiment/CARLA/configs/basic_voice_urban_5km.json` | `基础赛道提交材料/06_仿真测试全量报告/原始时序数据/scene_1_basic/` |
| 场景二：8 km 多模态闭环 | `experiment/CARLA/run_scene2_closed_loop.py` | `experiment/CARLA/configs/scene_2_submission_8_runtime.json` | `基础赛道提交材料/06_仿真测试全量报告/原始时序数据/scene_2_complex/` |

场景二随附运行记录是 8 条安全闭环指令版本，用于验证感知、VLA、FSM 和控制链路。完整复杂障碍事件配置仍保留在 `experiment/CARLA/configs/scene_2_town05_runtime.json`，本版不把简化运行结果表述为完整复杂避障验收。

## 快速检查

```bash
source submission_env.sh
sha256sum -c models/SHA256SUMS

python -m pytest -q \
  structured_command_parser/tests \
  scene_understanding/tests \
  lightweight_vla_adapter/tests \
  experiment/CARLA/tests
```

服务器审核结果：模型文件校验全部通过；上述四部分共 `455 passed，177 subtests passed`。

## 视频放置

Windows 审核目录预留：

```text
基础赛道提交材料/06_仿真测试全量报告/可视化材料/
├── scene_1_basic.mp4
└── scene_2_complex.mp4
```

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
- 大规模逐样本预测、图片帧、视频、日志和临时输出。

VLA 训练、数据构建、离线评测和三场景运行代码已进入 `main`；数据集原文件和权重按许可证及体积要求单独交付。新生成的数据和输出统一受 `.gitignore` 管理。

## 当前边界

- 当前仓库是基础赛道三场景统一 VLA 的提交准备版本；权重和 Docker 镜像仍需在最终交付时配套。
- 场景三 V6 传感器策略 6 km 正式视频正在生成；场景一 5 km、场景二 8 km 的最终权重全程证据仍需按统一脚本留档。
- CARLA 和离线指标只代表对应测试范围，不能解释为真实道路安全认证。
- 各模型必须遵守 Hugging Face 模型卡和上游数据集许可证。
