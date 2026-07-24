# Scene Understanding

场景理解模块负责把结构化驾驶指令与 CARLA 世界状态对齐，生成确定性的风险判断、控制决策和多步骤计划状态。模块通过 JSON 文件与指令解析模块和车辆控制模块联调，不直接依赖其他成员的 Python 包内部实现。

## 运行环境

- CARLA：0.9.16
- Python：3.12.13
- 操作系统：Linux（当前验证系统为 Ubuntu 22.04）
- CARLA Python API：必须与服务端保持 0.9.16
- 闭环实验需要正在运行的 CARLA 服务端和可用的团队控制模块
- JSON 对齐、风险判断和单元测试不要求启动 CARLA 服务端

统一使用数据盘环境：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
python --version
python -c "from importlib.metadata import version; print(version('carla'))"
```

预期输出分别为 `Python 3.12.13` 和 `0.9.16`。CARLA Linux 服务端的安装、
无窗口启动及图形依赖配置见 `experiment/CARLA/README.md`。

## 模块架构

```text
前视图像 ──> YOLOP + YOLO11s ──> ByteTrack ──> PerceptionFrame
                    │                                │
CARLA Actor/Map ────┼──> WorldState ────────────────┤
                    │                                v
关键帧 ──> Qwen/MiniCPM-V（异步可选）──> 视觉语义融合
                                                     │
DrivingIntent ──> 目标语义对齐 ──> 风险/TTC ──> 安全门控决策
                                                     │
                                                     v
                                            control_decision.json
```

模块遵循“同步安全链路优先、异步语义链路不阻塞”的原则：

- 同步链路由检测器、跟踪器、CARLA 真值、地图信息和确定性风险规则组成；
- 异步视觉大模型只补充场景摘要和开放语义，超时、非法或过期结果直接丢弃；
- 碰撞、TTC、交通灯状态和变道安全不能仅依赖视觉大模型文本；
- 输出通过稳定 JSON Schema 与指令解析、决策和车辆控制模块连接。

## 目录结构

```text
scene_understanding/
├── realtime_perception/ # 同步检测、ByteTrack、车道/可行驶区域和评测
├── training/            # BDD100K + nuScenes 专项检测器数据构建、训练和评测
├── async_semantics/     # Qwen/MiniCPM-V 等低频语义增强后端
├── core/       # WorldState、CARLA 采集、传感器、风险与视觉结果处理
├── src/        # 指令对齐、控制决策和多步骤计划执行器
├── scripts/    # JSON 联调命令与 CARLA 闭环实验入口
├── schemas/    # JSON Schema 和示例
├── prompts/    # 当前提示词与历史提示词归档
├── tests/      # 模块全部测试
└── *.md        # 各接口与闭环实验说明
```

## 双链路场景感知

场景感知按安全时限拆成两条互不阻塞的路径：

1. 同步实时路径：YOLOP 负责车辆、车道线和可行驶区域，ByteTrack 维护轨迹；
   CARLA/nuScenes/Waymo 的地图或标注负责车道存在性、变道合法性、路口、人行横道和
   停止线。该路径向 TTC、风险规则和控制门控供数。
2. 异步语义路径：Qwen2.5-VL 或 MiniCPM-V 只解释低频关键帧，补充场景摘要、目标
   描述、施工区域和地标等开放语义。结果超时、非法或过期时直接丢弃，控制循环不等待。

`schemas/perception_frame.schema.json` 是实时输出契约。对象范围包括车辆、行人、骑行者、
摩托车、交通灯、交通标志、锥桶、护栏和障碍物；道路结构包括主车道、左右相邻车道、
车道线、可行驶区域、路口、人行横道、停止线、路缘和停车区域。完整来源矩阵和部署边界
见 `PERCEPTION_ARCHITECTURE.md`。

变道安全必须同时满足：地图确认相邻驾驶车道存在、地图允许该方向变道、轨迹和度量
风险模块确认目标车道动态安全。视觉车道线或 VLM 文本不能单独把
`LEFT_LANE_SAFE/RIGHT_LANE_SAFE/TARGET_LANE_SAFE` 置为真。

安装实时依赖：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
python -m pip install --index-url https://pypi.org/simple \
  -r scene_understanding/requirements-realtime.txt
```

### 下载自训练 YOLO11s 权重

当前默认目标检测权重托管于 Hugging Face：

```text
https://huggingface.co/UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1
```

安装 Hugging Face CLI，并把权重下载到模块约定的位置：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
python -m pip install --index-url https://pypi.org/simple -U huggingface_hub

# AutoDL 可选网络加速；其他 Linux 环境会自动跳过。
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi

MODEL_DIR=/root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1
mkdir -p "$MODEL_DIR/weights"

hf download UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1 \
  best.pt \
  --local-dir "$MODEL_DIR/weights"
```

若仓库设置为私有或受限访问，先执行 `hf auth login`，再运行上述下载命令。
训练参数和验证结果不是推理必需文件，可按需下载：

```bash
hf download UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1 \
  args.yaml validation_metrics.json results.csv \
  --local-dir "$MODEL_DIR"
```

校验本次评测所用权重：

```bash
echo "a96c29ef518990f410f54ec2c4a4ef617b2b184996c2b980eb44072243070c44  $MODEL_DIR/weights/best.pt" \
  | sha256sum -c -
```

预期输出为 `best.pt: OK`。随后执行加载冒烟测试：

```bash
python -c "from ultralytics import YOLO; YOLO('$MODEL_DIR/weights/best.pt'); print('model load: OK')"
```

模型基于 Ultralytics YOLO11s，按 AGPL-3.0 发布；使用者还需遵守 BDD100K、
nuScenes 和 CARLA 数据集各自的许可与使用条款。权重不提交 Git。

CARLA 实时默认后端为 YOLOP 640 + CARLA 域适配 YOLO11s 640 + ByteTrack。
YOLOP 输出车道线和可行驶区域，YOLO11s 补充车辆、行人、骑行者、摩托车、交通灯、
交通标志、锥桶和护栏。权重不提交 Git，训练与复现见 `training/README.md`：

```bash
python -m scene_understanding.realtime_perception.run \
  --backend yolop_yolo11 \
  --yolop-root /root/autodl-tmp/models/external/YOLOP \
  --yolo11-weights /root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt \
  --image-size 640 \
  --object-image-size 640 \
  --score-threshold 0.10 \
  --capture-index /path/to/capture_index.jsonl \
  --output /path/to/perception_frames.jsonl \
  --summary /path/to/perception_summary.json
```

RTX 5090 实测结果：

| 配置 | BDD100K R/P | nuScenes R/P | CARLA 灯 R/P | 稳态 P95 |
|---|---:|---:|---:|---:|
| 通用专项 YOLO11s 640 | 61.16/64.96% | 57.36/45.01% | 13.87/47.69% | 27.11-49.10 ms |
| 通用专项 YOLO11s 768 | 63.29/65.20% | 58.79/45.23% | 15.21/37.99% | 34.18-44.17 ms |
| 域适配 YOLO11s 640 | 62.53/62.03% | 58.64/44.20% | 17.67/56.43% | 31.44-41.36 ms |
| 域适配 YOLO11s 768 | 65.10/62.57% | 59.91/44.35% | 19.91/59.33% | 25.65-48.14 ms |

BDD100K 和 nuScenes 各使用与训练、验证清单隔离的 1,000 帧测试集。CARLA 使用
修正 `get_light_boxes()` 投影后的 straight 36 帧、447 个可见灯头组件。768 可提高
小目标召回，但 CARLA P95 仅剩约 1.9 ms 余量；实时默认使用 640，独占 RTX 5090
时可改为 768。`--infrastructure-tiles` 会使 CARLA P95 升到 60.01 ms，只用于离线分析。

### 历史尝试与选型结论

| 尝试 | 结果 | 最终定位 |
| --- | --- | --- |
| Qwen2.5-VL-3B 直接解释 CARLA 关键帧 | 41/41 输出最终 Schema 合法；平均 6.325 s，P95 12.000 s，峰值显存约 7.16 GiB | 保留为低频异步语义基线，不进入实时安全环 |
| MiniCPM-V 4.6 16× | CARLA 9/9 Schema 合法，平均 3.636 s，峰值约 3.06 GiB；未输出可用目标框 | 只保留语义摘要对照 |
| 通用 YOLO11n/YOLOP | 延迟较低，但行人、骑行者、交通灯和标志召回不足 | 不作为最终默认配置 |
| 通用专项 YOLO11s | BDD100K、nuScenes 泛化优于小模型，CARLA 交通灯域差距明显 | 作为域适配前对照 |
| CARLA 域适配 YOLO11s 640 | 在保持 BDD100K/nuScenes 性能的同时提高 CARLA 目标召回；重复仿真稳态 P95 31.15 ms | 当前默认实时目标检测器 |
| 768 分辨率或基础图上半区切片 | 小目标召回继续提高，但延迟余量小；切片 P95 60.01 ms | 768 仅独占 GPU 精度模式，切片仅离线使用 |

上述比较说明，视觉大模型更适合低频开放语义，实时目标与车道感知应由小模型承担；
CARLA 真值和确定性规则仍负责安全兜底。闭环中“视觉未检出”不能直接解释为“道路安全”。

### 2026-07-23 CARLA 多轮回归

在 Linux、CARLA 0.9.16、Python 3.12.13、RTX 5090 上，对当前三个
`Town10HD_Opt` 短场景各运行 5 次，共 15 次。控制使用当前规则策略和 PID；
场景理解按 640×360、每 20 个仿真帧采集一个严格同帧样本。

| 场景 | 完成 | 平均时长 | 平均距离 | 碰撞/压线/超速 | 关键帧/超时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `straight_driving` | 5/5 | 18.09 s | 93.50 m | 0/0/0 | 90/0 |
| `pedestrian_crossing` | 5/5 | 14.54 s | 29.90 m | 0/0/0 | 74/0 |
| `emergency_brake` | 5/5 | 7.14 s | 21.99 m | 0/0/0 | 37/0 |
| 合计 | 15/15 | 13.26 s | 48.46 m | 0/0/0 | 201/0 |

控制规则与 PID 的逐轮响应 P95 均值为 0.317 ms，纯控制 P95 均值为
0.292 ms。该值不包含指令解析和视觉感知，不能当作比赛端到端延迟。

对 201 个关键帧使用默认实时配置
“YOLOP 640 + CARLA 域适配 YOLO11s 640 + ByteTrack”重新推理：

| 指标 | 结果 |
| --- | ---: |
| 稳态平均延迟 | 29.66 ms |
| 稳态 P95 | 31.15 ms |
| 首帧冷启动 | 77.43 ms |
| 决策相关前车召回 | 37/37（100%） |
| 决策相关行人召回 | 38/50（76%） |
| 全投影交通灯召回/精确率 | 33.06% / 67.35% |
| 全投影总体召回/精确率 | 35.67% / 45.34% |

严格总体指标包含 1,685 个远距离交通灯灯头组件，且 CARLA 场景投影没有给画面中
全部停放和背景车辆提供完整检测真值，因此车辆“精确率”会被额外预测显著低估。
本结果应按类别和决策相关性阅读，不能用总体值替代比赛要求的多模态语义对齐精度。

题目中的 90% 是 CARLA 系统场景任务完成率，不是检测 mAP 或召回率。当前
15/15 只证明固定短场景的回归稳定性；比赛正式工况仍要求 5 km 基础操控、
8 km 复杂避障、6 km 雨夜极限应急，以及不同路线、天气、交通流和组合指令。
完整原始证据保存在
`experiment/CARLA/outputs/runs/repeat_benchmark_v1/`，由 `.gitignore` 排除。
完整评测边界见 `PERCEPTION_EXPERIMENT_REPORT.md`。

对通用图像清单运行和评测：

```bash
python -m scene_understanding.realtime_perception.run_dataset \
  --manifest /path/to/val.jsonl \
  --output /path/to/perception.jsonl \
  --summary /path/to/summary.json \
  --backend yolop_yolo11 \
  --yolo11-weights /root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt \
  --image-size 640 --object-image-size 640 \
  --score-threshold 0.10 --limit 1000

python -m scene_understanding.realtime_perception.evaluate_dataset \
  --results /path/to/perception.jsonl \
  --manifest /path/to/val.jsonl \
  --output /path/to/metrics.json \
  --iou-threshold 0.5 --limit 1000
```

## JSON 联调接口

| 文件 | 生产方 | 消费方 | 作用 |
|---|---|---|---|
| `driving_intent.json` | 结构化指令解析模块 | 本模块 | 驾驶步骤、目标、依赖关系和阻塞策略 |
| `world_state.json` | CARLA 世界状态采集器 | 对齐与风险模块 | 主车、交通参与者、车道、环境和传感器事件 |
| `semantic_alignment.json` | 本模块 | 控制决策模块 | 将“行人、慢车、前车、车道”等目标关联到实体 |
| `risk_assessment.json` | 本模块 | 控制决策模块 | 距离、TTC、碰撞风险和左右变道安全性 |
| `control_decision.json` | 本模块 | 团队控制模块 | 单帧安全门控后的扁平控制动作 |
| `control_plan_state.json` | 本模块 | 下一帧计划执行器 | 多步骤计划状态和当前活动步骤 |
| `step_feedback.json` | 控制器或实验评估器 | 本模块 | 当前步骤的完成、失败或跳过反馈 |

下游 `control_decision.json` 已与团队控制模块的 `control.protocol.normalize_intent` 接口联调。主要动作包括：

```text
keep_lane
accelerate
decelerate
stop
emergency_brake
lane_change_left
lane_change_right
turn_left
turn_right
```

### Python 直接调用

集成进同一进程时，可直接调用稳定函数接口；输入仍使用与 Schema 一致的 Python
字典，返回值可直接序列化为对应 JSON：

```python
from scene_understanding.src.control_decision import build_control_decision
from scene_understanding.src.driving_intent_alignment import align_driving_intent
from scene_understanding.src.risk_interface import assess_scene_risk

alignment = align_driving_intent(driving_intent, world_state)
risk = assess_scene_risk(world_state)
decision = build_control_decision(
    driving_intent,
    world_state,
    alignment,
    risk,
)
```

多步骤组合指令使用
`scene_understanding.src.control_plan_executor.advance_control_plan(...)`，并在下一帧传回
`prior_state` 和可选 `feedback`。实时图像路径使用
`scene_understanding.realtime_perception.pipeline.RealtimePerceptionPipeline.process(...)`；
批量联调优先使用下述 CLI，以自动生成审计文件。

## 基本联调流程

### 1. 语义对齐

```bash
python -m scene_understanding.scripts.align_driving_intent \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --output outputs/semantic_alignment.json
```

目标不可见或不支持时会明确输出未匹配结果，不会虚构场景实体。

### 2. 风险评估

```bash
python -m scene_understanding.scripts.assess_risk \
  --world-state inputs/world_state.json \
  --output outputs/risk_assessment.json
```

风险输出包含安全跟车距离、TTC、目标风险、碰撞与压线事件，以及左右变道安全判断。

### 3. 单步控制决策

```bash
python -m scene_understanding.scripts.build_control_decision \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --output outputs/control_decision.json
```

风险规则始终高于普通驾驶动作。目标未匹配、车道不安全或输入状态无效时，模块按照 `on_blocked` 策略减速或停车。

### 4. 多步骤计划推进

初始化计划：

```bash
python -m scene_understanding.scripts.advance_control_plan \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --state-output outputs/control_plan_state.json \
  --decision-output outputs/control_decision.json
```

后续帧使用 `--state` 读取上一状态，并可用 `--feedback` 提交当前步骤的显式执行结果。计划支持 `PENDING`、`ACTIVE`、`WAITING`、`COMPLETED`、`SKIPPED` 和 `FAILED` 等步骤状态。

## CARLA 关键帧视觉桥接

视觉链路已经补充 CARLA Actor 三维框投影、同帧推理清单、Qwen 常驻服务和
`WorldState` 视觉语义融合。它将 Qwen 输出写入
`objects[].semantic_matches`，但不会覆盖 CARLA 提供的位置、速度、距离、TTC、
交通灯状态或传感器事件。

主要入口：

- `core/carla_bbox_projection.py`：将 CARLA Actor 包围框投影到归一化二维图像框；
- `core/prepare_carla_samples.py`：保存同帧采集记录并生成 Qwen JSONL manifest；
- `core/qwen_scene_service.py`：只加载一次模型，以长度为 1 的最新帧队列异步推理；
- `core/visual_semantic_fusion.py`：执行同类别、一对一的视觉对象与 CARLA Actor 匹配；
- `SCENE_FRAME_INTERFACE.md`：完整接口、运行命令、实时约束和评测规范。

紧急制动、TTC 和控制决策始终使用同步 CARLA 真值路径，不能等待视觉模型。视觉
模型失败、输出非法或结果过期时保留原始 `WorldState`。

当前 AutoDL 视觉基线使用：

```text
模型：Qwen/Qwen2.5-VL-3B-Instruct
本地权重：/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
Python：3.12.13
PyTorch：2.11.0+cu130（包含 sm_120）
Torchvision：0.26.0+cu130
Transformers：4.57.6
```

可复用数据盘环境运行：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torchvision==0.26.0+cu130
python -c "import torch; print(torch.cuda.get_arch_list())"
```

`experiment/CARLA/run_control_experiment.py --scene-capture` 可直接从 `lx` 场景生成
本模块需要的同帧采集索引、图像、`WorldState` 和 Actor 投影记录，具体命令见
`experiment/CARLA/README.md`。

当前 RTX 5090 冒烟测试使用一张 800×600 道路示意图，结果如下：

```text
生成耗时：5.63 s（不含约 2 s 模型加载）
峰值显存：7.19 GiB
处理后图像：728×532
最终 Schema：valid
```

原始输出给出了红灯状态但漏掉交通灯对象，触发了一致性校验。归一化器现会在没有
视觉对象支撑时把交通灯状态降级为 `unknown`，同时保留原始错误和归一化动作供审计，
不会为了通过 Schema 虚构交通灯对象。

真实 CARLA 0.9.16 离屏验证已完成。交通灯真值现在通过
`TrafficLight.get_light_boxes()` 按灯头独立投影，并用组件 ID 回连父 actor；旧的
actor 大框结果已作废。直线行驶、行人横穿和紧急制动三个短场景各重复 5 次，
当前回归任务完成率为 15/15。视觉检测、异步 VLM 和系统任务完成率使用不同指标，详细结果见
`PERCEPTION_EXPERIMENT_REPORT.md`。

MiniCPM-V 4.6 已作为第二个异步对照接入。固定环境要求
`transformers==5.7.0`，输入必须使用
`.to(model.device, dtype=model.dtype)`。16×模式在 CARLA 9 帧中 9/9 Schema
合法，平均 3.636 s、峰值 3.06 GiB；DriveLM-nuScenes 10 帧中 9/10 合法，有效帧
平均 4.604 s、峰值 3.31 GiB。两组均未输出可定位对象；4×细节模式虽能在摘要中提及
车辆或绿灯，仍没有目标框。因此 MiniCPM 只保留为语义摘要对照，不承担目标定位或
任何实时安全判断。

复现 MiniCPM 异步对照：

```bash
conda activate /root/autodl-tmp/conda_envs/minicpm_v46
python -m pip install \
  -r scene_understanding/async_semantics/requirements-minicpm.txt

PYTHONPATH=. python -m scene_understanding.async_semantics.run_minicpm_scene_inference \
  --manifest /path/to/scene_manifest.jsonl \
  --model-path /root/autodl-tmp/models/MiniCPM-V-4.6 \
  --output /path/to/minicpm_results.jsonl \
  --max-new-tokens 768 --downsample-mode 16x --max-slice-nums 9
```

该命令输出逐帧状态、原始文本、归一化 JSON、Schema 错误、耗时和峰值显存；
`--resume` 可跳过已有 `frame_id`。它是离线/异步命令，不作为控制循环启动项。

## 数据集

数据存放于 `/root/autodl-tmp/datasets/scene_understanding`，不提交 Git：

- `nuscenes_full/raw`：完整 nuScenes v1.0-trainval、sweeps、maps 和元数据；
- `nuscenes_full/organized`：204,894 条官方 train/val 六相机记录和 1,416,337 个
  可见 2D 投影框；
- `bdd100k/organized`：70,000 张训练图、10,500 张验证图及可读检测 JSONL；
- `drivelm_nuscenes`：DriveLM 图像、QA 和 CAM_FRONT 语义评测清单；
- CARLA 同帧采集仍位于 `experiment/CARLA/outputs`，由 `.gitignore` 排除。

数据环境安装和完整准备命令见 `scripts/data/README.md`。

## CARLA 闭环验证

已在 CARLA 0.9.16 中分别完成以下闭环实验：

1. 行人横穿时减速避让，行人通过并确认零碰撞后完成步骤；
2. 关联前方慢车，在合法长直路段完成左变道并稳定保持目标车道；
3. 在超车道加速，使同一慢车从主车前方变为至少后方 8 米，且全程零碰撞。

对应文档：

- `PEDESTRIAN_CONTROL_EXPERIMENT.md`
- `LANE_CHANGE_CONTROL_EXPERIMENT.md`
- `OVERTAKE_CONTROL_EXPERIMENT.md`
- `CONTROL_PLAN_EXECUTION.md`
- `CONTROL_DECISION.md`

实验运行入口位于 `scripts/run_*_control_experiment.py`。实验输出和完整时间线属于运行证据，不提交到源代码目录。

## 测试

在仓库根目录运行：

```bash
python -m unittest discover -s scene_understanding/tests -v
```

当前测试集共 171 项，覆盖：

- JSON 结构和确定性校验；
- WorldState 坐标与相对运动；
- CARLA Actor、车道、交通灯与传感器采集；
- 施工锥桶和道路障碍物等安全相关静态 Actor 类别映射；
- CARLA 三维框投影、关键帧清单和跨帧拒绝；
- 一对一视觉语义融合、低置信度过滤和异步最新帧替换；
- 风险评估和车道安全判断；
- 指令目标语义对齐；
- 控制决策安全优先级；
- 多步骤计划状态推进；
- 行人避让、变道和超车完成条件。

## Schema 与示例

所有稳定 JSON 契约位于 `schemas/`，可直接用于模块间字段确认。示例位于 `schemas/examples/`，包括：

- `world_state.example.json`
- `semantic_alignment.example.json`
- `risk_assessment.example.json`
- `control_decision.example.json`
- `control_plan_state.example.json`
- `step_feedback.example.json`

更完整的场景理解和视觉推理说明见 `schemas/README.md`。

## 当前边界

- 三个闭环步骤目前通过持久化 JSON 状态在独立 CARLA 场景中依次验证，并非同一场景进程中的一次连续演示。
- 当前超车完成条件是超过慢车并稳定保持超车道，尚未包含返回原车道。
- 风险阈值属于确定性研究规则，需要在更多场景和速度分布中继续标定。
- 视觉模型已在修正投影后的 36 个隔离 CARLA 关键帧上复测；交通灯小目标定位仍需改进。
- 当前 BDD100K、nuScenes 各 1,000 帧独立测试和 CARLA 36 帧测试都不替代面向目标
  硬件、目标摄像头和批量场景矩阵的大规模标定。
- CARLA 启动和 GPU 图形库兼容方式由部署环境决定，不随本模块提交大型镜像或运行产物。
- 当前通用权重对行人、骑行者、交通灯和交通标志召回不足，不能仅凭视觉未检出放行。
- Waymo 尚未下载；必须由项目成员先接受官方许可并完成 Google Cloud 登录。
