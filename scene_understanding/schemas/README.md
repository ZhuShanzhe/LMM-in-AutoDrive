# 场景理解输出格式

`scene_understanding.schema.json` 定义单张摄像头关键帧的多模态模型输出，
`examples/scene_understanding.example.json` 是一份合法示例。

`world_state.schema.json` 定义融合 CARLA 真值、传感器事件和视觉语义后的统一
`WorldState`，`examples/world_state.example.json` 是一份合法示例。它采用米、秒和
米每秒，并明确约定自车局部坐标为：纵向向前为正、横向向右为正、竖直向上为正。

## 设计规则

1. 模型必须只输出一个 JSON 对象，不要输出 Markdown 代码块或额外解释。
2. `frame_id`、`source` 和 `camera_name` 由数据读取程序填写，不能依靠模型猜测。
3. 每个可见目标使用临时编号 `vlm_obj_001`、`vlm_obj_002`，后续再与数据集真值或 CARLA `actor_id` 对齐。
4. `bbox_2d` 顺序固定为 `[x_min, y_min, x_max, y_max]`，坐标归一化到 0～1。
5. 看不清或无法确定时填写 `unknown` 或 `null`，不能编造信息。
6. `distance_level` 只是视觉上的远近描述，不能用于计算 TTC。
7. 精确距离、速度、TTC、最终风险等级和换道安全结果不属于该格式；这些结果必须由数据集或 CARLA 真值通过规则程序计算。
8. `confidence` 是模型自报置信度，不能当作真实概率；专项定位未提供时使用 `null`，不能由程序编造。
9. 交通灯和交通标志属于固定设施，`motion_state` 必须为 `unknown`。
10. 当前离线感知阶段固定输出空的 `potential_hazards`；风险由后续规则模块计算。
11. 如果场景给出红、黄或绿灯状态，必须同时提供至少一个交通灯目标框。
12. `WorldState` 的米制位置、速度和距离优先读取 CARLA Actor/Map API；Qwen
    只通过 `semantic_matches` 补充类别、描述和图像框，不能覆盖 CARLA 真值。
13. `relative_longitudinal_speed_mps` 定义为“目标减自车”的纵向速度；负数表示
    自车正在追近前方目标。`closing_speed_mps` 为正时表示两者三维距离正在缩小。

## 字段分工

| 字段 | 含义 | 后续用途 |
|---|---|---|
| `scene` | 道路和整体场景描述 | 生成场景解释 |
| `objects` | 车辆、行人、交通设施等目标 | 与真值目标对齐 |
| `bbox_2d` | 目标在图片中的位置 | 与投影框或标注框匹配 |
| `relative_position` | 相对自车的方向 | 辅助语义匹配 |
| `lane_relation` | 目标与自车车道的关系 | 判断是否影响自车路径 |
| `motion_state` | 视觉上观察到的运动状态 | 场景解释和匹配辅助 |
| `potential_hazards` | 模型观察到的潜在危险 | 提供解释，不直接决定最终风险 |

## 第一轮测试范围

第一轮只使用 `CAM_FRONT`，处理约 30 张 nuScenes 样本。确认格式稳定后，再扩展到其他摄像头或更多数据。

## 配套文件

- `../prompts/archive/scene_understanding/initial.txt`：初始提示词，保留用于复现实验。
- `../prompts/archive/scene_understanding/no_placeholder_objects.txt`：去除占位目标的历史提示词。
- `../prompts/archive/scene_understanding/complete_object_fields.txt`：规定完整对象字段的历史提示词。
- `../prompts/archive/scene_understanding/fixed_infrastructure_constraints.txt`：增加固定设施语义约束的历史提示词。
- `../prompts/archive/scene_understanding/perception_risk_separation.txt`：分离感知与风险计算的历史提示词。
- `../prompts/scene_understanding.txt`：强化小型交通设施扫描与灯色定位一致性的当前提示词。
- `../scene_understanding/core/validate_scene_output.py`：不依赖第三方库的输出校验程序。
- `../scene_understanding/core/normalize_scene_output.py`：保留审计记录的确定性输出适配器。
- `../scene_understanding/core/prepare_nuscenes_samples.py`：从 DriveLM 标注生成 `CAM_FRONT` 推理清单。
- `../scene_understanding/core/run_qwen_scene_inference.py`：使用本地 Qwen2.5-VL 模型运行确定性推理。
- `../scene_understanding/core/evaluate_scene_alignment.py`：使用类别、中心点、IoU 和信号灯状态对齐真值。
- `../scene_understanding/core/reprocess_scene_results.py`：不重跑模型，离线应用最新版确定性适配规则。
- `../prompts/archive/traffic_control_grounding/structured_output.txt`：完整结构专项定位的历史提示词。
- `../prompts/traffic_control_grounding.txt`：采用 Qwen 原生 `bbox_2d + label` 格式的当前专项定位提示词。
- `../scene_understanding/core/run_qwen_traffic_control_grounding.py`：运行交通设施专项定位。
- `../scene_understanding/core/merge_traffic_control_results.py`：去重并合并通用场景结果与专项定位结果。
- `world_state.schema.json`：CARLA、视觉语义和传感器事件的统一世界状态格式。
- `examples/world_state.example.json`：包含一辆前车的合法世界状态示例。
- `../scene_understanding/core/world_state.py`：世界状态校验、坐标换算和相对运动计算。
- `../scene_understanding/core/carla_world_state.py`：面向 CARLA 0.9.16 的世界状态采集适配器；
  接收队友场景提供的 `world` 和 `scenario.ego_vehicle`，不改变场景代码。
- `../scene_understanding/core/carla_sensor_manager.py`：统一管理前视 RGB、碰撞和压线传感器，
  使用 CARLA 原始帧号保存图片，并按帧缓存碰撞与压线事件。
- `risk_assessment.schema.json`：风险等级、逐目标 TTC 和左右变道判断的输出格式。
- `examples/risk_assessment.example.json`：使用示例 WorldState 计算出的合法风险结果。
- `../scene_understanding/core/risk_assessment.py`：保留队长方案的基础安全距离和 TTC
  阈值，并增加停车距离、加塞/横穿冲突预测、碰撞前紧急制动以及方向相关的
  变道 TTC；不依赖 CARLA 或模型运行时。
- `semantic_alignment.schema.json`：单个文本指令对象与 CARLA actor、车道或路口的对齐结果格式。
- `examples/semantic_alignment.example.json`：将单个“前车”引用对齐到具体 CARLA actor 的示例。
- `driving_intent_alignment.schema.json`：DrivingIntent 多步骤目标与单帧 WorldState 的批量对齐结果格式。
- `examples/driving_intent_alignment.example.json`：对行人避让、变道和超车三个步骤执行批量对齐的示例。
- `../core/object_matcher.py`：归一化“行人、前车、慢车、左车道、路口”等方案词汇，
  并按类别、方向、车道关系和距离选择候选对象。
- `../core/semantic_alignment.py`：保留单对象文本引用对齐接口，并为匹配实体附加风险等级。
- `../src/driving_intent_alignment.py`：兼容 DrivingIntent 1.0/1.1，并接收当前
  DrivingIntent 1.2 的共享实体、`target_ref` 和 `goal_conditions`，对每个步骤目标执行
  批量语义对齐；输出同时记录目标条件解析、属性匹配结果，并明确区分未匹配、
  未知类型、歧义匹配和 WorldState 能力暂不可用。

校验示例：

```bash
python -m scene_understanding.core.validate_scene_output \
  scene_understanding/schemas/examples/scene_understanding.example.json \
  --frame-id nuscenes_sample_000001 \
  --source nuscenes \
  --camera-name CAM_FRONT
```

生成 nuScenes 前视样本清单：

```bash
python -m scene_understanding.core.prepare_nuscenes_samples \
  /path/to/DriveLM/challenge/data/train_sample.json \
  --image-root /path/to/DriveLM/challenge/llama_adapter_v2_multimodal7b \
  --output outputs/nuscenes_cam_front_manifest.jsonl \
  --limit 30 \
  --require-images
```

在已分配 GPU 的计算节点上运行第一帧：

```bash
conda run -n vllm python -m scene_understanding.core.run_qwen_scene_inference \
  --manifest outputs/nuscenes_cam_front_manifest.jsonl \
  --model-path /mnt/beegfs/home/reco/models/Qwen2.5-VL-3B-Instruct \
  --output outputs/qwen_scene_results.jsonl \
  --limit 1 \
  --fail-fast
```

脚本只从本地加载模型，不访问网络；结果逐帧追加写入，后续可使用 `--resume` 继续。

对齐 DriveLM 真值并生成评测报告：

```bash
python -m scene_understanding.core.evaluate_scene_alignment \
  --manifest outputs/nuscenes_cam_front_manifest.jsonl \
  --inference outputs/qwen_scene_results.jsonl \
  --output outputs/qwen_scene_alignment.json
```

DriveLM 的一个大框可能包含多个同类设施。因此主定位指标采用“类别一致且预测框中心落入真值框”，同时保留 IoU 作为辅助指标。

针对指定帧运行第二阶段交通设施定位：

```bash
conda run -n vllm python -m scene_understanding.core.run_qwen_traffic_control_grounding \
  --manifest outputs/nuscenes_cam_front_manifest.jsonl \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --output outputs/traffic_control_grounding.jsonl \
  --frame-index 5 \
  --frame-index 6 \
  --frame-index 8
```

合并两阶段结果：

```bash
python -m scene_understanding.core.merge_traffic_control_results \
  --base outputs/qwen_scene_results.jsonl \
  --grounding outputs/traffic_control_grounding.jsonl \
  --output outputs/qwen_scene_results_merged.jsonl
```

校验统一世界状态示例（无需 CARLA、GPU 或第三方库）：

```bash
python -m scene_understanding.core.world_state scene_understanding/schemas/examples/world_state.example.json
```

连接 CARLA 0.9.16 后，可以在完成一次 `world.tick()` 后采集：

```python
from scene_understanding.core.carla_world_state import CarlaWorldStateCollector

collector = CarlaWorldStateCollector(world, scenario.ego_vehicle)
state = collector.collect()
```

采集器记录 CARLA `WorldSnapshot.frame` 和 `timestamp.elapsed_seconds`，并将所有速度
统一为 m/s。它不会使用视觉模型估算米制距离；`semantic_matches` 留给后续语义对齐。

传感器和 WorldState 的同步用法：

```python
from scene_understanding.core.carla_sensor_manager import CarlaSensorManager
from scene_understanding.core.carla_world_state import CarlaWorldStateCollector

sensors = CarlaSensorManager(world, scenario.ego_vehicle)
sensors.setup()
collector = CarlaWorldStateCollector(world, scenario.ego_vehicle)

try:
    frame = world.tick()
    events = sensors.drain_events_through(frame)
    state = collector.collect(sensor_events=events)
finally:
    sensors.destroy()
```

碰撞回调保存碰撞对象与法向冲量（N·s）；压线回调保存跨越的标线类型。事件只在
对应或更晚的 WorldState 帧中被取出，避免未来帧事件提前进入当前状态。

离线计算风险：

```bash
python -m scene_understanding.core.risk_assessment \
  scene_understanding/schemas/examples/world_state.example.json \
  --output outputs/risk_assessment.example.json
```

风险模块保留方案中的固定基础阈值：自车速度低于 30 km/h、30–60 km/h、
高于 60 km/h 时安全距离分别为 10 m、20 m、40 m；TTC 大于 4 s、
2–4 s、1–2 s、小于 1 s 分别对应无、低、中、高风险。

场景增强规则使用 WorldState 的纵向/横向相对位置和速度：停止车辆及静态障碍物
分别使用 4 m/s² 和 8 m/s² 制动距离划定提前减速区和紧急制动区；相邻车辆、
路侧行人和骑行者预测
4 s 内是否进入主车走廊，并要求进入时的预测纵向间距不大于当前安全距离。
左右变道分别根据前车变慢和后车逼近计算方向相关 TTC。碰撞、TTC 小于 1 s、
1 s 内迫近路径冲突或应急停车距离不足均触发高风险和
`emergency_brake` 建议；仅进入舒适制动区时建议 `decelerate`。
输出字段和 `schema_version: 1.0` 保持不变，新增场景语义通过 `reason_codes`
审计，不影响现有决策和控制接口。

语义对齐示例（可选地复用同一帧风险输出）：

```bash
python -m scene_understanding.core.semantic_alignment \
  "前车" \
  scene_understanding/schemas/examples/world_state.example.json \
  --risk outputs/risk_assessment.example.json \
  --output outputs/semantic_alignment.example.json
```

该模块接受普通文本，也接受上游指令解析器包含 `target_object`、`object`、
`target`、`entity`、`text` 或 `intent` 等常见字段的字典。无法识别或当前帧中
找不到的对象会明确返回 `alignment_success: false`，不会臆造 CARLA 对象。
