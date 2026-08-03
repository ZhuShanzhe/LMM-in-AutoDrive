# CARLA 关键帧解释与视觉语义融合接口

本接口位于摄像头采集与语义对齐之间：使用 Qwen2.5-VL 解释 CARLA
关键帧，将视觉对象与同帧 CARLA Actor 投影框进行一对一匹配，并把结果写入
`WorldState.objects[].semantic_matches`。后续语义对齐、风险判断和控制模块继续读取
原有 `WorldState`，无需依赖视觉模型内部实现。

## 模块边界

视觉链路只提供类别、图像框和文字描述。以下安全相关字段只能来自 CARLA Actor、
Map 或传感器 API，视觉结果不得覆盖：

- 世界坐标、相对位置和车道关系；
- 速度、距离、相对速度和 TTC；
- 碰撞、压线和变道安全结果；
- CARLA 已提供的交通灯状态。

视觉服务发生超时、输出非法或没有匹配对象时，系统保留原始 `WorldState`，同步风险
和控制链路仍可继续运行。

## 数据流

```text
CarlaSensorManager camera record
        +
CarlaWorldStateCollector WorldState
        +
CARLA actor 3D bbox projection
        |
        v
CARLA capture index / Qwen manifest
        |
        v
Qwen2.5-VL parsed_output
        |
        v
visual_semantic_fusion
        |
        v
enriched WorldState
        |
        v
semantic_alignment / risk_assessment
```

## 同帧约束

每个关键帧必须共享同一个 CARLA 原始帧号：

```text
camera_record.frame == WorldState.simulation_frame
capture.frame_id == WorldState.frame_id
projection.frame_id == Qwen parsed_output.frame_id
projection.camera_name == Qwen parsed_output.camera_name
```

任何一项不一致都必须拒绝融合，禁止把旧图像结果附加到新世界状态。

## CARLA 投影记录

`core/carla_bbox_projection.py` 使用 Actor 三维包围框、相机逆变换和相机内参生成归一化
二维框：

```json
{
  "schema_version": "1.0",
  "frame_id": "carla_000123",
  "camera_name": "front_rgb",
  "image_width": 800,
  "image_height": 600,
  "objects": [
    {
      "world_object_id": "carla_actor_42",
      "source_object_id": "42",
      "category": "vehicle",
      "bbox_2d": [0.4, 0.35, 0.6, 0.7]
    }
  ]
}
```

在 CARLA 循环中可调用：

```python
from scene_understanding.core.carla_bbox_projection import project_world_state_objects

projection = project_world_state_objects(
    world_state,
    world.get_actors(),
    sensors.front_camera_sensor,
    camera_name="front_rgb",
    image_width=800,
    image_height=600,
    fov_deg=90.0,
)

camera_record = sensors.camera_frame(world_state["simulation_frame"])
if camera_record is None:
    # The camera callback has not delivered this exact frame yet; skip capture.
    return
```

## 采集与推理清单

使用 `write_capture_bundle` 保存同帧 WorldState 与投影记录，并把返回值逐行写入
`capture_index.jsonl`。随后生成现有 Qwen 推理程序可直接读取的 manifest：

```bash
python -m scene_understanding.core.prepare_carla_samples \
  --capture-index outputs/carla_capture/capture_index.jsonl \
  --prompt scene_understanding/prompts/scene_understanding.txt \
  --output outputs/carla_manifest.jsonl
```

先运行 10 帧验证，再扩展到 100 帧和比赛同源的 1000 帧：

```bash
conda run -n vllm python -m scene_understanding.core.run_qwen_scene_inference \
  --manifest outputs/carla_manifest.jsonl \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --output outputs/carla_scene_results.jsonl \
  --limit 10 \
  --fail-fast
```

## 视觉语义融合

融合器按照同类别、IoU 和中心点包含关系建立确定性的一对一匹配。同一摄像头重复融合
时会替换旧匹配，不会重复附加。

```bash
python -m scene_understanding.core.visual_semantic_fusion \
  --world-state outputs/carla_capture/world_states/carla_000123.json \
  --projection outputs/carla_capture/projections/carla_000123.json \
  --inference outputs/frame_000123_inference.json \
  --output outputs/enriched_world_state.json \
  --audit outputs/frame_000123_fusion_audit.json
```

写入后的对象语义字段示例：

```json
{
  "camera_name": "front_rgb",
  "visual_object_id": "vlm_obj_001",
  "bbox_2d": [0.41, 0.36, 0.59, 0.69],
  "description": "vehicle; car; red; front; ego lane",
  "confidence": 0.9
}
```

## 异步实时接入

`QwenSceneService` 只加载一次模型，`LatestFrameWorker` 的待处理队列长度固定为 1。
模型忙碌时，新关键帧会替换尚未处理的旧关键帧，防止视觉任务无限积压。

```python
from pathlib import Path
from scene_understanding.core.qwen_scene_service import (
    LatestFrameWorker,
    QwenSceneConfig,
    QwenSceneService,
)

service = QwenSceneService(
    QwenSceneConfig(Path("/path/to/Qwen2.5-VL-3B-Instruct"))
)
worker = LatestFrameWorker(service)
worker.submit(manifest_record)

# 控制循环不等待模型；只读取仍在有效期内的最近结果。
latest = worker.latest(max_age_seconds=1.0)
```

紧急制动、碰撞处理和 TTC 规则必须始终走同步 CARLA 真值路径，不得等待
`LatestFrameWorker`。

## 评测要求

真实 CARLA 数据至少记录：

- 场景输出 JSON 合法率；
- 车辆、行人、锥桶、交通灯等类别的召回率与支持预测率；
- 平均最佳 IoU 和交通灯状态准确率；
- 未匹配视觉对象比例，用于观察幻觉；
- 推理时延 P50、P95、最大值和峰值显存；
- 视觉结果过期、模型异常和回退次数。

测试帧应按场景、路线和天气划分，不能把同一连续序列的相邻帧随机分到不同集合。

## 测试

```bash
python -m unittest discover -s scene_understanding/tests -v
```

投影、清单、跨帧拒绝、一对一融合、低置信度过滤、最新帧替换和异常回退均应由
单元测试覆盖。真实模型和 CARLA 集成结果另存到 `outputs/`，不提交图片、权重或大批量
JSONL 运行产物。
