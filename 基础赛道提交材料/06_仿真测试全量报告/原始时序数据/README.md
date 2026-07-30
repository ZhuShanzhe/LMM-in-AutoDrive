# 原始时序数据

当前未找到按正式三场景完整归档的全量时序数据。本目录保留以下最终结构：

```text
scene_1_basic/
scene_2_complex/
scene_3_emergency/
```

每次运行至少包含：

- `run_config.json`
- `voice_events.jsonl`
- `driving_intents.jsonl`
- `perception_frames.jsonl`
- `world_states.jsonl`
- `risk_assessments.jsonl`
- `decision_trace.jsonl`
- `vehicle_telemetry.jsonl`
- `safety_events.jsonl`
- `summary.json`
- 对应视频文件或视频索引

大文件不进入 Git，但必须进入最终提交压缩包或评委可访问的附件。
