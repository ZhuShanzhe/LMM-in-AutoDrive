# VLA 推理核心

| 文件 | 作用 |
|---|---|
| `structured_bev.py` | 将场景实体、车道和 LiDAR 特征栅格化 |
| `bev_encoder.py` | 编码多模态 BEV 特征 |
| `intent_encoder.py` | 编码 DrivingIntent token 特征 |
| `decision_adapter.py` | 融合场景与意图并预测高层动作 |
| `pipeline.py` | 组织模型推理、时序监督和 proposal 输出 |
| `safety_bridge.py` | 将 VLA proposal 与规则 FSM、安全门融合为 `ControlDecision` |
| `contracts.py` | 定义输入、输出和张量契约 |

CARLA 接入代码位于 `experiment/CARLA/scene2_closed_loop.py` 和
`experiment/CARLA/control/structured_vla_scene_bridge_policy.py`。
