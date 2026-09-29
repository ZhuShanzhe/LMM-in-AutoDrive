# 同源数据及结果

已有工具：[同源回放契约](../../../experiment/CARLA/evaluation/REPLAY_CONTRACT.md)、`experiment/CARLA/evaluation/challenge_capture_audit.py`、`experiment/CARLA/evaluation/sensor_replay.py`。

本轮新地图冻结数据及完整对照结果：**未完成**。需要记录数据来源、场景/天气/种子、有效帧数、标注方法、数据集SHA256、分割及剔除原因；原始版和优化版用同一帧和同一音频，时序模型按原始连续序列运行，不能跳帧破坏记忆。

真值仅用于独立评价，不能传给声称纯传感器输入的模型。关键目标、灯色、追踪、语义对齐、指令执行、ASR分别统计，整体场景完成率另列。
