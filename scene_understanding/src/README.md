# 场景理解与规则决策

| 文件 | 作用 |
|---|---|
| `driving_intent_alignment.py` | 将 DrivingIntent 目标与场景实体、车道和路线对象对齐 |
| `risk_interface.py` | 生成统一风险评估结构 |
| `high_level_driving_actions.py` | 将结构化步骤映射为车辆高层动作并校验风险字段 |
| `control_decision.py` | 综合意图、对齐结果和风险，输出 `ControlDecision 1.0` |
| `control_plan_executor.py` | 使用 FSM 管理复合步骤的激活、等待、完成和推进 |
| `execution_feedback.py` | 将车辆执行结果转换为步骤反馈 |

该目录构成规则 FSM 决策链；VLA 链路复用相同的风险、安全门和 `ControlDecision` 输出协议。
