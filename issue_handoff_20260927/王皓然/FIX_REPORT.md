# 王皓然修复记录（2026-09-27）

## 修改摘要

### W1 速度目标与步骤推进

- 将有效 `SET_SPEED` 明确目标作为绝对执行设定值；只在计划为 `READY`、动作允许正常前向运动且没有阻断原因时提升模型目标，停车、减速、风险和交通约束不会被覆盖。
- 在低层控制反馈中分别记录指令目标、控制器目标、实际安全目标、约束原因和 `REACHABLE/CONSTRAINED/UNREACHABLE` 状态。
- 道路限速、路口、换道过渡、曲率、轨迹稳定性及风险上限继续取优先级更高的较低目标。
- 受限目标连续达到后以 `observed_constrained_speed_completion` 推进；上游明确报告不可达时以 `target_speed_unreachable` 失败，避免步骤永久停留但没有状态说明。

### W2 等待谓词与转向目标

- `PEDESTRIAN_CLEAR`、`PASSENGERS_CLEAR`、`CYCLIST_PASSED`、`CROSSWALK_CLEAR` 等非风险谓词只消费与当前请求、当前步骤、当前帧绑定的显式证据。
- 证据必须包含有效来源并持续 0.5 秒；错误请求、错误步骤、旧帧、过期、无效或风险重新出现都会重置确认窗口。
- TURN 可从当前步骤的 `step_readiness` 或同帧 `SemanticAlignment` 消费可达道路目标；缺失、陈旧、歧义或不可达目标仍保持阻断。
- `universal_vla_controller.py` 已把 `semantic_alignment` 和 `step_readiness` 同时传入 `prepare` 与 `advance`，避免解析阶段和执行阶段使用不同目标。

与黄皓星侧约定的谓词输入示例：

```json
{
  "step_id": {
    "request_id": "request-id",
    "step_id": "step_id",
    "frame_id": "carla_100",
    "condition": "PEDESTRIAN_CLEAR",
    "satisfied": true,
    "valid": true,
    "source": "condition_observer",
    "observed_at_s": 5.0,
    "valid_until_s": 5.2
  }
}
```

道路目标使用同样的请求、步骤、帧和来源绑定，另带：

```json
{
  "reachable": true,
  "ambiguous": false,
  "target_location": {"x": 10.0, "y": 5.0, "z": 0.0}
}
```

### W3 返回原车道

- `RETURN_WHEN_SAFE` 现在编码为 `{"return_to":"ORIGINAL_LANE"}`，不再提前改写为 `RIGHT`。
- PID 执行反馈提供当前、左邻和右邻车道引用及换道合法性；运行时记录已完成换道的来源/目标车道，并允许记录跨命令保留。
- 返回步骤只在观测拓扑能把原车道解析为左侧或右侧相邻合法车道时发出对应动作；已在原车道时稳定确认后完成；缺引用、非相邻或道路标线不允许时显式等待，不编造方向。
- 返回动作继续经过对应目标车道的风险门，风险解除后恢复同一请求和同一步骤。

## 修改文件

- `lightweight_vla_adapter/src/driving_plan_runtime.py`
- `lightweight_vla_adapter/src/longitudinal_contract.py`
- `experiment/CARLA/control/generic_route_pid.py`
- `experiment/CARLA/control/pid_controller.py`
- `experiment/CARLA/control/generic_instruction_fsm.py`
- `experiment/CARLA/scene2_runtime_interface.py`
- `experiment/CARLA/universal_vla_controller.py`
- `scene_understanding/src/control_decision.py`
- 对应测试文件

## 验证结果

- `python issue_handoff_20260927/check_package.py --owner 王皓然`：证据完整性 `PASS`，原始证据未覆盖。
- Python 语法编译和 `git diff --check`：通过。
- 速度、计划推进、等待谓词、转向目标、命令派发、FSM、返回车道及计划执行定向回归：`139 passed`。
- `RETURN_WHEN_SAFE` 契约独立探针：通过，输出保留 `ORIGINAL_LANE` 且不含固定方向。

本机未启动 CARLA 服务，也未加载 Torch 模型或权重，因此不声明三场景闭环验收通过。仍需合并黄皓星的真实谓词/道路目标生产端后进行短程集成与刘旭侧独立全程测试。
