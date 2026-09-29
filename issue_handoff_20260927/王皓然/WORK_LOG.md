# 王皓然：纵向执行、条件推进与回原车道修复日志

本日志记录 `whr-execution-fix-20260927` 分支的实际修改与验证结果。代码在仓库原模块中修改；本目录保留问题证据、处理说明和复核入口，不覆盖原始输入与旧日志。

## 快速复核

从仓库根目录运行：

```bash
python issue_handoff_20260927/check_package.py --owner 王皓然
PYTHONPATH=.:experiment/CARLA python -m pytest -q \
  lightweight_vla_adapter/tests/test_driving_plan_runtime.py \
  lightweight_vla_adapter/tests/test_speed_setpoint_contract.py \
  experiment/CARLA/tests/test_cruise_speed_contract.py \
  experiment/CARLA/tests/test_scene2_town05.py
```

第一条校验证据包完整性，不启动 CARLA、不加载模型权重。第二条覆盖本轮新增的速度目标、计划推进和场景二接口回归；完整定向测试结果见 `FIX_REPORT.md`。

## 完成记录

### W1 [P0] 速度目标与步骤推进

**证据：** 场景二原运行在 `SET_SPEED` 长时间停留，实测速度未持续达到 45 km/h，后续动作未激活。原链路混用了指令目标、模型目标、控制器目标和安全约束后的实际目标，受限时也缺少明确完成或失败状态。

**处理：** 保留明确 `SET_SPEED` 作为长期绝对目标，仅在计划允许正常前向运动且没有阻断原因时提升执行设定值；道路限速、曲率、路口、换道过渡、轨迹稳定性和风险上限继续优先。低层反馈分别记录指令目标、控制器目标、实际安全目标、约束原因及 `REACHABLE/CONSTRAINED/UNREACHABLE` 状态。

**验收：** 可达目标按持续反馈完成；受限目标可按观测到的受限速度推进；上游明确不可达时显式失败，不再无限停留。停车、减速、风险和交通约束不会被 45 km/h 目标覆盖。

### W2 [P0] 等待谓词与 TURN 目标接入

**证据：** 缺少与当前请求、当前步骤及当前帧绑定的显式谓词时，等待不能完成；TURN 目标在解析阶段与执行阶段存在输入不一致风险。

**处理：** 非风险谓词只消费带有效来源、请求、步骤、帧和时效的显式证据，并要求持续确认 0.5 秒。错误请求、错误步骤、旧帧、过期、无效或风险重新出现都会重置确认窗口。TURN 同时接入 `step_readiness` 与同帧 `SemanticAlignment`，并在 `prepare`、`advance` 两阶段使用同一目标契约。

**验收：** 证据缺失时保持等待；证据满足后按序推进；风险恢复后继续当前请求与步骤；旧命令、旧步骤和歧义或不可达道路目标不能完成当前动作。

### W3 [P1] 返回原车道语义

**证据：** `RETURN_WHEN_SAFE` 原先被固定映射为 `RIGHT`，没有使用起始车道、当前车道和邻接拓扑，无法覆盖右移后左返或已经位于原车道的情况。

**处理：** 将返回意图保留为 `{"return_to":"ORIGINAL_LANE"}`；PID 反馈提供当前、左邻、右邻车道引用及换道合法性；运行时记录已完成换道的来源与目标车道。只有观测拓扑能把原车道解析为合法相邻车道时才发出左右动作，已在原车道时稳定确认完成。

**验收：** 左超车后右返、右移后左返、已在原车道、原车道不可达及风险阻断恢复均有覆盖；缺少引用、非相邻或道路标线不允许时显式等待，不固定方向、不编造合法车道。

## 修改位置

- `lightweight_vla_adapter/src/driving_plan_runtime.py`
- `lightweight_vla_adapter/src/longitudinal_contract.py`
- `experiment/CARLA/control/generic_instruction_fsm.py`
- `experiment/CARLA/control/generic_route_pid.py`
- `experiment/CARLA/control/pid_controller.py`
- `experiment/CARLA/scene2_runtime_interface.py`
- `experiment/CARLA/universal_vla_controller.py`
- `scene_understanding/src/control_decision.py`
- 对应速度、计划推进、场景二和返回车道测试文件

精确问题来源、输入与原始输出仍见本目录 `README.md`、`issues.json`、`code_locations.json` 和 `inputs/`；完整修改摘要见 `FIX_REPORT.md`。

## 验证结果

- 证据包完整性检查：`PASS`，原始证据未覆盖。
- Python 语法编译与 `git diff --check`：通过。
- 速度目标、计划推进、等待谓词、TURN 目标、命令派发、FSM、返回车道及计划执行定向回归：`139 passed`。
- `RETURN_WHEN_SAFE` 独立契约探针：通过，输出保留 `ORIGINAL_LANE`，不含固定左右方向。

## 修改边界

本轮没有启动 CARLA 服务，也没有加载 Torch 模型或固定权重，因此不声明三个场景闭环验收通过。当前结论仅覆盖纯 Python 契约、执行状态机、控制接口和定向回归。

后续需要合并黄皓星侧真实谓词及道路目标生产端，由刘旭执行短程集成和独立全程测试；若闭环结果与离线回归不一致，应分别排查感知证据、接口时效、控制饱和与物理反馈，不将保护层介入直接计为模型错误。
