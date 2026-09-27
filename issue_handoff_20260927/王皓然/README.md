# 王皓然：纵向目标契约、计划推进、等待恢复与回原车道

本目录用于定位和修改，不表示问题已修复。代码在仓库原模块中修改；source_snapshot仅是带行号的取证副本。

## 快速开始

从仓库根目录运行：

```bash
python issue_handoff_20260927/check_package.py --owner 王皓然
PYTHONPATH=.:experiment/CARLA python -m pytest -q lightweight_vla_adapter/tests/test_driving_plan_runtime.py experiment/CARLA/tests/test_cruise_speed_contract.py experiment/CARLA/tests/test_dispatch_wait_budget.py experiment/CARLA/tests/test_return_original_lane.py
```

第一条只校验证据并展示问题基线，不将已知问题算作通过。第二条是已有回归测试；它们通过不意味着本目录的新验收条件已满足。

共用输入位于 `../_common/`：三场景摘录、环境、完整场景二路线/地图、配置与额外采集。传感器压缩包解压到任意新空目录后可用原有ModelRigReplayDataset读取。

## 问题与任务

### W1 [P0] 45km/h目标与动作序列推进不一致

**证据：** 场景二90秒行驶361.347米，但1777帧停留在SET_SPEED，最高实测42.688km/h，后续TURN尚未激活；20秒采集的独立重复中最高44.161km/h，仍未持续满足完成条件。

**修改任务：** 逐层核对指令目标、VLA建议速度、风险上限、道路限速、PID输入和物理反馈；将明确指令目标与安全/曲率上限组成统一执行契约，并处理目标受限时的推进或显式失败。不可无条件把模型速度替换为45。

**验收：** 无风险可达目标能持续到达；风险/限速优先；不可达目标有明确状态且不死锁；后续路口动作不因无期限等待前一步而默默错过。

### W2 [P0] 等待条件与TURN接口接入

**证据：** 条件探针提供低风险但没有行人/上下客完成证据，因此等待不完成本身是正确的；缺口是缺少显式谓词生产到消费的路径。TURN探针显示目标坐标契约需一致。

**修改任务：** 接入黄皓星的有效谓词和道路目标；保留证据新鲜度、持续确认、风险阻断与恢复逻辑；明确真实完成反馈而非仅发出动作即完成。

**验收：** 证据缺失保持等待；证据满足后按序推进；风险再次出现及时阻断；旧命令/旧step反馈不能完成当前步骤。

### W3 [P1] 回原车道不能固定为右侧

**证据：** RETURN_WHEN_SAFE适配器当前无条件映射RIGHT，缺少起始车道语义。

**修改任务：** 传递原车道引用，结合车道拓扑及换道事件记录解析返回方向；返回是任务语义，不是固定左右动作。

**验收：** 左超车后右返、右移后左返、已在原车道、原车道不可达及风险阻断恢复均有测试；禁止固定RIGHT或编造合法车道。

## 修改边界

精确文件、函数及行号见 `code_locations.json`；依赖另一位同学的接口先对齐字段，不同时改同一个文件。输入缺少真值的现象不得直接定性为感知误检。

提交时在本目录补充修改说明和测试结果，实际代码改原模块；保留原始证据，不覆盖旧日志。

## 本人输入包

`inputs/`已按本人问题保存精简输入与实际输出，`evidence_index.json`列出共享证据位置。完整日志片段以`.json.gz`压缩保存，可用Python的gzip与json读取。

通用接口探针（不启动CARLA、不加载权重）：

```bash
python issue_handoff_20260927/check_package.py --probe-contracts
```

传感器原始输入解码与字节完整性检查：

```bash
python issue_handoff_20260927/check_package.py --decode-capture
```
