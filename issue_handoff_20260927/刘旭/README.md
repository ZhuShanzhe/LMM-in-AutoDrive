# 刘旭：任务与地图一致性、独立评测边界和闭环验证入口

本目录用于定位和修改，不表示问题已修复。代码在仓库原模块中修改；source_snapshot仅是带行号的取证副本。

## 快速开始

从仓库根目录运行：

```bash
python issue_handoff_20260927/check_package.py --owner 刘旭
PYTHONPATH=.:experiment/CARLA python -m pytest -q experiment/CARLA/tests/test_lane_continuity.py experiment/CARLA/tests/test_task_oracle.py experiment/CARLA/tests/test_scene2_first_task_contract.py experiment/CARLA/tests/test_task_geometry_binding.py
```

第一条只校验证据并展示问题基线，不将已知问题算作通过。第二条是已有回归测试；它们通过不意味着本目录的新验收条件已满足。

共用输入位于 `../_common/`：三场景摘录、环境、完整场景二路线/地图、配置与额外采集。传感器压缩包解压到任意新空目录后可用原有ModelRigReplayDataset读取。

## 问题与任务

### X1 [P0] 场景二任务13/14/15几何绑定未通过

**证据：** 静态真实地图绑定发现任务13、14、15为MISMATCH，而运行器的route_command_audit报告mismatch_count=0。两种校验口径不一致。

**修改任务：** 对照完整路线、激活窗口、指令和独立profile核验；保证任务在实际道路上可执行，统一预检与独立评测的绑定口径。

**验收：** 15条均给出一致的拓扑检查结果；无法执行的任务启动前拒绝并定位，不得仅改competition_ready标签。

### X2 [P0] 车道证明范围耗尽后的评测归因

**证据：** 场景二首任务实测SCENE_INVALID，原因speed lane corridor does not cover current progress。车道证明仅覆盖可确认前缀，而速度步骤未完成便继续行驶。

**修改任务：** 区分真实地图/夹具无效与车辆未完成任务导致越过有效区间；检查trace_lane_corridor、EpisodeAssessment和TaskOracle边界。结合W1定位，不盲目延长走廊或放宽判定。

**验收：** 有效范围、路口分岔、未完成驶出、真实坏夹具分别有子单元用例；成功必须有完整证据；失败与场景无效原因清晰。

### X3 [P1] 分段到全程的可追溯复测

**证据：** 本包场景一40秒、场景二90秒、场景三60秒以及20秒额外采集均为短程诊断，退出3，不能代表三场景全程通过。

**修改任务：** 先按问题包验证已修改模块，再运行真实场景；保留代码/配置/权重哈希、命令、seed、任务完成及违规指标。不要将配置计划模式当作自然语言解析测试。

**验收：** 全程报告包含每条任务状态、碰撞/违规、超时原因和有效时延样本；未覆盖或无效指标明确列出，不用0填充未知。

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

## 无仿真服务的地图复现

安装项目carla 0.9.16 Python包后，直接读取包内OpenDRIVE，不需要GPU和CARLA进程：

```bash
python issue_handoff_20260927/reproduce_geometry.py
```

已复现任务13、14、15的MISMATCH，结果见`inputs/offline_geometry_reproduced.json`。BOUND只代表指定几何校验通过，不代表驾驶成功。
