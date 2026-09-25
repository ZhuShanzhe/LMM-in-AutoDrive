# CARLA 仿真控制与统一评测

更新日期：2026-09-22  
开发分支：`lx-challenge`

## 已实现功能

- **场景管理**：统一基础操纵、复杂避障、应急驾驶三个场景，配置38项指令任务，支持按场景和任务编号选择、生成运行计划。
- **交通与事件**：支持背景车辆生成、补充和回收，以及行人横穿、慢车、骑行者、公交站乘客和施工事件。
- **任务分段**：保留完整场景运行入口，提供调速、变道、转弯、行人避让和部分组合任务的独立运行入口。
- **多模态采集**：同步记录多视角图像、LiDAR和车辆状态，提供传感器标定及模型输入帧关联。
- **同源回放**：支持录制数据的离线回放和重复运行，记录随机种子与运行配置。
- **控制与反馈**：对接车辆控制接口，记录控制指令、执行反馈及异常状态。
- **任务评价**：根据仿真数据判断速度调整、车道保持、变道、转弯、行人避让、超车及组合动作顺序，输出成功、失败、超时等状态。
- **日志与报告**：记录演员轨迹、交通分布、碰撞及实线违规事件，统计运行延迟和资源占用，输出JSON与Markdown报告。
- **运行管理**：支持中断数据保留、磁盘容量保护，以及传感器、车辆和仿真设置的清理恢复。

## 使用方式

在已有CARLA Python环境中执行，工作目录为：

```powershell
cd D:\CARLA\LMM-in-AutoDrive-main-inspect\experiment\CARLA
```

### 查看任务与生成计划

无需启动CARLA。场景名称为`scene_1`、`scene_2`、`scene_3`，任务可填写编号、任务ID或`all`。

```powershell
python -m benchmark list --scene scene_1
python -m benchmark list --scene scene_2 --task 3
python -m benchmark plan --scene scene_2 --task 3 --seed 42 --output outputs/scene2_task3_plan.json
```

计划中的`isolated_adapters`列出可用的独立运行入口；生成计划不会启动仿真。

### 运行单项任务

先启动CARLA 0.9.16，默认连接`127.0.0.1:2000`。使用空闲服务器，每次指定新的输出目录。

```powershell
python -m benchmark preflight --scene scene_2 --host 127.0.0.1 --output outputs/scene2_route
python -m benchmark run --scene scene_2 --task s2_t05_cmd_03 --host 127.0.0.1 --route outputs/scene2_route/route.json --traffic-config configs/basic_voice_traffic_preview.json --duration-s 70 --seed 42 --output outputs/scene2_task3
```

示例使用脚本控制基线和背景车流，运行“行人避让、左变道、超越慢车”。`--duration-s`指定仿真时长，`--seed`指定随机种子。

### 运行完整场景

以下使用场景二默认控制入口。`--duration 0`表示运行至8公里路线结束；使用模型时需额外提供模型配置与权重参数。

```powershell
python run_complex_avoidance_town05.py --config configs/scene_2_town05_runtime.json --duration 0 --benchmark-assessment --output-dir outputs/scene2_full
```

添加`--benchmark-task s2_t05_cmd_03`可选择汇总的任务，但不改变完整行驶路线。

### 查看结果与汇总报告

- 单项结果：`outputs/scene2_task3/run_result.json`。
- 逐帧运动与演员状态：单项输出目录下的`motion.jsonl`和`actor_snapshots/`。
- 完整场景任务汇总：`outputs/scene2_full/benchmark/summary.json`。

```powershell
python -m benchmark.report outputs/scene2_full/benchmark/summary.json --output outputs/scene2_report
```

输出`report.md`和`report.json`；可传入多个场景的结果文件进行汇总。

### 核对三场景同源数据

三个场景先分别完成模型传感器录制（`--vla-record-sensors`）和独立任务评测
（`--benchmark-assessment`），再对实际采集的目录运行：

```powershell
python -m evaluation.challenge_capture_audit --scene-1 outputs/scene1_full --scene-2 outputs/scene2_full --scene-3 outputs/scene3_full --frames 1000 --output outputs/challenge_1000_frames.json
```

审计要求每场景有足够的模型决策帧、同一传感器帧的四视角与 LiDAR，
且每个决策帧均有同帧、同时间的独立仿真真值；
输出三个场景的采集指纹、评测状态及均匀抽样的帧索引。未采集到的数据不会补造。
索引用于指标抽样；有状态模型必须从各自采集的完整原序列回放，不能直接跳帧推理。
采集真值只供评测，不传入模型适配器。该审计不代替真实闭环或任务成功判定。

## 详细说明

- [任务管理与运行参数](benchmark/README.md)
- [多模态采集与同源回放](evaluation/REPLAY_CONTRACT.md)
- [指令调度](control/COMMAND_DISPATCH.md)
- [传感器输入配置](control/SENSOR_CONTRACT.md)
