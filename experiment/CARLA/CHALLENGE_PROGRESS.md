# CARLA 仿真控制与统一评测

更新日期：2026-09-21  
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

## 运行

启动 CARLA 0.9.16 后，在仓库根目录使用已安装 CARLA Python API 的环境运行。每次指定新的输出目录：

```text
python experiment/CARLA/run_control_experiment.py basic_voice_control_5km --benchmark-assessment --record-multimodal --duration-s 800 --output-dir outputs/scene1_run
python experiment/CARLA/run_complex_avoidance_town05.py --benchmark-assessment --record-multimodal --duration 0 --output-dir outputs/scene2_run
python experiment/CARLA/run_emergency_response_6km.py --benchmark-assessment --record-multimodal --duration 0 --output-dir outputs/scene3_run
```

`--benchmark-task` 可指定任务编号或 ID，但只筛选评测，不改变驾驶路线或从任务起点开始。完整多模态同步情况见各运行目录的 `multimodal/capture_summary.json`，任务结果见 `benchmark/summary.json`。默认控制器是场景基线；接入模型时需明确选择模型控制入口，不能将基线结果算作模型闭环结果。
