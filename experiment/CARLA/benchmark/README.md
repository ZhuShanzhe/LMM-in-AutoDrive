# 仿真任务目录与离线检查

## 正式运行的评测范围

三个正式运行器均支持`--benchmark-assessment --benchmark-task <任务ID或激活顺序编号>`。
默认all评测全场景；选择单项时只把该项计入任务汇总，必要的前置任务仍独立采集和评测，
保存在summary.json的prerequisite_tasks中，不能通过选择单项跳过前置证据。
该参数不改变路线、指令调度或交通流，也不会在目标完成时自动结束仿真。
它是评测范围选择，不是完整的任意子场景生成器。分段起点跳过前置任务时不能凭空判成功。
无效选择在连接仿真前拒绝；未启用独立评测时不允许指定单项。

## 隔离任务实测

场景三几何预检（需要运行中的CARLA，但不生成车辆/行人或加载模型）：

```text
python -m benchmark preflight --scene scene_3 --host 127.0.0.1 --output outputs/benchmark/scene3_geometry_preflight
```

复用正式路线构建器与逻辑车道适配器，输出route.json和preflight.json；逐事件检查必要车道，
行人按其自身start_s_m检查起终点，而非只检查事件中心里程。端点没有完整跨过自车车道时列为无效。
ready仅表示这些无actor几何检查通过，不证明可步行性、交通安全、动态触发或模型效果。
服务器已有运行actor时拒绝切换地图；预检不改变天气，不声称验证雨夜视觉效果。

场景三在线预检同时保存map_snapshot（OpenDRIVE和有序出生点），以后可以离线重做：

```text
python -m benchmark preflight --scene scene_3 --map-snapshot outputs/benchmark/scene3_geometry_preflight/map_snapshot --output outputs/benchmark/scene3_offline_recheck
```

离线模式不连接服务器，但需要CARLA Python API。校验地图名称、OpenDRIVE和出生点顺序哈希。
只有旧map.xodr不够，因为离线carla.Map不自带服务器推荐出生点；禁止用猜测坐标替代源出生点编号。
快照不含渲染网格、导航网格或信号灯actor，不能用于证明视觉效果、可步行性或信号状态。

场景三`scene3_worker_crossing`已增加派生隔离入口，沿用`benchmark run`参数，需提供
Town05_Opt路线及匹配地图的背景车流配置。入口保留源雨夜天气、3330米横穿锚点和1.8m/s步速，
只构建指定行人的横穿冲突测试，起终点现复用正式逻辑车道适配器；其他施工角色和犹豫行为尚未复现。
这些差异写入fixture_adjustments，不能将此入口当作完整场景三或模型能力验收。
基线先等待观测到实际冲突及持续清空，再恢复行驶；场景仍使用独立TaskOracle评测。
已完成一次45秒CARLA物理短测，判据通过；仍非完整场景/模型验收，详见进展记录。

`runtime.py` 已接通场景一首条调速、左变道与左转任务的独立运行入口：创建背景交通、同步采集快照与
传感器事件、独立判定、保存证据、离线重评和资源清理。当前控制器是 Traffic Manager
测试基线，不调用语音或模型，不代表正式整条场景已经验收。

单步调速入口现也支持路线中段任务：在原激活位置前120米范围内寻找实际地图匹配、
非路口的准备入口，以30 km/h测试基线接近，到原触发位置才设置任务目标速度。
派生路线/触发/结束里程进行同量平移，source_spec及源入口偏移保留。
没有合法入口或依赖前置任务/跨区间约束时拒绝派生，不假装恢复完整运行历史。
准备速度仅用于场景与判据检查，不证明源指令的加速/减速过程语义已经完整覆盖。
中段任务需给足运行时长，尚未到触发点时fixture_exercised为false。

隔离入口统一使用源场景天气，不再对非复合任务一律替换为ClearNoon。
run_result.json的weather记录requested、applied和verified_fields；应用后从world读回核对，
配置缺失或参数不一致时报错。此检查不证明视觉效果、曝光或路面摩擦符合要求。

```text
python -m benchmark run --scene scene_1 --task 1 --host 127.0.0.1 --route <route.json> --traffic-config configs/basic_voice_traffic_preview.json --output outputs/benchmark/speed_fixture --duration-s 10 --seed 42
python -m benchmark run --scene scene_1 --task c04_change_left --host 127.0.0.1 --route <route.json> --traffic-config configs/basic_voice_traffic_preview.json --output outputs/benchmark/lane_fixture --duration-s 10 --seed 42
python -m benchmark run --scene scene_1 --task c07_turn_left --host 127.0.0.1 --route <route.json> --traffic-config configs/basic_voice_traffic_preview.json --output outputs/benchmark/turn_fixture --duration-s 30 --seed 42
```

输出目录必须不存在；服务器存在车辆、行人或传感器时拒绝覆盖。
路线入口必须匹配实际地图的道路和车道，运行记录保留配置、路线哈希以及测试基线标记。
`run_result.json` 是本次运行最终有效性记录；`assessment/task_result.json` 只表示子任务判据，
不能替代整段运行的安全结论。`episode_safety` 统计任务完成后的事件，安全失败或未知时命令不以成功退出。
采集包含截图与逐帧运动数据，不录制完整视频。

变道入口通过 `lane_fixture.py` 在所给路线中寻找连续、合法、同向的相邻车道，
保留原始评测档 `source_spec.json`，局部激活距离改为0，原始距离与入口偏移另行记录。
该入口以30 km/h作为测试基线，检查目标车道周围车辆的间距和相对速度后才请求变道。
这是测试基线的保守触发检查，不是模型决策、安全保证或通用交通规划器。
未触发时 `fixture_exercised=false`；短采集未达到判据标记为不完整，不伪装成模型失败。

2026-09-19 的实测及限制见 [RUNTIME_ACCEPTANCE_20260919.md](RUNTIME_ACCEPTANCE_20260919.md)。
变道实测与重复性核对见 [LANE_ACCEPTANCE_20260919.md](LANE_ACCEPTANCE_20260919.md)。
左转拓扑、事件准备与剩余工作见 [PROGRESS_20260920.md](PROGRESS_20260920.md)。
左转入口使用同一地图的官方有向拓扑和GlobalRoutePlanner构建独立路线，不假定源路线含有所需左转。
`event_fixture.py` 提供角色名唯一绑定与官方斑马线冲突区构建。
`composite_fixture.py` 接通场景二 `s2_t05_cmd_03` 的派生隔离基线，包含真实步行、左变道和慢车超越。
它修正隔离入口的斑马线与角色生命周期，不修改三个正式场景的原始配置；并非模型控制实现。

```text
python -m benchmark preflight --scene scene_2 --host 127.0.0.1 --output outputs/benchmark/scene2_geometry_preflight
```

该命令只生成原配置的实际地图路线及事件几何报告，不运行车辆；输出目录必须不存在，
服务器存在运行actor时拒绝切换地图。目前会报告慢车提前退休和斑马线错配，退出码1表示尚未就绪。
`plan` 同时输出事件角色生命周期冲突；RESOLVED不等同于评测目标已完成。

```text
python -m benchmark run --scene scene_2 --task s2_t05_cmd_03 --host 127.0.0.1 --route outputs/benchmark/scene2_geometry_preflight/route.json --traffic-config configs/basic_voice_traffic_preview.json --output outputs/benchmark/composite_fixture --duration-s 70 --seed 42
```

前述预检保存的路线可用于此派生隔离测试。保留阴天傍晚天气与原三步阈值；
局部入口和步骤位置另行记录。目标慢车初始驻留，待行人几何清空后以20 km/h行驶，
自车基线在变道后以45 km/h追越。任务总时限不变，不以模型或脚本报告的完成状态替代几何判定。

## 同帧真值与采集会话

隔离运行入口额外输出`actor_snapshots/`：逐帧写入自车和绑定任务演员的位姿、速度及缺失角色列表。
正常采集结束标记`CAPTURED`，保存帧数和SHA256；异常收尾标记`INTERRUPTED`。
进程被强制结束时manifest可能停留在`RECORDING`，不能作为完整测试使用。
完成后的独立评测逐行读回快照，使用最终归档的安全事件，不在内存保留整段CARLA WorldSnapshot。
该记录不是CARLA世界恢复点，不包含所有背景演员、传感器或完整物理状态；不能据此宣称支持精确断点续跑。
每帧flush会产生磁盘开销，未做性能验收；统计motion仍在内存缓冲。

新增 `truth_capture.py`、`monitor.py`，供现有runner调用。它们不启动CARLA、不拥有tick、
不控制车辆，只读取传入快照及静态地图，输出 `task_truth.jsonl`、`task_result.json` 和配置哈希。
新增变道入口检查、路线局部投影、角色绑定与异常位移检查、行人足迹/冲突区相交判断。
停车线使用车身前缘，超车净距使用两车边缘，而非仅比较车辆中心点。

```text
python -m benchmark audit-route --route outputs/traffic_preview/route.json --output outputs/benchmark/route_audit.json
python -m benchmark evaluate --spec outputs/task_capture/spec.json --observations outputs/task_capture/task_truth.jsonl --output outputs/task_replay_result.json
```

路线审计输出路口访问、进入/离开道路、朝向变化及可疑坐标跳段；它不证明交通法规许可、
车道数量或这条路线属于指定场景配置。三条正式runner已通过`--benchmark-assessment`接入采集会话。
`safety_events.py` 按事件原始帧归档，停止监听后经过有界静默等待再评测；
这不是传输层确认，不将“某帧没有回调”当作实时安全证明。迟到事件使本次运行无效。
当前事件覆盖碰撞和实线压线，不覆盖红灯等完整违规指标。
完整接口及示例见 [TASK_TRUTH.md](TASK_TRUTH.md)。

## 任务计划与独立评测

计划新增`isolated_adapters`，与`run`使用同一能力注册表，逐任务给出适配器类型、缺失条件和不支持原因。
`adapter_available=true`只表示代码入口存在，不表示地图/演员已准备好或物理测试已通过。
顶层`execution_supported=false`继续表示该计划尚未物化为可直接执行的完整场景。
带前序成功依赖或区间约束的任务不会因为动作简单就跳过历史检查。

新增 `planning.py` 和 `task_oracle.py`。任务计划保留准备距离、随机种子、原始指令、
事件依赖及源文件哈希，不修改原场景配置，也不把前置事件伪造成已完成。
独立评测支持持续调速、合法目标车道稳定、路口转向、行人避让、绑定目标超车及顺序组合。
详细字段和适用边界见 [TASK_TRUTH.md](TASK_TRUTH.md)。

```text
python -m benchmark plan --scene scene_2 --task 3 --seed 42 --output outputs/benchmark/scene2_task3_plan.json
python -m benchmark evaluate --spec configs/benchmark/s2_t05_cmd_03.json --observations outputs/task_truth.jsonl --output outputs/task_result.json
```

`evaluate` 读取仿真侧独立逐帧真值，输出任务状态、步骤证据、原因及输入哈希。
退出码0表示该任务成功，1表示失败/无效/未完成，2表示文件或配置错误。
示例观测文件需要实际采集器产生，目前不会在缺文件时伪造输入。

38个显式评测档位于 `configs/benchmark/`；计划仅绑定哈希匹配的档位，缺失或不匹配项仍列入清单。
通用计划仍保持 `execution_supported=false`：尚不支持所有任务的自动隔离场景构建；正式入口已接通独立评测。
已开放的隔离调速、左变道和左转测试不改变此标志；后续扩展行人和超车任务的真实入口准备。

统一读取三个现有场景配置，不加载CARLA或模型。保留原始指令、结构化步骤、事件条件和源文件SHA256，提供稳定任务ID与按激活距离排列的编号。

## 已完成

- 场景一15条、场景二15条、场景三8条指令的统一目录。
- 任务ID和数字编号选择、UTF-8 JSON导出。
- 重复ID、非有限数值、越界触发距离、无效事件依赖与指令引用检查。
- 报告任务重叠、源编号差异、缺失结束窗口、独立判据及入口fixture。
- 区分配置合法与具备正式benchmark条件。当前 `benchmark_ready=false`。

## 使用

在 `experiment/CARLA` 目录执行：

```text
python -m benchmark list --scene scene_1
python -m benchmark list --scene scene_2 --task 3
python -m benchmark manifest --scene scene_3 --output outputs/benchmark/scene_3.json
python -m benchmark validate --scene scene_2
python -m benchmark validate --scene scene_2 --require-ready
```

`--task` 可指定 `all`、激活顺序编号或稳定ID。场景一第4项是 `c08_keep_50`，因为源配置和原运行策略采用激活距离排序；原始编号可通过 `source_order` 查询。

退出码：0表示目录导出或配置校验通过；1表示配置规则未达标，或指定 `--require-ready` 时尚未具备正式评测条件；2表示配置读取、格式、参数选择或输出错误。

导出内容是场景管理与评测配置，不是模型输入。`source_command` 里的预设动作、目标参数和事件真值不得绕过指令解析直接输入模型。

## 后续接入

### 超车返回走廊离线准备

场景二cmd07现在有派生隔离运行入口（尚未完成物理验收）：

```text
python -m benchmark run --scene scene_2 --task s2_t05_cmd_07 --route outputs/benchmark/scene2_preflight_crosswalk_fixed_20260920/route.json --traffic-config configs/basic_voice_traffic_preview.json --duration-s 60 --seed 42 --output outputs/benchmark/cyclist_isolated
```

该入口保留源动作顺序/阈值及14 km/h骑行者速度，局部30米触发、55米放置骑行者。
使用CARLA Traffic Manager脚本基线，不执行模型推理；背景车流保持启用。
脚本执行减速、等待左侧安全间隙、左变道、超越、等待返回间隙、返回原车道。
到走廊末端刹停而不是任由TM驶出验证范围；脚本阶段完成不等于独立TaskOracle成功。
源触发点、局部起点和调整清单写入run_result，原配置不被修改，不能用于宣称原场景任务通过。
当前短走廊是否足够、骑行者物理稳定性以及实际背景车流仍需短测确认。

`overtake-geometry`使用已保存的OpenDRIVE及路线，不连接CARLA服务器。
检查左侧借道和右侧返回的许可、同向相邻车道、路口、采样间隔、走廊长度和路线匹配。
输出源配置、路线和地图哈希，保留失败原因；失败退出1，不生成可执行成功标记。

```text
python -m benchmark overtake-geometry --scene scene_2 --task s2_t05_cmd_07 --route outputs/benchmark/scene2_preflight_crosswalk_fixed_20260920/route.json --map-snapshot outputs/benchmark/scene3_geometry_live_20260920/map_snapshot --output outputs/benchmark/overtake_geometry.json
```

当前几何准备检查相邻采样点的唯一前向连接，允许弯道及连续道路编号变化，仍要求非路口并允许往返变道。
搜索触发点前后500米，不把相邻坐标近似当作车道连续。
场景二骑行者任务附近找到约121米走廊，但源触发点之后只剩约28米；280米走廊尚未找到。
输出同时记录源触发点是否位于走廊内、触发后的剩余距离。几何可用不代表整个任务可以完成。
`--corridor-length-m`仅控制预检长度，不修改源任务的完成判据。

`run` 目前支持单步调速、左变道、左转及场景二上述三步复合任务的隔离几何测试，其余任务选择只导出配置。任务结束位置缺失时保留null，不以“下一条指令触发”推定成功。

后续使用已有路线审计、角色绑定和同帧采集接口构建更多入口fixture。重叠任务需要明确并发与优先级；前置事件需要真实准备过程，不能直接伪造RESOLVED。评测档覆盖不代表完整指令语义或真实运行已经通过。

测试：`python -m pytest tests/test_benchmark_catalog.py tests/test_sensor_replay.py -q`。
# 独立评测退出码

三个正式运行器启用 `--benchmark-assessment` 后，评测收尾输出
`benchmark/run_outcome.json`，并将结果传递给进程退出码：

| 退出码 | 含义 |
| --- | --- |
| 0 | 本次记录的任务检查通过，且没有记录到已覆盖的安全违规 |
| 2 | 任务失败、超时，或整段记录出现碰撞/已覆盖违规 |
| 3 | 证据不完整，例如缺少评测配置、未触发任务、场景无效或安全记录不确定 |
| 4 | 评测器异常或结果无法保存 |

退出码0不代表完整路线、所有交通法规或模型指标全部验收通过；
结果文件固定注明 `full_benchmark_acceptance=false`。
分段起点之前的NOT_RUN不计为已完成；全部任务均NOT_RUN仍返回3。
后续碰撞不能被较早的任务SUCCESS掩盖。已有运行异常/用户中断不被成功评测覆盖。
不启用独立评测时，保留运行器原有退出行为。
