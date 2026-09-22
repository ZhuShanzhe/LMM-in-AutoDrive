# 独立任务评测接口

## 边界

`task_oracle.py` 只评测，不生成驾驶动作。输入由仿真侧真值观测器生成，
不得来自模型的成功标志、DrivingIntent、风险等级或场景脚本的 RESOLVED 状态。
已提供离线评测器、任务计划、同帧采集适配器和采集会话，尚未将三个运行入口接入此接口。

状态：WAITING、RUNNING、SUCCESS、FAILURE、SCENE_INVALID、TIMEOUT。
任务开始后只按仿真时间计时；缺帧、时间倒退、目标消失、目标替换、入口条件错误归为场景无效。
碰撞、违规或占用冲突区时越过停车线归为失败。数据流提前结束不冒充正常超时。
终态不可覆盖，单任务成功后仍需独立的全程安全监测，不能据此证明后续路线无碰撞。

## 观测字段

每行一个JSON，所有动态字段属于同一CARLA快照。应由唯一tick所有者在快照完成后记录，
不能混用读取不同帧得到的actor位置。`source` 是接口声明，不是对数据真实性的密码学证明。

```json
{
  "schema_version": "task_truth/1.0",
  "source": "simulator_truth",
  "task_id": "c04_change_left",
  "source_sha256": "与场景配置对应的SHA256",
  "frame": 1234,
  "sim_time_s": 61.7,
  "scenario_valid": true,
  "ego": {
    "route_s_m": 1300,
    "speed_kmh": 40,
    "lane_key": "entry_lane",
    "road_key": "entry_road",
    "in_junction": false,
    "lateral_error_m": 0.1,
    "heading_error_deg": 1,
    "yaw_deg": 0,
    "route_corridor_id": "corridor_with_lap_id",
    "half_length_m": 2.3
  },
  "safety": {"collisions": 0, "violations": 0},
  "actors": {},
  "fixture": {
    "steps": {
      "0": {
        "direction": "LEFT",
        "legal": true,
        "entry_lane_key": "entry_lane",
        "target_lane_key": "audited_left_neighbor"
      }
    }
  }
}
```

示例中的车道、道路名称仅说明字段，运行时必须用地图审计得到的稳定拓扑键替换，
不能直接当作现有地图的可运行fixture。评测器会检查fixture在任务内不变。

- `route_s_m`：同一有向路线、同一圈次的投影里程，不能用全图最近点混淆不同圈次或岔路。
- `lateral_error_m`、`heading_error_deg`：相对当前实际车道中心线及切线，不是相对模型预测轨迹。
- `yaw_deg`：CARLA朝向角，增大为右转；跨±180度规范化后判断转向。
- `safety`：从本测试片段准备开始累计的碰撞/违规事件数，开始为零，不得中途清零；
  重复接触事件的合并和违规定义需要采集器统一，评测器不通过调整分母掩盖违规。
- `scenario_valid`：由独立场景有效性检查生成，例如角色确实存在、入口可达、传感器同帧。
  不能由策略自己声明通过。

## 角色与判据

`actors` 按固定角色名索引，均需 `actor_id`、`alive`；检测到传送时设置 `teleported=true`。
同一角色不能悄悄绑定新actor。运行接入仍需完成角色生命周期及传送检测，不能依靠缺省false证明没有传送。

| 判据 | 必要证据 |
|---|---|
| speed | 实测速度连续落入容差，保持车道任务还需车道不变；抖出容差重新计时 |
| lane_change | 从入口车道出发，到达指定合法相邻车道，横向误差及朝向误差持续合格 |
| turn | 从入口道路出发，实际经过路口，按正确方向改变足够角度，稳定进入指定不同出口道路 |
| yield_pedestrian | 观察到同一行人在冲突区，自车停车或明确减速，未越过停车线；行人真正清空后持续确认 |
| overtake | 同一目标从前方变为后方，扣除两车半长后后向净距合格，且处于相同有向路线走廊 |
| progress | 当前步骤开始后确实前进指定距离，不以指令宣布代替行驶 |

转向fixture另需 `entry_road_key`、`exit_road_key`、`direction`、`legal`。
行人fixture需 `stop_line_route_s_m`；行人观测需 `in_conflict_zone`，由足迹与预先定义冲突区的几何相交得到。
车辆目标需 `route_s_m`、`half_length_m`、`route_corridor_id`。
所有步骤严格依次完成，每一步保留证据帧和仿真时间。尚未提供公交停靠区域、掉头拓扑审计等专用判据。

## 当前配置与限制

`configs/benchmark/` 提供4个首批评测档：45km/h巡航、左变道、路口左转、行人避让后变道超车。
它们绑定现有场景配置SHA256；源配置改动后不得静默继续沿用。
容差、保持时间和减速幅度是待实测校准的工程阈值，不是赛事官方定义，也不是最终冻结评分标准。
场景二档允许减速避让，不强制停车；还需要核查慢车和行人在完整组合任务期间均可绑定和观测。

目前既未证明现有场景几何满足fixture，也未把历史视频重新判为通过。
当前证据是合成逐帧正反例测试，正式模型成功率需运行接口接通后另测。

## Runner接入方式

`ActorBinding.from_actor()` 只从actor读取静态ID与包围盒。
`SnapshotTruthCollector.collect()` 的动态位置、朝向和速度均从传入的 `WorldSnapshot.find(actor_id)` 获取，
没有另行读取live actor的动态状态，不推进仿真，不把观测传给模型。

runner准备好实际角色、静态路线、初始投影里程和审核后的fixture后可调用：

```python
from benchmark.truth_capture import ActorBinding, SnapshotTruthCollector
from benchmark.monitor import TaskMonitor

collector = SnapshotTruthCollector(
    spec=task_spec,
    route=route_points,
    world_map=world.get_map(),
    ego=ActorBinding.from_actor(ego_actor, entry_route_s_m),
    roles=bound_task_roles,
    fixture=audited_fixture,
    corridor_id=route_and_lap_id,
)
with TaskMonitor(collector, fresh_output_directory) as monitor:
    # The existing runner retains tick ownership.
    for snapshot, finalized_safety, finalized_validity in runner_observations:
        task_feedback = monitor.observe(snapshot, finalized_safety, finalized_validity)
```

这是接入接口示例，不是包含场景生成的独立启动脚本；示例变量需来自实际runner。
`bound_task_roles` 是角色名到 `ActorBinding` 的映射，所有必需目标必须准备并绑定，不能在缺角色时忽略判据。
`prepare_lane_fixture(map, entry_location, direction)` 检查入口处同向相邻行车道和变道许可，
不证明整个变道过程中所有位置均允许变道。

安全包需包含 `frame`、`complete=true`、`collisions`、`violations`；
有效性包需包含同一 `frame`、`complete=true`、`valid` 及无效时的 `reason`。
`complete` 必须由runner的事件收集同步过程产生，不能仅在 `world.tick()` 返回后立即硬编码为true。
当前旧碰撞收集器没有提供这一完成契约，需处理延迟回调和对应帧归属后才能正式接入。

每次会话使用新目录，拒绝覆盖旧运行结果。保存 `spec.json`、`fixture.json`、`route.json`、
`manifest.json`、逐帧观测和结果，哈希按实际文件字节计算。
在线逐帧评测与导出后离线 `evaluate` 的状态、证据和哈希已做一致性测试。
`evaluation_wall_ms` 只表示真值采集适配及评测耗时，不是模型推理或端到端延时。

## 几何与适用限制

- 路线投影使用局部里程窗口，保留倒车时的真实投影变化；近距离多圈歧义拒绝判定，不强行跳到其他圈次。
- 行人冲突区需提供有序凸多边形；使用带局部包围盒偏移/朝向的地面足迹判断，接触边界仍算冲突。
- 车辆前后缘按局部路线切线投影，考虑车身朝向及包围盒偏移；急弯上的投影仍是局部近似。
- 地面足迹忽略pitch/roll，仅适用于当前普通道路的地面冲突区；立体交叉与坡道需要高度分层。
- 位移检查拒绝明显超过速度积分加3米余量的跳变，不能保证检测所有小幅传送。
- 自车明显偏离路线归为行为失败，而目标丢失、快照缺帧等归为场景/采集无效；两类不能混为模型失败。
- 当前没有自动开放单任务启动，也没有新增仿真运行成绩。验收证据仍为接口测试和已有路线文件的离线审计。
