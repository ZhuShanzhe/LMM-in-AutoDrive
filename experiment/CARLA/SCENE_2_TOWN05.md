# 场景二：Town05 复杂避障场景

## 设计目标

该实现使用 CARLA 官方 `Town05_Opt`，替代缺少完整道路资产的手工
OpenDRIVE 地图。场景面向复杂组合指令演示，包含城市主干道、连续路口、
阴天傍晚低照度、混合交通流及可重复触发的特殊交通参与者。该入口是提交视频
应使用的场景二入口；旧的手工 OpenDRIVE 入口仅保留作合同与真值回归。

核心文件：

- `configs/scene_2_town05_runtime.json`：路线、天气、交通流、事件和 15 条
  演示指令。
- `scenarios/complex/town05_scene2.py`：路线进度、交通流和确定性事件。
- `run_complex_avoidance_town05.py`：场景生命周期、预览控制、录像和日志。
- `tests/test_scene2_town05.py`：配置契约和重复路线进度回归测试。

本场景独立于原手工地图实现，两者可以并行保留。

## 路线

路线由 CARLA `GlobalRoutePlanner` 在 Town05 的出生点 239 与经审计候选点 284
之间往返生成，
裁剪后的总长度为 8 km。路线经过多车道城市道路、十字路口、弯道和高架附近
道路，可用于展示跟车、加减速、变道、转弯和避让。

`RouteProgressTracker` 只在当前索引附近搜索最近路径点。即使 8 km 路线多次
经过相同物理道路，进度也不会跳到后续循环。

## 天气

天气按比赛场景二配置为阴天傍晚低照度：

- 80% 云量、太阳高度角 5°、20% 轻雾和 10% 湿路面；
- 低照度是竞赛条件，不再使用白天高太阳高度角替代；
- 加载 `Town05_Opt` 的全部可选地图层，使用原生街景和原生车道标线，不再
  用纯 OpenDRIVE 道路或调试线模拟街景。

## 交通流

背景交通基于 CARLA Traffic Manager：

- 固定随机种子，保证重复运行具有相同的初始交通分布；
- 默认生成 70 辆车辆，其中包含 3 辆公交车；
- 默认生成 24 名环境行人；
- 行人优先生成在录制路线两侧的人行道，避免随机落到镜头外的城区；
- 优先在自车起点和路线附近生成车辆，保持城市道路的可见车流密度；
- 禁用混合物理和运行时重生，避免车辆在镜头内突然出现或瞬移；
- 使用合理跟车距离、轻微速度差和低概率随机变道；
- 车辆外观、驾驶员和车型按固定种子多样化。

交通流由 Traffic Manager 管理，特殊事件角色由场景模块单独管理，避免两套
逻辑同时控制同一 actor。

## 可复现的小场景变体

| 事件 | 锚点 | 触发点 | 行为 |
| --- | ---: | ---: | --- |
| 慢车 | 500 m | 接近场景即存在 | 以 20 km/h 行驶并保持车道 |
| 行人过街 | 693 m | 620 m | 从道路一侧以 1.45 m/s 横穿至另一侧 |
| 公交站乘客 | 1045 m | 930 m | 公交车靠边停车，三名乘客执行上下车方向移动 |
| 慢速自行车 | 1150 m | 850 m | 触发前隐藏，触发后以 14 km/h 沿车道行驶 |

四类事件各提供三种微场景。`--variant-index 0/1/2` 分别选择一组可复现的
慢车速度、行人方向/停走行为、公交乘客速度和自行车速度。这样同一套感知与
控制链路可以在不同动态条件下复测，而不是只记住一条固定时间线。正式对比时
应至少跑完三个 variant，并分别保存结果，不能挑选单次最好结果。

特殊行人不再只尝试一个出生坐标。运行器会沿人行道方向平移、采用三档安全
高度和多个行人蓝图重试；仍被 Town05 静态碰撞体占用时，才回退到 28 m 内的
导航网格点。实际尝试次数和回退来源写入 `summary.json` 的
`spawn_diagnostics`，必需角色全部失败仍会明确报错，不会把缺行人的视频判为成功。

事件状态统一为 `STAGED`、`ACTIVE`、`RESOLVED`。状态变化写入
`events.jsonl`，避免仅凭视频画面推断事件是否触发。

## 输出

一次运行输出：

- `runtime.jsonl`：路线进度、自车速度、周围车辆/行人数量和安全事件；
- `events.jsonl`：特殊事件状态变化；
- `commands.jsonl`：15 条指令的触发帧和路线位置；
- `summary.json`：路线、交通流、事件、碰撞和压线汇总；
- `selected_variants.json`：本轮四类微场景的确定性选择；
- `frame_ground_truth.jsonl`：启用 `--record-ground-truth` 后输出同帧 CARLA
  Actor、风险标签及允许/禁止控制动作；
- `driving_intent.jsonl`：15 条组合语音指令的有序结构化步骤；
- `world_state.jsonl` 和 `multimodal_frame_bundle.jsonl`：四路 RGB、LiDAR 与
  车辆状态按严格相同 `simulation_frame` 联结；
- `interface_manifest.json`：ASR、VLA、安全门和控制决策边界；
- `route_command_audit.json`：结构化左转、右转、直行步骤与实际 GRP 路线的
  预检结果；
- H.264 MP4：无须先保存大量逐帧图片。

`--start-progress-m` 仅用于针对某个事件做短时诊断。正式 8 km 运行必须从
0 m 开始。

## 运行与字体

```bash
python experiment/CARLA/run_complex_avoidance_town05.py --validate-only

python experiment/CARLA/run_complex_avoidance_town05.py \
  --route-preflight --output-dir /tmp/scene2_route_preflight

python experiment/CARLA/run_complex_avoidance_town05.py \
  --host 127.0.0.1 --port 2000 --duration 0 \
  --variant-index 0 \
  --video-overlay \
  --record-ground-truth --ground-truth-every-n 1 \
  --video-output experiment/CARLA/outputs/scene2_town05/scene2_submit.mp4
```

正式 8 km VLA 运行必须使用外部控制并启用比赛门：

```bash
python experiment/CARLA/run_complex_avoidance_town05.py \
  --competition-run --external-ego-control --duration 0 \
  --output-dir experiment/CARLA/outputs/scene2_town05_competition \
  --video-output experiment/CARLA/outputs/scene2_town05_competition/scene2.mp4
```

`--competition-run` 会自动启用四视角 RGB、LiDAR、真值和 HUD 记录。传感器
回调只有在五类数据都已原子落盘后才跨越同帧屏障并发布 bundle；任何超时会
使正式运行失败，不允许用前后相邻帧补齐。该模式还要求从 0 m 开始、完整
8 km、外部控制，并在生成 actor 前拒绝路线与转向指令不一致的配置。

预览默认使用 CARLA `BehaviorAgent` 以便录制和检查场景，它不是比赛 VLA 的
模型输出，`summary.json` 会将其标为 `competition_metric_eligible: false`。
语义对齐和控制准确率必须把模型/控制器的同帧预测作为独立文件，再与
`frame_ground_truth.jsonl` 做影子评测；不能用事件调度规则给自己打分。

若 Python 能导入 `carla` 但不能导入导航代理，请设置 `CARLA_ROOT` 指向解压后的
CARLA 根目录，并在当前环境安装 `shapely` 与 `networkx`。

HUD 会依次查找 Noto CJK、文泉驿和常见中文字体。服务器缺少中文字体时不会再
显示方框乱码，而是显示配置内的等价英文短句。若需强制显示中文，可安装
`fonts-wqy-zenhei`，或通过 `--hud-font /path/to/font.ttc` / 环境变量
`CARLA_HUD_FONT` 指定字体文件。

## 当前验证

- 配置通过契约校验：Town05、8 km、15 条指令和四类特殊事件齐全；
- 多模态证据要求四视角 RGB、LiDAR、WorldState 使用完全相同的 CARLA 帧号；
- 正式模式要求多模态 bundle 完整率 100%，同时保留不完整帧诊断，绝不静默
  改写为 COMPLETE；
- 路线累计航向变化写入 `summary.json`，可审计弯道行驶而非只看直线里程；
- 70 辆 NPC 可稳定生成，附近 85 m 内可观测车辆数量明显高于旧手工场景；
- 行人过街、公交站乘客和慢速自行车均可按路线进度触发；
- 短时事件测试未发生碰撞，录像无丢帧；
- 仍需在最终控制链路接入后，以完整 8 km 运行结果作为驾驶策略验收依据。
