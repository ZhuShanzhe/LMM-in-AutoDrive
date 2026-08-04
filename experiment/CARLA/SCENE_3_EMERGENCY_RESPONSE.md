# 场景三：Town05_Opt 雨夜应急驾驶（6 km）

场景三已重构为 CARLA 官方 `Town05_Opt`。运行器不再加载自建
`VLA_EmergencyRoad_6km.xodr`，而是加载官方建筑、绿化、路灯和道路资产，并在官方道路拓扑上构造累计里程不少于 6000 m、完成点严格为 6000 m 的连续路线。

## 场景合同

- 环境：雨天夜间、低光、100% 湿路、路面反光、雨雾遮挡；
- 道路：Town05 城市快速路走廊，包含施工预警、锥桶渐变、右车道封闭和车道收窄；
- 动态干扰：左侧车辆突发加塞、施工人员临时横穿、维修车阻塞及目标车道安全间隙释放；
- 输入：前/左/右/后四路低信噪比 RGB、逐帧动态车辆状态、带轻微雨声/座舱/道路噪声合同的中文模糊或应急指令；
- 完成条件：累计路线进度 6000 m、7 个事件全部解除、零碰撞、零非法车道侵入、四路相机均有输出；
- 评测门槛：应急响应时延不高于 120 ms，多模态语义对齐准确率不低于 97%。这两个指标由外部 VLA/评测链实测，场景运行器只提供同步帧、事件和真值数据，不伪造达标结果。

## 实现文件

| 文件 | 作用 |
|---|---|
| `configs/scene_3_emergency_6km_runtime.json` | 官方地图、路线、雨夜、湿滑、传感器、语音、事件和指标合同 |
| `scene3_town05_route.py` | 6 km 路线生成、累计进度跟踪和逻辑车道到 Town05 实际车道的适配 |
| `emergency_scene_3_events.py` | 加塞、施工、锥桶、横穿行人和阻塞车辆 Actor 行为 |
| `run_emergency_response_6km.py` | 正式运行器，实际调用天气、灯光、摩擦触发器、相机和车辆状态记录 API |
| `scene3_video_preview.py` | 完整 6 km 四视角与第三人称 H.264 预览入口 |
| `tests/test_scene3_town05_rebuild.py` | 不依赖 CARLA 安装的重构合同测试 |

旧的 XODR 生成器和地图文件仅保留为历史兼容材料，场景三配置、正式运行器和预览入口均不再引用它们。

## 6 km 路线与事件

路线从 Town05 官方 spawn point 239 出发，以 spawn point 289 为折返点锚点，复用官方拓扑规划器生成双向走廊段，并按官方道路中心线累计重复拼接到至少 6000 m。启动时会校验所有事件锚点所需车道；施工区每 50 m 校验左、中、右三条可行驶车道，缺失时在 Actor 生成前直接失败。

场景内部继续使用稳定逻辑车道编号 `-1/-2/-3/-4/-5` 表示左/中/右/路肩/人行道，适配器在每个路线进度点解析 Town05 的真实 `road_id/lane_id`。因此原有事件代码可复用，同时不再错误依赖自建直路的固定道路编号。

| 顺序 | 路线位置 | 场景事件 |
|---|---:|---|
| 1 | 980–1280 m | 突发车辆加塞，紧急减速避让 |
| 2 | 1450–1850 m | 施工提前警示 |
| 3 | 1750–2400 m | 反光锥桶渐变封闭右车道并收窄 |
| 4 | 2300–4300 m | 施工车辆占用右车道，左/中车道通行 |
| 5 | 3100–3450 m | 施工人员临时横穿，减速/停车让行 |
| 6 | 4200–4750 m | 维修车阻塞，等待安全间隙后向左避让 |
| 7 | 4950–5200 m | 施工区结束、车道恢复 |

## 参数如何真正作用

- `weather` 被转换为 `carla.WeatherParameters` 并传给 `world.set_weather`；
- `Town05_Opt` 的 `MapLayer.All` 被加载，Street/Building/Other 灯组在雨夜模式下开启；
- 湿滑摩擦系数通过沿 6 km 路线铺设 `static.trigger.friction` 实际作用于轮胎接触区域；缺少必要蓝图属性或生成失败会立即报错；
- 湿路限速会限制自车 Traffic Manager 目标速度；
- 四路相机的手动曝光、快门、ISO、gamma、运动模糊和眩光参数写入 CARLA 相机蓝图；
- `vehicle_state.jsonl` 每帧记录速度、加速度、角速度、实际车道和控制量，并以 `simulation_frame` 为同步键；
- `voice_command_schedule.jsonl` 输出指令、语义目标、触发进度及噪声注入合同。音频波形的混噪仍由语音/ASR 模块完成，场景侧不把配置项冒充已合成音频。

## 离线检查

```powershell
python experiment/CARLA/run_emergency_response_6km.py --validate-config-only
python -m unittest experiment.CARLA.tests.test_scene3_town05_rebuild
```

## 完整场景预览

启动 CARLA 0.9.16 服务端后，在仓库根目录执行：

```powershell
python experiment/CARLA/scene3_video_preview.py --host 127.0.0.1 --port 2000
```

该入口使用 `--duration 0`，会一直运行到累计进度 6000 m，并启用严格完成检查。默认输出目录为
`experiment/CARLA/outputs/scene3_town05_preview/`，其中包括：

- `scene3_town05_complete_preview.mp4`：带事件 HUD 的第三人称 H.264 完整预览；
- `rgb/front_rgb`、`left_rgb`、`right_rgb`、`rear_rgb`：四路低信噪比图像；
- `vehicle_state.jsonl`：逐帧动态车辆状态；
- `voice_command_schedule.jsonl`：带噪声合同的语音指令计划；
- `event_timeline.jsonl`、`frame_ground_truth.jsonl`、`scene_summary.json`：事件与评测证据。

仅检查预览命令和配置、不连接 CARLA：

```powershell
python experiment/CARLA/scene3_video_preview.py --validate-only --no-strict-completion
```
