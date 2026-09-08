# 场景三 Town05_Opt 重构改动日志

日期：2026-08-04

## 重构结论

场景三正式运行链路已由自建 `VLA_EmergencyRoad_6km.xodr` 切换为 CARLA 官方 `Town05_Opt`。原有 7 类应急事件和安全审计能力被保留，通过“路线累计进度 + 逻辑车道适配”迁移到官方弯曲、多道路编号的拓扑中。

## 新增文件

### `scene3_town05_route.py`

- 复用场景二已验证的 Town05 路由构建与进度跟踪代码；
- 从官方 spawn point 239/289 构建不少于 6000 m 的连续路线；
- 用单调累计路线进度替代自建 XODR 的固定 `waypoint.s`；
- 将原有逻辑车道 `-1/-2/-3/-4/-5` 动态映射到 Town05 真实 waypoint；
- 启动前检查各事件及整个施工区的车道可用性。

### `scene3_video_preview.py`

- 提供一条命令完整预览 6 km 场景；
- 同时录制前/左/右/后四视角和第三人称 H.264 视频；
- 默认启用雨夜模式、逐帧真值、HUD 和严格完成检查；
- 支持无 CARLA 环境下的 `--validate-only`。

### `tests/test_scene3_town05_rebuild.py`

- 校验官方地图和 6000 m 完成点；
- 校验雨、夜、湿滑、雾、低信噪比多视角、三类动态干扰和施工收窄合同；
- 校验逻辑车道适配、相机参数传递、中文语音及 120 ms/97% 指标合同；
- 校验预览入口可在未安装 CARLA 时完成配置检查。

## 修改文件

### `configs/scene_3_emergency_6km_runtime.json`

- 地图改为 `/Game/Carla/Maps/Town05_Opt`，删除正式链路中的 XODR 路径；
- 增加 6000 m 路线构建和进度跟踪配置；
- 增加湿路摩擦触发器、40 km/h 湿路限速和低信号相机参数；
- 增加 8 条中文模糊/应急指令及 18 dB 轻微环境噪声合同；
- 每个事件关联对应 `voice_command_id`；
- 应急响应门槛收紧为 `<=120 ms`，语义对齐门槛保持 `>=0.97`。

### `run_emergency_response_6km.py`

- 使用 `client.load_world("Town05_Opt")` 和 `MapLayer.All`，校验实际加载地图；
- 启用官方建筑、绿化、街景及 Town05 夜间灯光；
- 创建真实 `static.trigger.friction` 湿滑区域，并在 API 属性缺失或生成失败时中止；
- 将雨夜配置写入 `carla.WeatherParameters`，将低光参数写入相机蓝图；
- 使用路线累计进度触发事件和判断严格 6000 m 完成；
- 动态解析合法 Town05 车道，不再把固定 XODR lane ID 当成仿真真值；
- 新增逐帧 `vehicle_state.jsonl` 和 `voice_command_schedule.jsonl`；
- 完整预览时可同时保留四路低信号图像与第三人称视频。

### `emergency_scene_3_events.py`

- 保留既有加塞、施工锥桶、施工车辆、行人和阻塞车辆代码；
- Actor 的 `get_waypoint_xodr` 调用由适配器翻译到官方地图；
- 行人横穿由“只修改全局 Y 坐标”改为沿 Town05 实际起终点二维插值，适配弯路；
- 阻塞车道间隙改为真实空间距离，保留旧测试替身兼容逻辑；
- 加塞完成判定改为比较当前官方 waypoint 与逻辑目标车道。

### `SCENE_3_EMERGENCY_RESPONSE.md`

- 删除旧自建直路的运行说明和已经过时的完成声明；
- 补充 Town05 路线、事件、参数实际生效位置、输出物和完整预览方法。

## 参数生效边界

已经直接作用于 CARLA 的参数：天气、雨量、积水、湿度、雾、太阳角度、官方地图灯光、物理摩擦、Traffic Manager 目标速度、相机曝光/快门/ISO/gamma/运动模糊/眩光。

作为跨模块接口输出的参数：语音噪声 SNR 与峰值限制。场景运行器会生成带触发进度和噪声合同的计划，实际音频混噪应由组员负责的音频预处理/ASR 链完成。

作为评测门槛而非预设结果的参数：120 ms 响应时延和 97% 语义对齐准确率。必须由 VLA 推理与评测模块依据同步日志计算，当前代码不会直接写死“达标”。

## 本机验证记录

当前设备未安装 CARLA，未执行服务端、图形渲染和完整 6 km 实车流仿真。已完成：

- Python 语法编译检查；
- JSON 配置加载及正式运行器 `--validate-config-only`；
- 新增 Town05 场景三合同测试；
- 原场景三调度、Actor mock、安全审计和预览辅助测试回归。

需要在有 CARLA 0.9.16、Town05_Opt 和 FFmpeg 的设备上执行一次完整预览，最终确认 spawn point 239/289 在目标 CARLA 包版本中一致、所有摩擦触发器均能生成、雨夜渲染观感和 6 km 严格完成结果。
