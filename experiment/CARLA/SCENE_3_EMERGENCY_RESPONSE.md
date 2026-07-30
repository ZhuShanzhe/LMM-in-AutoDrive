# 场景三：6 km 应急响应场景

本场景用于验证车辆在雨夜、湿滑、低能见度道路中对连续突发事件的响应能力。
路线为 6 km 双向直线快速路，每个方向包含三条机动车道，不包含路口、掉头点或
不可恢复的必撞事件。

## 场景目标

- 在统一路线中连续触发 7 个可恢复应急事件；
- 提供前、左、右、后四路 RGB 图像；
- 记录事件激活、解除、路线完成、碰撞和车道使用情况；
- 为上层 VLA 提议、风险判断与控制模块提供可重复的 CARLA 场景；
- 所有危险事件都必须保留足够的预警距离或安全避让空间。

## 正式文件

| 文件 | 用途 |
|---|---|
| `maps/generate_emergency_road_xodr.py` | 生成并静态校验 6 km OpenDRIVE 地图 |
| `maps/maps/output/VLA_EmergencyRoad_6km.xodr` | 生成的正式地图 |
| `configs/scene_3_emergency_6km_runtime.json` | 天气、交通参与者、传感器和事件配置 |
| `emergency_scene_3_events.py` | 创建并更新事件 Actor |
| `run_emergency_response_6km.py` | 场景三正式运行入口 |
| `tests/test_emergency_response_6km.py` | 不依赖 CARLA 服务端的合同与调度测试 |

## 道路与环境

- 路线长度：6000 m；
- 交通规则：右侧通行；
- 自车方向车道：左侧 `-1`、中间 `-2`、右侧 `-3`；
- 路肩：`-4`，人行道：`-5`；
- 自车起点：road `1`、lane `-2`、s `50 m`；
- 路线完成点：s `5990 m`；
- 天气：100% 阴云、80% 降雨、80% 积水、100% 湿滑；
- 夜间太阳高度角：`-15°`；
- 雾密度：35，雾可视距离：75 m。

交通参与者总量为 16 辆社会车辆、2 辆施工车辆、1 辆维修车辆和 2 名施工人员。
图像传感器固定为 `front_rgb`、`left_rgb`、`right_rgb` 和 `rear_rgb`。
正式提交视频使用同步第三人称 `chase_rgb` 相机，挂载于自车后上方；四路 RGB 模式仍保留用于传感器合同和离线证据。

## 事件序列

| 顺序 | 位置 | 事件 | 预期安全响应 |
|---|---:|---|---|
| 1 | 1080 m | 左侧车辆切入自车车道 | 减速、保持可控跟车距离 |
| 2 | 1550 m | 施工提前警示 | 减速并持续观察 |
| 3 | 1850–2400 m | 锥桶渐变封闭右车道 | 减速并向左安全并线 |
| 4 | 2400–4300 m | 右侧车道施工区 | 保持开放车道并低速通过 |
| 5 | 3200 m | 临时施工人员横穿 | 减速、停车或让行 |
| 6 | 4300 m | 维修车阻塞中间车道 | 等待左侧安全间隙后变道 |
| 7 | 5050 m | 施工区结束、右车道恢复 | 保持车道并逐步恢复车速 |

第 6 个事件会先创建不安全的左侧目标车道间隙。只有实测前向间距不小于
30 m、后向间距不小于 25 m 后，目标车道才会被标记为可用。

## 离线验证

在仓库根目录运行：

```bash
python experiment/CARLA/maps/generate_emergency_road_xodr.py

python experiment/CARLA/run_emergency_response_6km.py \
  --validate-config-only

PYTHONPATH="$PWD/experiment/CARLA" \
python -m unittest discover \
  -s experiment/CARLA/tests \
  -v
```

离线测试覆盖地图结构、地图可重复生成、事件顺序、Actor 数量、四路相机方向、
配置拒绝规则、事件激活和解除、路线完成、碰撞记录及非法车道检测。

## CARLA 实际运行

先在具有 NVIDIA Vulkan 图形能力的计算节点启动 CARLA 0.9.16 服务端，再运行：

```bash
python experiment/CARLA/run_emergency_response_6km.py \
  --host 127.0.0.1 \
  --port 2000 \
  --duration 0 \
  --record-ground-truth \
  --ground-truth-every-n 1 \
  --require-complete-scene
```

`--duration 0` 表示持续运行到车辆到达 5990 m，或由用户按 `Ctrl+C` 停止。
`--require-complete-scene` 会在以下任一条件不满足时返回失败：

- 路线没有完成；
- 7 个事件没有全部解除；
- 发生碰撞；
- 自车驶入 `-1`、`-2`、`-3` 以外的车道；
- 任一路 RGB 相机没有产出图像。

## 输出产物

默认输出目录为 `experiment/CARLA/outputs/emergency_scene_3/`：

- `rgb/front_rgb/`、`rgb/left_rgb/`、`rgb/right_rgb/`、`rgb/rear_rgb/`：
  四路相机图像；
- `runtime_config_snapshot.json`：本次运行配置快照；
- `event_timeline.jsonl`：事件激活和解除时间线；
- `scene_summary.json`：路线、事件、碰撞、车道和图像数量汇总。
- `frame_ground_truth.jsonl`：切入车、施工区、横穿工人、阻塞车与安全间隙的
  精确帧真值；影子评测方法见 `SCENE_GROUND_TRUTH_EVALUATION.md`。

运行产物不应提交到 Git。

## 当前验证状态

场景三的地图、配置、事件调度、安全审计及录像链已经通过自动化测试。
2026 年 7 月 29 日已完成真实 CARLA 6 km 验收运行：路线到达 5990.4 m，7 个事件全部解除，碰撞数为 0，非法车道采样数为 0。

正式第三人称视频为 1920×1080、30 FPS、12598 帧，直接编码为 H.264 MP4，完整解码验证通过。运行产物、日志和视频不提交到 Git。
