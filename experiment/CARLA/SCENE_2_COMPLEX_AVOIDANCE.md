# 场景二：8 km 复杂避障语音操控

本目录提供场景二的正式 CARLA 实现，并复用现有 `decorate_complex_scene.py` 和
`VLA_ComplexRoad_8km.xodr`。运行器不使用手写横向
坐标随机摆放 actor，而是使用 CARLA OpenDRIVE waypoint、同步时钟、确定性随机
种子和版本化 JSON 契约。

实现依据（比赛阈值仍须以主办方原始任务书复核）：

- 当前场景合同中的目标：8 km、违规不超过 1、无碰撞、ASR 不低于 96%、
  组合步骤无遗漏、语义对齐不低于 98.5%；
- 当前场景配置中的 6 段路线、混合交通与事件顺序；
- `configs/scene_2_complex_avoidance_8km_runtime.json` 的 15 条组合指令、6 个特殊事件和传感器；
- `CARLA_INTERFACE_GUIDE.md`、`control/protocol.py` 以及场景理解模块的
  DrivingIntent、WorldState、MultimodalFrameBundle、VLA 安全门和
  ControlDecision 边界。

## 对齐的比赛验收要求

| 项目 | 实现/证据 |
|---|---|
| 8 km 连续路线 | 主路 `road=1` 长度为 8000 m；配置分为 6 个里程段 |
| 阴天/傍晚复杂交通 | 固定天气；24 辆私家车、3 辆公交车、6 辆自行车、18 名行人 |
| 组合语音 | 15 条在线指令，全部包含至少 3 个有序步骤 |
| 动态避障 | 行人横穿、公交乘客、自行车、慢车与条件安全间隙共 6 个事件 |
| 摄像头 | 场景一/三同款 `chase_rgb` 第三人称追尾视角；可直编码 H.264 |
| 多模态接口 | 四路 RGB、LiDAR、VehicleState；以 `simulation_frame` 严格同步 |
| 决策接口 | DrivingIntent 1.2 → VlaActionProposal → 确定性安全门 → ControlDecision 1.0 |
| 安全验收 | 碰撞 0；违规不超过 1；事件/指令/传感器日志可审计 |
| 本地目标（待任务书复核） | ASR ≥96%；组合步骤遗漏为 0；语义对齐 ≥98.5% |

线上 8 km 单次运行只能证明路线、安全和接口可用。ASR 与指令解析应使用独立
固定语料统计；多模态语义对齐与控制必须使用同一 `simulation_frame` 的 CARLA
逐帧真值和模型输出，详见 `SCENE_GROUND_TRUTH_EVALUATION.md`。

## 文件

- `decorate_complex_scene.py`：正式运行器。
- `scene_2_complex_avoidance_8km_runtime.json`：场景、指令、事件、接口和验收契约。
- `scene2_runtime_interface.py`：DrivingIntent、严格多模态 Bundle 和安全门后
  ControlDecision 的纯 Python 公共接口。
- `VLA_ComplexRoad_8km.xodr`：队友提供的 8 km OpenDRIVE 地图。
- `test_scene2_contract.py`：不需要 CARLA 服务端的合同测试。

## 一次性检查

```bash
python -m py_compile \
  experiment/CARLA/maps/decorate_complex_scene.py \
  experiment/CARLA/scene2_runtime_interface.py

python experiment/CARLA/maps/decorate_complex_scene.py \
  --validate-config-only

python -m unittest \
  experiment/CARLA/tests/test_scene2_contract.py
```

## 30 秒冒烟测试

CARLA 0.9.16 已在 2000/2001 端口启动后执行：

```bash
SCENE2_SMOKE_DIR="$PWD/experiment/CARLA/outputs/scene2_smoke_${SLURM_JOB_ID}"
FFMPEG_BIN="$(
  python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
)"

python experiment/CARLA/maps/decorate_complex_scene.py \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-manager-port 8000 \
  --duration 30 \
  --camera-width 1280 \
  --camera-height 720 \
  --video-fps 20 \
  --video-output "$SCENE2_SMOKE_DIR/scene2_smoke.mp4" \
  --ffmpeg "$FFMPEG_BIN" \
  --video-overlay \
  --output-dir "$SCENE2_SMOKE_DIR"
```

画面必须显示自车后上方的第三人称视角，而不是原脚本的固定 95 m 高空俯视。

## 完整 8 km 运行

```bash
set -o pipefail

SCENE2_FINAL_DIR="$PWD/experiment/CARLA/outputs/scene2_final_${SLURM_JOB_ID}"
FFMPEG_BIN="$(
  python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
)"

python experiment/CARLA/maps/decorate_complex_scene.py \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-manager-port 8000 \
  --duration 3600 \
  --fixed-delta-seconds 0.05 \
  --camera-width 1920 \
  --camera-height 1080 \
  --video-fps 20 \
  --video-output "$SCENE2_FINAL_DIR/scene2_complex_avoidance_8km_master.mp4" \
  --ffmpeg "$FFMPEG_BIN" \
  --video-overlay \
  --record-multimodal \
  --record-ground-truth \
  --ground-truth-every-n 1 \
  --sensor-tick 0.5 \
  --output-dir "$SCENE2_FINAL_DIR" \
  --require-complete-scene \
  2>&1 | tee /tmp/scene2_final.log

SCENE2_STATUS=${PIPESTATUS[0]}
echo "SCENE 2 FINAL STATUS: $SCENE2_STATUS"
```

## 输出接口

- `driving_intent.jsonl`：DrivingIntent 1.2.0 调度输入。
- `world_state.jsonl`：按仿真帧输出的自车、车道、天气与安全状态。
- `multimodal_frame_bundle.jsonl`：四路 RGB、LiDAR 与 WorldState 的同步状态。
- `interface_manifest.json`：上下游 Schema 与安全门边界。
- `command_timeline.jsonl`：语音发布与有序步骤数量。
- `event_timeline.jsonl`：可见事件激活和解除。
- `frame_ground_truth.jsonl`：不读取模型输出的逐帧 CARLA 真值、证据等级和
  允许/禁止控制动作。
- `scene_summary.json`：路线、actor 数量、碰撞、违规、录像与待测赛事指标。

## 当前地图边界

当前正式地图包含一条 8 km 双向主路、双向机动车道/自行车道/人行道、一个真实
OpenDRIVE 十字路口和一个公交站声明。它尚未为 S2-05、S2-06 提供第二个物理信号
路口和独立合法掉头几何，因此运行器会在 `scene_summary.json.map_contract` 中明确
记录这一点，不能把直路上的事件绑定冒充为完整拓扑验收。正式提交前应补齐地图
几何并在 CARLA 中实际验证左转、右转和掉头 waypoint 连通性。
