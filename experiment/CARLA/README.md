# CARLA 自动驾驶场景仿真平台

一个基于 CARLA 的模块化自动驾驶仿真场景构建框架。本目录统一按
Linux、CARLA 0.9.16、Python 3.12.13 维护，不再使用旧的 Windows
CARLA 0.9.15 / Python 3.7 环境。

## 当前集成版本

- 场景代码来源：`origin/lx`；当前副本用于 `zsz` 分支上的场景帧解释实验。
- CARLA 服务端与 Python API：统一使用 `0.9.16`，二者版本必须一致。
- 推荐 Python：`3.12.13`；当前 AutoDL 环境已验证 PyTorch 可识别 RTX 5090 的 `sm_120`。
- 默认 CARLA 路径：`/root/autodl-tmp/CARLA_0.9.16`，也可通过 `CARLA_ROOT` 覆盖。
- 统一集成环境：Linux（当前验证系统为 Ubuntu 22.04）。

Linux 环境优先直接安装 CARLA 0.9.16 自带的 wheel；`carla_bootstrap.py` 也会从
`$CARLA_ROOT/PythonAPI/carla/dist` 查找 `.whl` 或 `.egg`：

```bash
export CARLA_ROOT=/root/autodl-tmp/CARLA_0.9.16
python -m pip install "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.16-*.whl
python -c "from importlib.metadata import version; import carla; print(version('carla'))"
```

若只需先安装 Python 客户端，也可以使用官方 PyPI（AutoDL 的默认阿里云源不提供该包）：

```bash
python -m pip install -i https://pypi.org/simple carla==0.9.16
```

### AutoDL 服务端验证状态

当前容器已完成以下安装：

```text
CARLA 服务端：/root/autodl-tmp/CARLA_0.9.16（约 19 GB）
CARLA Python API：0.9.16 / CPython 3.12
```

该容器最初因用户态 EGL/X Server 运行库不完整，`vulkaninfo` 返回
`ERROR_INCOMPATIBLE_DRIVER`。以下依赖组合已在当前 Ubuntu 22.04 / RTX 5090 容器验证，
安装后 `vulkaninfo --summary` 可识别 NVIDIA 580.105.08 和 RTX 5090：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  acl libvulkan1 vulkan-tools mesa-utils libegl1 libgles2 libgbm1 \
  xserver-xorg-core xserver-xorg-video-dummy
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
```

CARLA 服务端禁止以 root 身份运行，但 root 可以运行 Python 客户端。AutoDL 默认登录
用户为 root，因此使用专用 `carla` 用户，并只授予其穿过 `/root` 到数据盘的权限：

```bash
export CARLA_ROOT=/root/autodl-tmp/CARLA_0.9.16
export CARLA_CACHE_DIR=/root/autodl-tmp/carla_cache
id carla >/dev/null 2>&1 || useradd -m -s /bin/bash carla
setfacl -m u:carla:--x /root
install -d -o carla -g carla "$CARLA_CACHE_DIR/runtime" "$CARLA_CACHE_DIR/logs"
chmod 700 "$CARLA_CACHE_DIR/runtime"

runuser -u carla -- env \
  HOME="$CARLA_CACHE_DIR" XDG_RUNTIME_DIR="$CARLA_CACHE_DIR/runtime" \
  bash -lc 'cd /root/autodl-tmp/CARLA_0.9.16 && \
    ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000'
```

`-RenderOffScreen` 不显示窗口，但 RGB 摄像头仍正常渲染并把图像直接传给 Python 代码，
适合远程服务器闭环测试。无需安装桌面环境或通过远程桌面查看画面。

## 场景理解数据采集

`run_control_experiment.py` 已接入 `scene_understanding` 的同帧采集桥。它不会让视觉
模型参与紧急制动或 TTC 控制，只保存后续场景帧解释、语义对齐和离线评测所需的数据。

先启动 CARLA 0.9.16 服务端，再从仓库根目录执行：

```bash
export PYTHONPATH="$PWD"
cd experiment/CARLA
python run_control_experiment.py emergency_brake \
  --duration-s 25 \
  --scene-capture \
  --scene-capture-every-n 10 \
  --output-dir outputs/runs/emergency_scene_capture
```

同样可将场景名替换为 `straight_driving` 或 `pedestrian_crossing`。采集结果位于：

```text
outputs/runs/<run>/scene_understanding/
├── capture_index.jsonl
├── sensors/front_rgb/*.png
├── world_states/*.json
└── projections/*.json
```

回到仓库根目录生成 Qwen manifest 并先跑 10 帧冒烟测试：

```bash
cd ../..
python -m scene_understanding.core.prepare_carla_samples \
  --capture-index experiment/CARLA/outputs/runs/emergency_scene_capture/scene_understanding/capture_index.jsonl \
  --prompt scene_understanding/prompts/scene_understanding.txt \
  --output experiment/CARLA/outputs/runs/emergency_scene_capture/scene_manifest.jsonl

python -m scene_understanding.core.run_qwen_scene_inference \
  --manifest experiment/CARLA/outputs/runs/emergency_scene_capture/scene_manifest.jsonl \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --output experiment/CARLA/outputs/runs/emergency_scene_capture/scene_results.jsonl \
  --limit 10 \
  --fail-fast
```

模型输出通过 `scene_understanding.core.visual_semantic_fusion` 与同帧 Actor 投影框融合，
再进入现有语义对齐、风险评估和控制决策接口。图片、权重和 `outputs/` 均为运行产物，
不提交 Git。

## 场景理解模块联调

本目录只说明 CARLA 场景、控制与采集方法。实时检测、异步视觉模型、语义对齐、
历史模型对比和多轮仿真性能结论统一维护在
`../../scene_understanding/README.md`，避免把场景构建与感知模型指标混在一起。

## 功能特点

- Ego车辆仿真
- 基于场景的自动驾驶环境构建
- 动态行人与交通参与者交互
- Ground Truth 数据生成
- 摄像头传感器支持

## 当前支持场景

- 直线行驶场景
- 紧急制动场景
- 行人横穿场景

### 与 XH-202602 正式工况的覆盖关系

当前三个场景分别提供了基础操控、复杂避障和应急响应的代码入口与最小闭环，
但仅属于开发验证场景，不能视为已经完整覆盖比赛方案中的正式工况。

| 题目正式工况 | 当前对应场景 | 已覆盖 | 尚未覆盖 |
| --- | --- | --- | --- |
| 基础语音操控 | `straight_driving` | 直行、车道保持、速度控制、到达终点、碰撞/压线/超速记录 | 5 km 连续路线、双向 6 车道、启动/停止/加减速/转弯/变道完整指令序列 |
| 复杂避障 | `pedestrian_crossing` | 行人横穿、减速避让、碰撞检测 | 阴天傍晚、8 km、十字路口和公交站、混合交通流、多视角相机与激光雷达、避让后变道超车 |
| 极限应急 | `emergency_brake` | 前车突然制动、紧急停车、安全车距 | 雨天夜间、6 km 快速路、施工路段和车道收窄、突发加塞/锥桶/临时横穿行人 |

因此，本目录当前可以支撑模块联调和阶段回归测试；正式验收前仍需由场景负责人
按题目规定补齐路线里程、天气、交通参与者、组合动作和传感器配置。

## 系统架构

### Scenario（场景模块）

负责：

- 仿真环境初始化
- NPC行为控制
- 交通事件生成


### Vehicle（车辆模块）

负责：

- Ego车辆管理
- NPC车辆管理


## 后续规划

- 决策模型接入
- 车辆控制接口
- 自动驾驶规划模块
# 7.20更新

## CARLA 场景框架功能完善

### 更新概述

完成 CARLA 场景框架第一阶段重构。

针对三个基础自动驾驶测试场景：

- `StraightDrivingScenario`
- `EmergencyBrakeScenario`
- `PedestrianCrossingScenario`

新增统一的任务目标定义、成功/失败检测、运行状态管理以及日志输出功能。

---

# 1. 场景状态统一管理

所有场景现在支持统一状态：

- `RUNNING`：场景运行中
- `SUCCESS`：任务成功完成
- `FAILURE`：任务失败


场景运行过程中记录：

- 当前状态
- 结束原因
- 运行时间
- 关键指标


示例：

```json
{
    "status": "SUCCESS",
    "reason": "ego_reached_goal"
}
````

失败示例：

```json
{
    "status": "FAILURE",
    "reason": "collision_with_vehicle"
}
```

---

# 2. 场景任务目标定义

为三个场景增加明确任务目标。

---

## 2.1 StraightDrivingScenario

### 任务目标

自车沿预设直线路线行驶，到达指定目标位置。

### 成功条件

* Ego 到达预设终点。

### 失败条件

* 发生碰撞。
* 超过最大运行时间。

---

## 2.2 EmergencyBrakeScenario

### 任务目标

模拟前车紧急制动场景。

场景包含：

* Ego车辆
* 前方目标车辆
* 同车道行驶关系

### 成功条件

* Ego 安全完成场景。
* 无碰撞。

### 失败条件

* 与前车发生碰撞。
* 发生道路违规。

---

## 2.3 PedestrianCrossingScenario

### 任务目标

模拟行人横穿道路场景。

场景包含：

* Ego车辆
* 横穿行人
* 行人运动轨迹

### 成功条件

* 行人完成横穿。
* Ego 未发生碰撞。

### 失败条件

* 与行人发生碰撞。
* 超时。

---

# 3. 运行状态实时获取

增加场景状态查询接口。

运行过程中可以实时获取：

* 场景信息
* actor ID
* 当前状态
* 任务结果
* 运行指标

调用：

```python
scenario.get_status()
```

示例输出：

```json
{
    "status": "RUNNING",
    "reason": "",
    "actors": {
        "ego": 85,
        "front_vehicle": 86
    }
}
```

---

# 4. Actor 管理与编号记录

增加关键 Actor 注册机制。

现在可以直接获取：

* ego_vehicle id
* front_vehicle id
* walker id
* collision sensor id

例如：

```json
{
    "ego":85,
    "front_vehicle":86
}
```

方便后续：

* 碰撞对象分析
* 真值数据记录
* 场景评测

---

# 5. 碰撞检测功能

为场景增加 collision sensor。

新增记录：

* 碰撞次数
* 碰撞对象
* 失败原因

示例：

```
[StraightDriving] Collision:
vehicle.tesla.model3
```

最终状态：

```json
{
    "status":"FAILURE",
    "reason":"collision_with_vehicle"
}
```

---

# 6. Ego 外部控制支持

所有场景支持：

```python
external_control=True
```

当开启外部控制时：

* 场景不控制 Ego
* 不调用 Ego 自动驾驶逻辑
* 仅控制 NPC、行人和环境

为后续接入：

* 自动驾驶算法
* VAD
* 规划控制模型

提供接口。

---

# 7. 场景结束日志输出

场景结束后自动输出运行结果。

示例：

```
[Scenario] Finished

status: SUCCESS

reason:
ego_reached_goal


metrics:

{
    "collision_count":0,
    "simulation_time":12.5
}
```

失败示例：

```
[Scenario] Finished

status: FAILURE

reason:
collision_with_walker
```
