# CARLA 自动驾驶场景仿真平台

一个基于 CARLA 的模块化自动驾驶仿真场景构建框架。

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
可以，改成中文版本，适合直接放仓库 `UPDATE_LOG.md`：

````md
# 更新日志

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

