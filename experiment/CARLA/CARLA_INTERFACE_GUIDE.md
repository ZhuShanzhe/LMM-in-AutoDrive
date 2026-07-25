# CARLA 接口说明与控制链路

## 1. 整体链路

当前系统可以理解为一条闭环控制链：

```text
CARLA 场景/世界状态
  -> 感知与世界状态封装
  -> 决策模块输出驾驶意图
  -> 控制模块转成 carla.VehicleControl
  -> vehicle.apply_control(...)
```

其中，前端输入来自 CARLA 世界，后端输出是车辆控制命令。

---

## 2. 前端输入：CARLA 场景反馈信息

### 2.1 数据来源

当前实现通过以下模块生成世界状态：

- [perception/world_state.py](perception/world_state.py)
- [perception/ground_truth.py](perception/ground_truth.py)

这些模块从 CARLA 中获取 ego 自车状态、周围车辆、行人、车道信息、交通灯、天气以及碰撞状态。

### 2.2 世界状态样例

```json
{
  "ego": {
    "location": {"x": 12.3, "y": 45.6, "z": 0.0},
    "rotation": {"pitch": 0.0, "yaw": 90.1, "roll": 0.0},
    "speed(km/h)": 24.3
  },
  "vehicles": [
    {
      "id": 42,
      "type": "vehicle.tesla.model3",
      "distance": 18.5,
      "speed_kmh": 22.0,
      "relative_position": {"x": 14.2, "y": 0.8},
      "direction": "front"
    }
  ],
  "pedestrians": [
    {
      "id": 100,
      "type": "walker.pedestrian.0001",
      "distance": 12.0
    }
  ],
  "obstacles": [],
  "lane": {
    "road_id": 7,
    "section_id": 0,
    "lane_id": 1,
    "lane_type": "LaneType.Driving",
    "is_junction": false
  },
  "traffic_lights": [],
  "weather": {
    "cloudiness": 0.0,
    "precipitation": 0.0,
    "sun_altitude_angle": 45.0
  },
  "collision": {"status": false}
}
```

### 2.3 主要字段说明

- ego：自车当前位置、朝向、速度
- vehicles：周围车辆列表，包含相对距离、相对位置、速度
- pedestrians：周围行人列表
- obstacles：静态障碍物
- lane：当前车道信息
- traffic_lights：附近红绿灯状态
- weather：天气状态
- collision：当前碰撞状态

这部分信息是后续决策模块的输入上下文。

---

## 3. 中间接口：驾驶意图 / 控制决策

决策模块可以输出两种形式之一：

1. 扁平动作字典
2. 结构化 DrivingIntent JSON

### 3.1 扁平动作字典示例

```json
{
  "action": "decelerate",
  "target_speed_kmh": 20.0,
  "target_lane": null,
  "target_location": null,
  "emergency": false,
  "reason": "front_vehicle_close",
  "request_id": "frame-001"
}
```

### 3.2 结构化 DrivingIntent 示例

```json
{
  "request_id": "frame-001",
  "parse_result": {
    "status": "VALID",
    "confidence": 0.95
  },
  "intent": {
    "urgency": "NORMAL",
    "steps": [
      {
        "step_id": "s1",
        "action": "ADJUST_SPEED",
        "parameters": {
          "change": "DECREASE",
          "target_speed_mps": 5.5
        }
      }
    ]
  }
}
```

### 3.3 支持的动作类型

当前协议层支持以下控制动作：

- keep_lane
- accelerate
- decelerate
- stop
- emergency_brake
- lane_change_left
- lane_change_right
- turn_left
- turn_right

这些动作由 [control/protocol.py](control/protocol.py) 统一规范化。

---

## 4. 后端接口：车辆控制输出

控制模块会把决策意图转成低层车辆控制命令，最终通过 CARLA 的 `VehicleControl` 对象发送给汽车。

### 4.1 规范化后的决策字典

控制层实际接收到的规范化字典形式如下：

```python
{
    "action": "keep_lane",
    "target_speed_kmh": 25.0,
    "target_lane": None,
    "target_location": None,
    "emergency": False,
    "reason": "rule_cruise",
    "request_id": "frame-001",
    "command_id": None,
    "voice_text": "",
    "structured_command": {}
}
```

### 4.2 低层控制输出

输出对象是 `carla.VehicleControl`：

```python
control = carla.VehicleControl(throttle=0.42, brake=0.0, steer=-0.12)
vehicle.apply_control(control)
```

### 4.3 当前控制器实现

当前项目中有两种控制器入口：

- [control/pid_controller.py](control/pid_controller.py)
  - 使用 PID 做纵向和横向控制
- [control/agents.py](control/agents.py)
  - 使用 CARLA 自带 Agent 进行控制

其中默认主流程是：

```python
control, intent = controller.run_step(intent, dt)
vehicle.apply_control(control)
```

---

## 5. 关键代码入口

### 5.1 决策入口

- [run_control_experiment.py](run_control_experiment.py)
- [control/decision_provider.py](control/decision_provider.py)

### 5.2 协议规范化入口

- [control/protocol.py](control/protocol.py)

### 5.3 控制执行入口

- [control/pid_controller.py](control/pid_controller.py)
- [control/agents.py](control/agents.py)

---

## 6. 接入方式建议

如果后续要接入视觉模型、语言模型或 LLM 决策器，建议维持现有接口边界：

```text
CARLA 世界状态 -> 你的决策/规划模块 -> 统一动作字典/DrivingIntent
-> 控制模块 -> carla.VehicleControl -> CARLA
```

也就是说，前端只需要负责“读取世界状态并输出统一意图”，后端只负责“把意图翻译为车辆控制”。

这使得决策模块可以替换，而不需要改动控制和评测链路。