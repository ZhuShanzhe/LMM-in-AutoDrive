---
language:
- zh
- en
library_name: pytorch
tags:
- autonomous-driving
- vision-language-action
- carla
- multimodal
- pytorch
---

# Universal Three-Scene Sensor-Policy VLA V6

这是 `LMM-in-AutoDrive` 基础赛道最终部署权重。三个题目场景使用同一模型架构、同一 checkpoint、同一高层决策接口，不按场景切换模型。

- Hugging Face：`UNIC0RN-Zhu/universal-three-scene-sensor-policy-vla-v6`
- GitHub：`ZhuShanzhe/LMM-in-AutoDrive`
- 对应 Git 提交：`cb712ab`
- 模型文件：`model.pt`
- 模型大小：21,165,103 bytes
- 模型 SHA-256：`53e949b37c84d6010ab45bfd473cb9d39a88cd89cd7729f55d3e9bb1baddaad3`
- 部署配置 SHA-256：`40164752c522779330a2a2f68a869968eaacb075eb409bc91813143a3ef9c39e`

## 架构和接口

固定在线链路：

```text
raw text + synchronized sensors + ego/environment state
  -> UnifiedSensorBatch
  -> Lightweight VLA multimodal fusion and decision heads
  -> Generic Temporal Risk Supervisor
  -> Generic Instruction FSM
  -> Generic Route PID
  -> carla.VehicleControl
```

`UnifiedSensorBatch` 固定包含文本、前/左/右/后 RGB、LiDAR BEV、自车状态、环境状态及模态 mask。不可用模态使用零张量和 `mask=false`，所以场景间不会改变输入 shape 或模型类。

输出包括九类驾驶动作、目标速度、目标车道、置信度与视觉风险。在线决策代码不读取 `scene_id`、`event_id`、`command_id` 或 CARLA actor 真值。声音不进入模型；测试链路仅使用文本指令。

## 通用安全边界

学习型 VLA 与可审计的物理/规则安全层共同控制车辆：

- 前向危险：根据规划走廊内雷达、视觉风险和合法目标车道选择制动、让行或安全变道；
- 后向逼近：若后车 TTC/闭合速度构成碰撞风险，在道路与路线限速内加速脱险；达到速度上限时只考虑合法且视觉安全的变道；
- 文本停车、前向紧急危险和道路限速优先级高于后向脱险；
- 传感器启动期在第一组同步 RGB 到达前执行安全驻车，并单独记录为 `sensor_warmup_safe_hold_count`，不与运行期 fallback 混淆。

这些规则对三个场景一致，不依赖事件里程或演员角色。

## 训练和独立测试

随模型上传的 `training_report.json` 是该 checkpoint 的原始训练报告。记录的划分为训练 26,309、验证 5,328、测试 4,578，随机种子 `20260805`。本权重未使用服务器上的 nuScenes 或 Waymo 资产训练；不能把它们写成已用训练数据。

独立测试集结果：

| 指标 | 结果 |
|---|---:|
| 动作准确率 | 91.37% |
| 宏平均动作准确率 | 93.27% |
| 紧急制动准确率 | 77.37% |
| 目标速度 MAE | 2.34 km/h |
| 视觉风险总体 / 宏平均准确率 | 75.91% / 61.47% |
| low / medium / high 风险准确率 | 85.05% / 61.58% / 37.80% |
| 视觉反事实准确率 | 85.52% |
| 环境速度顺序准确率 | 97.34% |

## CARLA 三场景结果

| 场景 | 结果 |
|---|---|
| 场景一 Town04 | 路线运行器 `SUCCESS`；路线进度 4977.011/4995 m；15/15 指令；0 碰撞；0 非法车道；0 fallback |
| 场景二 Town05 | 8000.913 m 完成；15/15 指令；4/4 事件；70 辆车、21 行人；0 碰撞；0 车道侵入；0 fallback |
| 场景三 Town05 雨夜 | 6000.404 m 完成；7/7 事件；0 碰撞；0 受限标线侵入；物理后向加速脱险触发 42 次 |

场景三完整 r32 运行发生过 1 次“首组同步 RGB 尚未到达”的启动期安全驻车，发生在车辆行驶前。提交代码将其修复为独立预热计数；随后 20 秒 smoke test 产生 133 次真实决策，`fallback_count=0`、`sensor_warmup_safe_hold_count=1`、0 碰撞、0 非法车道，传感器到决策延迟 120 ms 内比例 100%。由于项目时间限制，没有在该口径修复后重复完整 6 km；Model Card 不把这两份证据伪装为同一次运行。

## 使用

将文件放在仓库相对模型目录：

```text
models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/
  model.pt
  config.json
```

推荐使用仓库脚本和 Linux 相对路径。Docker 镜像中可用 `MODEL_ROOT` 指向只读权重挂载目录。模型依赖 CARLA 0.9.16、ModernBERT 指令解析器及仓库内 VLA 代码，不能把 `model.pt` 当成独立 Transformers 模型直接加载。

## 限制

- high 风险视觉头独立准确率仅 37.80%，不能脱离时序确认、物理雷达和规则安全层单独部署；
- 三个固定 CARLA 场景通过不等于任意开放道路的安全保证；
- 场景二使用 100 m hybrid-physics 半径：近场为完整物理，远场由 Traffic Manager 简化；
- 权重只用于研究和比赛验证，不用于真实车辆或安全关键生产部署。
