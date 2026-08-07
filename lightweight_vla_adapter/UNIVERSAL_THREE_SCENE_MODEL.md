# 三场景通用传感器 VLA 模型卡
## 统一在线链路（2026-08-07 更新）

提交版使用唯一的在线控制器入口 `experiment/CARLA/universal_vla_controller.py`（`UniversalVLAController`），固定链路为：

```text
UnifiedSensorBatch
  -> Universal VLA Pipeline
  -> Generic Temporal Risk Supervisor
  -> Generic Instruction FSM
  -> Route PID
  -> carla.VehicleControl
```

- 统一输入超集：`lightweight_vla_adapter/src/unified_sensor_batch.py`（`UnifiedSensorBatch`，`schema_version=unified_sensor_batch/1.0`），固定字段 `text_tokens / front_rgb / left_rgb / right_rgb / rear_rgb / lidar_bev / vehicle_state / environment_state / camera_view_mask / modality_mask / schema_version`；缺失模态以零张量 + mask 表示。
- 通用指令 FSM：`experiment/CARLA/control/generic_instruction_fsm.py`，输出 `parsed_intent`（KEEP_LANE / SET_SPEED / DECELERATE / EMERGENCY_BRAKE / YIELD / CHANGE_LANE_LEFT / CHANGE_LANE_RIGHT / STOP / RESUME 及转弯意图），不依赖 command/event/scene id。
- 通用时序风险监督器：`experiment/CARLA/control/generic_temporal_risk_supervisor.py`，仅使用全局/目标车道视觉风险历史、停车时长、车速、车辆状态与 `modality_mask`；覆盖名为 `low_risk_deceleration_crawl`、`temporal_hazard_clearance`、`cautious_hazard_resume`、`target_lane_visual_clearance`。
- 通用 Route PID：`experiment/CARLA/control/generic_route_pid.py`（底层复用 `EgoPIDController`），不读取 actor 真值、不含场景/事件/命令专用分支。低/中风险谨慎爬行下限为 10 km/h，路口/道路切换前统一限速 9 km/h；爬行油门增强保证恢复窗口内车辆实际前进。

三个场景运行脚本均调用 `UniversalVLAController`；场景代码只保留地图、天气、传感器安装、文本指令时间表、事件生成、独立评估与记录。旧 `Scene3VlaController` 已删除。

## 交付版本

## 交付版本

- 模型：`universal-three-scene-sensor-policy-vla-v6`
- 配置：`lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json`
- 权重目标路径：`models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt`
- 权重大小：21,165,103 bytes
- 权重 SHA-256：`06c774e3a5eead95e230b55b65a3d86c52ba93110c6046506be21dd69ecb2165`
- 配置 SHA-256：`47204682f7cdb5ee7be3ec0a077f1ee04c0a067b0a6a07a2aa9434949a4aa20d`

权重不写入 Git，由 Docker 镜像或只读 `models/` 挂载目录交付。三个场景使用同一份权重和同一套高层决策代码，不按场景切换 checkpoint。`submission_env.sh` 和 `experiment/CARLA/scripts/run_universal_vla.sh` 只使用仓库相对路径，并允许用 `MODEL_ROOT` 覆盖模型挂载根目录。

## 输入、融合和控制边界

一次 VLA 决策联合编码当前原子文本指令、同步 RGB、车辆状态和环境状态；场景二额外使用物理 LiDAR 点云实时栅格化的 4 通道 BEV。

| 场景 | 模型可见输入 |
|---|---|
| 场景一 | 文本、前视 RGB、车辆状态、环境状态 |
| 场景二 | 文本、前/左/右/后 RGB、真实 LiDAR BEV、车辆状态、环境状态 |
| 场景三 | 文本、低信噪比前/左/右/后 RGB、车辆状态、环境状态 |

模型使用 4 层、256 hidden、8 heads 的跨模态融合网络，输出九类驾驶动作、目标速度、目标车道、置信度和三类视觉风险。复合指令先拆成有序原子步骤，VLA 每次只处理当前步骤，执行反馈驱动 FSM 进入下一步，最后由路线/PID 将高层决策转换为车辆控制。

V6 的策略输入边界禁止 CARLA actor 真值：

- `use_candidate_entities=false`，候选实体张量始终清零；
- CARLA 生成的 camera-BEV 不进入模型；
- 场景一/三的 LiDAR 兼容张量为零；
- 场景二仅保留真实 LiDAR 传感器生成的 BEV；
- 碰撞风险和变道风险门使用原始 RGB 分支的学习式风险头；
- CARLA actor API 只用于事件生成、训练标签和独立评测记录，不进入在线 VLA；运行日志逐决策记录 `policy_truth_access=false`。

规则/FSM作为独立、可审计的失效保护边界，处理指令合法性、物理速度上限和低层执行，不替代模型的一般驾驶决策。该分层符合“VLA连接传感器输入与车辆控制接口”的要求，也保留了弱视觉模型在安全关键任务中必要的防护。

## 训练数据与防泄漏处理

最终清单共 34,132 个样本，按轨迹/反事实组隔离划分为训练 24,435、验证 5,119、测试 4,578：

| 来源 | 样本数 | 用途 |
|---|---:|---|
| CARLA 场景三、天气/光照反事实和事件动作 | 25,362 | 雨夜、施工、低可见度、应急动作和反事实约束 |
| nuScenes 多相机/点云样本 | 8,080 | 真实道路视觉与 LiDAR 分布 |
| CARLA Town04 场景一硬负样本 | 690 | 正常巡航、加减速，抑制无风险急刹 |

训练和验证时执行与部署相同的输入掩码：所有 CARLA 样本的 actor-BEV/候选实体均清零；只有标记为 `nuScenes` 的样本允许保留传感器 BEV。这样测试指标不会通过训练脚本重新引入仿真真值。

服务器另保留 1.66 GiB 的 Waymo v2 perception 精选集（3 个训练 segment、1 个验证 segment，共 16 个 Parquet）。该精选集只有相机图像、相机框、标定和统计，没有本轮需要的 LiDAR/vehicle pose，因此未参与 V6 训练，不能在提交材料中列为已用训练来源。

## 独立测试集结果

| 指标 | V6 结果 |
|---|---:|
| 动作准确率 | 93.60% |
| 宏平均动作准确率 | 95.25% |
| 紧急制动准确率 | 80.85% |
| 停车准确率 | 100.00% |
| 左/右转准确率 | 99.26% / 98.06% |
| 左/右变道准确率 | 97.95% / 97.90% |
| 目标速度 MAE | 1.93 km/h |
| 视觉反事实准确率 | 87.67% |
| 视觉风险总体/宏平均准确率 | 82.02% / 69.63% |
| 中风险准确率 | 78.03% |
| 高风险准确率 | 41.73% |
| 环境速度上限违规率 | 0.00% |
| 环境反事实顺序准确率 | 100.00% |

相较 V5，V6 在完全移除 actor 候选和真值风险后，动作准确率仅下降 0.35 个百分点，紧急制动提高 2.71 个百分点，高风险视觉识别由 32.28% 提高到 41.73%。

## 已知不足

高风险视觉识别 41.73% 仍不足以单独承担端到端安全保证，尤其是雨夜遮挡行人、远距离近失和反光路锥。当前版本必须保留独立规则失效保护和碰撞传感器评测，不能宣称已经达到量产自动驾驶安全水平。下一轮数据工作应优先增加真实/仿真的雨夜遮挡 VRU、近失时序片段和困难负样本，而不是继续堆叠普通巡航样本。

场景三配置提供 `baseline`、`cautious_sparse`、`dense_dynamic` 三套确定性变体；每套覆盖全部七类事件，并可由 `--event-variant auto --seed ...` 自动复现。完整路线测试仍应至少覆盖一个非基准变体。
