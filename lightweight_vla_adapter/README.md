# 轻量多模态 VLA 决策适配模块

## 模块定位

本模块在现有语音、指令解析、场景理解、风险评估与 CARLA 控制链路之间增加一个可训练的场景条件化决策层。部署链路不直接运行大体量 VLA，而是使用轻量学生模型；UniDriveVLA-Base、OpenDriveVLA-0.5B 和 SimLingo 只用于离线教师蒸馏及对照实验。

当前链路：

```text
ASR / 翻译
  -> ModernBERT DrivingIntent 1.2
  -> 多视角摄像头与 LiDAR BEV 特征
  -> 候选场景实体与车辆状态
  -> 4 层 Cross-Attention Decision Adapter
  -> VLADecisionProposal
  -> 现有确定性风险安全门
  -> 现有 ControlPlan FSM
  -> ControlDecision 1.0
  -> CARLA control.protocol
```

安全门和 FSM 始终处于学生模型下游。学生模型不能：

- 绕过 `emergency_brake` 或风险减速建议；
- 把左变道指令改成右变道；
- 在目标车道不安全时执行变道；
- 在解析失败、对齐失败或计划阻塞时覆盖安全兜底；
- 向 CARLA 输出协议之外的动作。

## 目录结构

```text
lightweight_vla_adapter/
├── configs/
│   ├── student_base.json
│   └── teachers.json
├── schemas/
│   ├── sensor_bundle.schema.json
│   └── vla_decision_proposal.schema.json
├── scripts/
│   ├── evaluate_predictions.py
│   ├── run_offline_inference.py
│   ├── train_student.py
│   └── verify_teacher_assets.py
├── src/
│   ├── bev_encoder.py
│   ├── contracts.py
│   ├── decision_adapter.py
│   ├── distillation.py
│   ├── intent_encoder.py
│   ├── pipeline.py
│   ├── safety_bridge.py
│   ├── structured_bev.py
│   └── teacher.py
└── tests/
```

## 输入接口

### 1. DrivingIntent

直接接收 `structured_command_parser` 输出的 DrivingIntent 1.2。ModernBERT 的同一套 backbone 通过 `ModernBertIntentEncoder` 输出 token 特征，默认冻结 backbone，只训练投影层和决策适配器。

### 2. SensorBundle

`SensorBundle 1.0` 用于对齐多视角图片、激光雷达、车辆状态与候选实体：

```json
{
  "schema_version": "1.0.0",
  "frame_id": "carla_100",
  "timestamp_s": 5.0,
  "cameras": [
    {
      "name": "front",
      "timestamp_s": 5.0,
      "image_path": "front/000100.png"
    }
  ],
  "lidar": {
    "timestamp_s": 5.0,
    "points_path": "lidar/000100.bin"
  },
  "ego_state": {
    "speed_mps": 5.0,
    "acceleration_mps2": 0.0,
    "yaw_rate_rps": 0.0
  },
  "candidate_entities": [
    {"entity_id": "vehicle_front"}
  ],
  "feature_refs": {
    "camera_bev": "features/camera/000100.pt",
    "lidar_bev": "features/lidar/000100.pt"
  }
}
```

传感器时间戳默认允许的最大偏差为 `100 ms`。`target_entity_id` 只能从 `candidate_entities` 中选择，不允许模型自由生成对象。

### 3. BEV 特征

支持两种接入方式：

1. **真实感知模式**：BEVFusion、LSS、PointPillars 等上游输出 `camera_bev` 和 `lidar_bev` 张量。
2. **CARLA 联调模式**：`StructuredBEVRasterizer` 将现有 `WorldState` 和 CARLA 真值候选实体栅格化，用于接口联调、规则回归和学生模型早期训练。

CARLA 栅格化模式只是结构代理，不作为真实摄像头与 LiDAR 感知精度结论。

张量接口为：

```text
camera_bev        [B, camera_channels, H, W]
lidar_bev         [B, lidar_channels, H, W]
ego_features      [B, 8]
candidate_features[B, N, 12]
candidate_mask    [B, N]
intent_tokens     [B, L, intent_dim]
intent_mask       [B, L]
```

## 模型结构

默认配置：

- BEV 双分支卷积编码器；
- 统一维度 `256`；
- BEV token 网格 `8 x 8`；
- 2 个查询 token，分别用于高层动作和目标实体；
- 4 层 Cross-Attention，可配置为 4 至 6 层；
- 8 个注意力头；
- 动作、目标速度、目标车道、候选实体指针和置信度输出头。

动作空间沿用现有 CARLA 协议：

```text
keep_lane
accelerate
decelerate
stop
emergency_brake
lane_change_left
lane_change_right
turn_left
turn_right
```

## 输出与下游对接

学生模型先输出 `VLADecisionProposal 1.0`：

```json
{
  "schema_version": "1.0.0",
  "request_id": "request-1",
  "frame_id": "carla_100",
  "action": "lane_change_left",
  "target_speed_kmh": 18.0,
  "target_lane": "left",
  "target_location": null,
  "target_entity_id": "vehicle_front",
  "confidence": 0.91,
  "model": "lightweight-vla-modernbert-bev-base",
  "latency_ms": 12.4
}
```

`advance_vla_control_plan()` 的处理顺序：

1. 调用现有 `advance_control_plan()` 生成 canonical `ControlDecision`；
2. 依据风险、活动步骤和动作兼容表校验 VLA proposal；
3. 接受安全且语义一致的 proposal，或退回 canonical decision；
4. 输出原有 `ControlPlanState + ControlDecision`；
5. `ControlDecision` 可直接传给 `experiment/CARLA/control/protocol.py`。

因此下游 CARLA 控制器不需要修改。

在线推理必须通过 `LightweightVLAPipeline.from_checkpoint(...)` 加载训练后的学生权重。直接用随机初始化模型构造 Pipeline 时，`predict_proposal()` 会拒绝执行。完整在线链路不读取 `configs/teachers.json`，也不加载任何教师模型。

常驻服务启动后必须在接收请求前调用一次：

```python
pipeline.warmup(example_batch, iterations=30)
```

首次 CUDA 调用包含 kernel 和显存初始化时间，不能作为稳态推理时延。

## 教师和对照模型

`configs/teachers.json` 固定了三个离线模型：

| 名称 | 用途 | 实时链路 |
|---|---|---|
| UniDriveVLA-Base Stage 3 | Bench2Drive 轨迹主教师、CARLA 闭环参考 | 不进入 |
| OpenDriveVLA-0.5B | nuScenes 多模态开环教师、0.5B VLA 对照 | 不进入 |
| SimLingo | CARLA 语言-动作一致性教师、Bench2Drive 闭环参考 | 不进入 |

三者互补而不是相互替代：

- UniDriveVLA 为 CARLA/Bench2Drive 提供感知、决策和轨迹联合参考；
- OpenDriveVLA 接收 3D 环境感知、车辆状态和驾驶命令，适合验证多模态开环决策，但其官方权重需要先在 Hugging Face 接受访问条件；
- SimLingo 直接覆盖 CARLA 指令跟随和语言-动作一致性。官方代码使用 CARLA 0.9.15，因此只在独立环境生成离线标签，不与项目 CARLA 0.9.16 进程混装。

教师模型在各自官方仓库中运行。输出转换为统一 JSONL：

```json
{
  "sample_id": "sample-1",
  "model": "unidrivevla-base",
  "action_logits": [0.0, 0.1, 0.2, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
  "target_speed_kmh": 20.0,
  "latency_ms": 250.0,
  "trajectory": [[0.0, 0.0], [1.0, 0.1]]
}
```

训练损失包括：

```text
硬动作交叉熵
+ 教师动作 KL 蒸馏
+ 目标速度 Smooth L1
```

教师轨迹保留在统一记录中，用于对照实验和后续轨迹头训练，但当前实时学生只输出高层动作。

### 数据盘布局

教师代码和权重不提交 Git，服务器上的布局为：

```text
/root/autodl-tmp/external/teacher_repositories/
├── UniDriveVLA/   # a93c175
├── OpenDriveVLA/  # 10e8095
└── SimLingo/      # 743b243

/root/autodl-tmp/models/teachers/
├── UniDriveVLA_B2D_Base_Stage3/
│   └── UniDriveVLA_Stage3_Bench2drive_2B.pt
├── OpenDriveVLA-0.5B/
│   └── model.safetensors
└── SimLingo/simlingo/checkpoints/epoch=013.ckpt/
    └── pytorch_model.pt
```

已校验权重：

| 权重 | 大小 | SHA256 |
|---|---:|---|
| UniDriveVLA B2D Base Stage 3 | 5,923,425,558 B | `b127eebb319aa52dbd35f2099bb1bda5c1c7732c2f811cae88454ce15e6c23fc` |
| OpenDriveVLA-0.5B | 1,466,080,342 B | `73471d9e9b6cb4dc8b4dc4bf66eead27d1b4dc8113078694943c9f78e54d97de` |
| SimLingo epoch 13 | 2,569,679,322 B | `ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28` |

OpenDriveVLA safetensors 已读取到 2046 个张量；SimLingo 检查点已读取到 992 个张量。可再次检查路径和大小：

```bash
python lightweight_vla_adapter/scripts/verify_teacher_assets.py
```

需要逐字节复核哈希时运行：

```bash
python lightweight_vla_adapter/scripts/verify_teacher_assets.py --hash
```

## 环境配置

复用项目现有环境：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
cd /root/autodl-tmp/LMM-in-AutoDrive
pip install -r lightweight_vla_adapter/requirements.txt
```

已验证环境：

```text
Python 3.12
PyTorch 2.11.0+cu130
CUDA 13.0
CARLA 0.9.16
```

## 训练

训练数据文件为可信的批量 PyTorch tensor 字典，至少包含：

```text
camera_bev
lidar_bev
ego_features
candidate_features
candidate_mask
intent_tokens
intent_mask
action_targets
speed_targets
teacher_action_logits  # 可选
```

运行：

```bash
python lightweight_vla_adapter/scripts/train_student.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --dataset /root/autodl-tmp/datasets/vla_student/train_tensors.pt \
  --output /root/autodl-tmp/models/lightweight_vla_adapter/model.pt
```

## 离线推理

```bash
python lightweight_vla_adapter/scripts/run_offline_inference.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/model.pt \
  --request-json sample_request.json \
  --tensor-batch sample_tensors.pt \
  --output output/decision.json
```

输出文件同时包含：

```text
vla_proposal
control_plan_state
control_decision
```

## 教师对照

```bash
python lightweight_vla_adapter/scripts/evaluate_predictions.py \
  --student-jsonl output/student_predictions.jsonl \
  --teacher-jsonl output/unidrivevla_predictions.jsonl
```

输出当前学生与教师的动作一致率。比赛验收仍应以人工标签、指令满足度、安全违规率、闭环完成率和端到端时延为主，教师一致率不能替代真实指标。

## 测试

```bash
python -m unittest discover \
  -s lightweight_vla_adapter/tests \
  -p "test_*.py"
```

测试覆盖：

- SensorBundle 同步与字段校验；
- ModernBERT token 投影；
- 4 层 Cross-Attention 张量形状；
- CARLA WorldState 结构化 BEV 代理；
- 相反方向变道拒绝；
- 紧急风险覆盖模型输出；
- 现有 FSM 状态保持；
- 最终 ControlDecision 到 CARLA 协议对接；
- UniDriveVLA/OpenDriveVLA/SimLingo 教师 JSONL 适配。

## 已完成实验与结果

2026-07-27 在 RTX 5090、PyTorch 2.11.0+cu130、FP16、batch size 1 下完成：

| 项目 | 结果 |
|---|---:|
| Decision Adapter 参数量 | 4,039,182 |
| Adapter 平均时延 | 2.0608 ms |
| Adapter P50 | 2.0198 ms |
| Adapter P95 | 2.0771 ms |
| Adapter 最大值（200 次） | 5.7550 ms |
| 真实 ModernBERT 意图编码平均时延 | 11.0453 ms |
| 真实 ModernBERT 意图编码 P95 | 11.3727 ms |
| 预热后单次 Adapter 联调时延 | 2.1586 ms |
| 模块单元测试 | 16/16 通过 |

联调使用真实权重 `/root/autodl-tmp/models/modernbert-drive-command-compositional`。测试指令经 ModernBERT、轻量 Adapter、安全门和 FSM 后输出 `lane_change_left`，FSM 为 `ACTIVE`，最终 `ControlDecision` 被现有 CARLA 协议直接接受。

最初冷启动 Adapter 测得约 214 ms，原因是 CUDA kernel 和显存初始化；加入 `pipeline.warmup(..., iterations=30)` 后稳态约 2.16 ms。部署时延必须在预热后统计。

上述结果证明接口、运行时延和确定性安全兜底已打通；它不是学生模型任务精度结论。当前学生权重尚未经过教师伪标签蒸馏，因此不能用随机初始化输出申报决策准确率。

同一环境下还完成了上、下游回归：指令解析 113/113、场景理解 194/194、CARLA 控制与场景 31/31，全部通过。

## 当前实现边界

- 已实现独立模块、稳定接口、学生模型、蒸馏损失、三种教师记录适配、安全门和离线联调测试。
- 未在仓库内复制 BEVFusion、UniDriveVLA、OpenDriveVLA、SimLingo 源码及权重。
- `StructuredBEVRasterizer` 用于当前 CARLA 真值链路联调，不代表真实传感器融合结果。
- 实时链路只运行轻量学生；教师模型的秒级或百毫秒级时延不计入学生部署结果。
