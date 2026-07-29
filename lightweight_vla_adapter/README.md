# 轻量多模态 VLA 决策适配模块

## 模块定位

本模块位于结构化指令解析、场景理解和 CARLA 控制器之间，用轻量学生模型完成场景条件化高层驾驶决策：

```text
ASR / 翻译
  -> structured_command_parser（DrivingIntent 1.2）
  -> 多视角相机 BEV + LiDAR BEV + 候选实体 + 自车状态
  -> ModernBERT 意图 token
  -> 4 层 Cross-Attention Decision Adapter
  -> VLADecisionProposal 1.0
  -> 确定性风险安全门
  -> ControlPlan FSM
  -> ControlDecision 1.0
  -> CARLA control.protocol
```

该结构属于模块化轻量 VLA：语言、视觉/点云和车辆状态在可训练的决策适配器中联合参与动作预测。大模型教师不进入默认实时链路；当前完整推理只需要 ModernBERT、轻量 VLA 权重、场景特征和既有安全控制模块。

## 对原架构的替代与接入关系

原链路的决策段为：

```text
DrivingIntent
  + WorldState
  -> driving_intent_alignment
  -> risk_assessment
  -> high_level_driving_actions / control_decision（规则映射）
  -> control_plan_executor（FSM）
  -> ControlDecision 1.0
  -> CARLA control.protocol
```

本模块替代或增强的是其中“仅依赖固定规则生成名义高层驾驶动作”的部分。它把 `DrivingIntent`、相机 BEV、LiDAR BEV、候选实体和自车状态联合编码为 `VLADecisionProposal 1.0`，使同一条语言指令能够根据当前场景产生不同的高层动作建议。

| 原模块或阶段 | v7 中的处理 | 是否保留 |
|---|---|---|
| `structured_command_parser` | 继续生成 `DrivingIntent 1.2`，并由 ModernBERT 意图编码器读取 | 保留，上游输入 |
| `scene_understanding` 感知与 `WorldState` | 转换为 `SensorBundle`、结构化 camera/LiDAR BEV、候选实体和自车特征 | 保留，上游输入 |
| `driving_intent_alignment` | 提供目标实体对齐和语义约束，限制模型不能自由编造目标 | 保留 |
| `high_level_driving_actions` 的纯规则名义动作选择 | 由轻量 VLA proposal 提供场景条件化建议；规则结果仍作为 canonical fallback | 替代名义建议，保留兜底 |
| `risk_assessment` 与风险门控 | 对 VLA proposal 做确定性覆盖和拒绝 | 完整保留 |
| `control_plan_executor` / FSM | 管理多步骤计划、触发条件和执行状态 | 完整保留 |
| `ControlDecision 1.0` 与 `CARLA control.protocol` | 输出协议不变，现有控制器直接消费 | 完整保留 |

因此，本模块不替代 ASR、翻译、指令解析、目标语义对齐、实时感知、轨迹规划、PID 或 CARLA 控制器，也不允许模型直接输出油门、制动和转向量。它接入现有链路的位置是：

```text
上游一：structured_command_parser
  DrivingIntent 1.2
                \
                 -> lightweight_vla_adapter -> VLADecisionProposal 1.0
                /
上游二：scene_understanding / CARLA perception
  SensorBundle + WorldState + RiskAssessment

VLADecisionProposal + canonical ControlDecision + RiskAssessment
  -> safety_bridge
  -> ControlPlanState + ControlDecision 1.0
  -> experiment/CARLA/control/protocol.py
  -> 规则规划器 / PID / CARLA
```

在独立链路中调用 `advance_vla_control_plan()`，由它生成 canonical decision、推进 FSM 并门控 proposal。在最新 `scene_bridge_policy` 已经生成 canonical decision 并推进 FSM 的链路中，只调用 `gate_vla_proposal()`，禁止再次调用 `advance_vla_control_plan()`，避免同一帧重复推进状态机。

安全门和 FSM 始终位于学生模型下游。学生模型不能：

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
├── examples/
├── schemas/
│   ├── sensor_bundle.schema.json
│   └── vla_decision_proposal.schema.json
├── scripts/
│   ├── build_proxy_dataset.py
│   ├── build_simlingo_tensor_shard.py
│   ├── build_carla_capture_dataset.py
│   ├── download_simlingo_full.sh
│   ├── audit_simlingo_download.py
│   ├── merge_tensor_datasets.py
│   ├── rebalance_tensor_dataset.py
│   ├── subsample_tensor_dataset.py
│   ├── audit_simlingo_dreamer.py
│   ├── train_student.py
│   ├── evaluate_student_checkpoint.py
│   ├── evaluate_carla_capture.py
│   ├── benchmark_end_to_end_latency.py
│   └── run_offline_inference.py
├── src/
│   ├── bev_encoder.py
│   ├── contracts.py
│   ├── decision_adapter.py
│   ├── intent_encoder.py
│   ├── pipeline.py
│   ├── safety_bridge.py
│   └── structured_bev.py
└── tests/
```

数据集、模型权重、检查点和运行结果受仓库根目录 `.gitignore` 管理，不提交 Git。

## 环境配置

已验证环境：

```text
Linux
Python 3.12.13
PyTorch 2.11.0+cu130
CUDA 13.0
GPU RTX 5090（sm_120）
CARLA 0.9.16
```

复用项目环境：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
cd /root/autodl-tmp/LMM-in-AutoDrive
pip install -r lightweight_vla_adapter/requirements.txt
```

模型默认放在数据盘：

```text
/root/autodl-tmp/models/modernbert-drive-command-compositional/
/root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt
```

VLA 检查点不提交 Git。当前最终检查点大小约 16 MB，SHA256：

```text
e201f79750ebbcabf4eac27a2836aa5d34ded64422b97da6943498db887d1d8c
```

加载时必须同时使用仓库中的 `configs/student_base.json`。ModernBERT 权重的下载和接口说明见 `structured_command_parser/README.md`。

## 输入接口

### DrivingIntent

直接接收 `structured_command_parser` 输出的 DrivingIntent 1.2。`ModernBertIntentEncoder` 使用同一 ModernBERT backbone 输出 768 维 token 特征；默认冻结 backbone，只训练轻量 VLA。

### SensorBundle

`SensorBundle 1.0` 对齐多视角图像、激光雷达、车辆状态和候选实体：

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

传感器时间戳默认最大偏差为 `100 ms`。`target_entity_id` 只能从 `candidate_entities` 中选择，不允许语言模型自由生成对象。

### 张量批次

```text
camera_bev         [B, 8, H, W]
lidar_bev          [B, 4, H, W]
ego_features       [B, 8]
candidate_features [B, N, 12]
candidate_mask     [B, N]
intent_tokens      [B, L, 768]
intent_mask        [B, L]
```

支持两种上游：

1. 真实感知模式：接收 BEVFusion、LSS、PointPillars 或其他编码器生成的 `camera_bev` 和 `lidar_bev`。
2. CARLA 联调模式：`StructuredBEVRasterizer` 将 `WorldState` 和 CARLA 真值候选实体栅格化。

## 模型结构

默认配置：

- 相机和 LiDAR 双分支卷积 BEV 编码器；
- 隐藏维度 256，BEV token 网格 `8 x 8`；
- 2 个查询 token，分别预测高层动作和目标实体；
- 4 层 Cross-Attention、8 个注意力头；
- 动作、目标速度、目标车道、候选实体指针和置信度输出头；
- Decision Adapter 共 4,170,254 个参数。

动作空间：

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

## 输出接口

学生模型输出 `VLADecisionProposal 1.0`：

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
  "latency_ms": 2.5
}
```

独立链路下游调用 `advance_vla_control_plan()`：

1. 现有 `advance_control_plan()` 生成 canonical `ControlDecision`；
2. 安全桥依据风险、活动步骤和动作兼容表校验 proposal；
3. 接受安全且语义一致的 proposal，或退回 canonical decision；
4. 输出原有 `ControlPlanState + ControlDecision`；
5. `ControlDecision` 直接传给 `experiment/CARLA/control/protocol.py`。

若接入已有 `experiment/CARLA/control/scene_bridge_policy.py`，scene bridge 已完成步骤 1 和 FSM 推进，此时只调用 `gate_vla_proposal(proposal, canonical_decision, risk_assessment)`，返回被安全门接受或回退后的 `ControlDecision`。

下游 CARLA 控制器不需要修改。完整离线推理输出同时包含：

```text
vla_proposal
control_plan_state
control_decision
```

在线服务必须从检查点加载，并在接收请求前预热：

```python
import json
import torch

from lightweight_vla_adapter.src.decision_adapter import (
    LightweightDecisionAdapter,
)
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline

with open(
    "lightweight_vla_adapter/configs/student_base.json",
    encoding="utf-8",
) as handle:
    config = json.load(handle)

model = LightweightDecisionAdapter(
    camera_channels=config["camera_channels"],
    lidar_channels=config["lidar_channels"],
    candidate_dim=config["candidate_dim"],
    ego_dim=config["ego_dim"],
    intent_dim=config["intent_dim"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    dropout=config["dropout"],
    bev_grid=tuple(config["bev_grid"]),
)
pipeline = LightweightVLAPipeline.from_checkpoint(
    model,
    "/root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt",
    model_name=config["model_name"],
    device="cuda",
    dtype=torch.float16,
)
pipeline.warmup(example_batch, iterations=30)
```

随机初始化的 Pipeline 会拒绝在线推理。首次 CUDA 调用含 kernel 和显存初始化开销，不能作为稳态时延。

## 数据构建与训练

训练张量至少包含：

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
lane_targets
unsafe_mask
teacher_action_logits  # 可选
```

数据按 route 稳定哈希划分为训练、验证和测试，避免同一路线帧跨集合泄漏。SimLingo 官方仓库审计得到 182 个目标文件、约 1.174 TB；完整下载脚本以文件数、文件大小、缺失文件、临时文件和大小不一致五项严格校验作为完成条件，网络失败会断点续传，不能仅凭 Hugging Face CLI 的退出码判定完成。

v7 训练只使用已经完整且能够按同帧键对齐的 raw/Dreamer 归档。新发现的 14 个有效配对分片共构建 218,447 条样本，再按固定种子确定性抽样 100,000 条参与训练；3 个归档因不存在同帧键被拒绝，另有 12 个 raw 归档没有对应 Dreamer 文件，均未生成伪配对数据。

全量下载和严格审计：

```bash
source /etc/network_turbo
cd /root/autodl-tmp/LMM-in-AutoDrive
nohup bash lightweight_vla_adapter/scripts/download_simlingo_full.sh \
  > /root/autodl-tmp/logs/simlingo_full_download.log 2>&1 &

python lightweight_vla_adapter/scripts/audit_simlingo_download.py \
  --target /root/autodl-tmp/datasets/vla_student/simlingo_hf \
  --output /root/autodl-tmp/datasets/vla_student/simlingo_hf/download_audit.json \
  --require-complete
```

审计成功必须同时满足：目标文件为 182 个、缺失文件为 0、`.incomplete` 文件为 0、文件大小不一致为 0。`--require-complete` 未满足时退出码为 2。

下载脚本默认使用 4 个并行 worker；网络不稳定时可在命令前设置
`MAX_WORKERS=2`，带宽充足时可测试 `MAX_WORKERS=6`。多个下载进程共享同一
Hugging Face 缓存和本地目录时，应使用互不重叠的 `--include` 文件范围。

| 数据 | 样本数 | 用途 |
|---|---:|---|
| v6 replay | 110,179 | 保持已验证的 9 类动作与 CARLA 域能力 |
| 新增 SimLingo 配对样本 | 100,000 | 从 218,447 条有效新样本确定性抽样 |
| CARLA 0.9.16 域适配样本 | 10,032 | 3 类场景训练样本重复采样 |
| v7 训练集总计 | 220,211 | 最终训练，CARLA 样本占约 9.1% |
| v7 组合验证集 | 27,712 | replay 验证集与 CARLA 验证集，只用于选模 |
| CARLA 独立测试集 | 908 | 最终测试，不参与训练和选模 |
| SimLingo 官方 validation 独立测试集 | 24,361 | 3 组官方 validation raw/Dreamer 配对，不参与训练和选模 |

训练命令：

```bash
python lightweight_vla_adapter/scripts/train_student.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --dataset /root/autodl-tmp/datasets/vla_student/full_training_v7/train.pt \
  --validation-dataset /root/autodl-tmp/datasets/vla_student/full_training_v7/validation.pt \
  --output /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt \
  --metrics-output /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/train_metrics.json \
  --init-checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/carla_adapted_v6/model.pt \
  --epochs 8 \
  --batch-size 256 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --patience 3 \
  --class-balance none \
  --seed 2027 \
  --device cuda \
  --precision bf16
```

## 推理与评测

离线推理：

```bash
python lightweight_vla_adapter/scripts/run_offline_inference.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt \
  --request-json sample_request.json \
  --tensor-batch sample_tensors.pt \
  --output output/decision.json \
  --device cuda \
  --precision fp16
```

独立张量测试：

```bash
python lightweight_vla_adapter/scripts/evaluate_student_checkpoint.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt \
  --dataset /root/autodl-tmp/datasets/vla_student/carla_domain_v1/test.pt \
  --output /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/evaluations/carla_test.json \
  --batch-size 256 \
  --device cuda \
  --precision bf16
```

端到端时延：

```bash
python lightweight_vla_adapter/scripts/benchmark_end_to_end_latency.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt \
  --language-model /root/autodl-tmp/models/modernbert-drive-command-compositional \
  --text "Keep the current lane." \
  --device cuda \
  --precision fp16 \
  --max-length 32 \
  --warmup 50 \
  --runs 500
```

CARLA 捕获序列联调：

```bash
python lightweight_vla_adapter/scripts/evaluate_carla_capture.py \
  --capture-index experiment/CARLA/outputs/runs/vla_domain_adaptation_v1/test/straight_driving/scene_understanding/capture_index.jsonl \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/full_training_v7/model.pt \
  --language-model /root/autodl-tmp/models/modernbert-drive-command-compositional \
  --instruction "Keep the current lane." \
  --output output/carla_capture_eval.json \
  --device cuda \
  --precision fp16 \
  --max-length 32
```

## 已完成实验与结果

### 训练迭代

| 版本 | 主要变化 | 关键结果 |
|---|---|---|
| v1 | 9 类均衡代理数据 | proxy 测试准确率 99.44% |
| v2 | 加入 SimLingo route 数据并修正相对速度 | route 84.18%，宏 F1 70.64% |
| v3 | 加入 5 类场景和安全样本 | route 88.56%，core 84.30% |
| v4 | 扩展右转、无场景和随机天气 | core 84.81%，extra 81.36% |
| v5 | 安全类重采样和长尾校准 | route 87.76%，core 84.92%，extra 81.66% |
| v6 | v5 replay + CARLA 0.9.16 域适配 | CARLA 独立测试 98.68%，旧域基本保持 |
| v7 | 扩展 14 个有效 SimLingo 配对分片并保留 CARLA replay | 官方 validation 独立测试较 v6 提升 3.53 个百分点 |

### v7 独立测试

| 测试集 | 样本数 | 动作准确率 | 受支持类别宏 F1 | 速度 MAE | 车道准确率 |
|---|---:|---:|---:|---:|---:|
| CARLA 0.9.16 域测试 | 908 | 98.68% | 98.89% | 0.38 km/h | 100.00% |
| 9 类 proxy | 2,700 | 99.63% | 99.63% | 3.44 km/h | 99.85% |
| SimLingo route | 2,648 | 89.43% | 90.65% | 2.51 km/h | 98.56% |
| 5 类 core scenario | 5,247 | 85.15% | 74.82% | 5.34 km/h | 96.59% |
| 3 类 extra scenario | 4,930 | 82.17% | 74.14% | 7.16 km/h | 94.95% |
| 官方 validation 独立测试 | 24,361 | 79.70% | 70.05% | 8.43 km/h | 95.52% |

CARLA 测试集只含 `keep_lane`、`decelerate`、`emergency_brake` 三类，因此报告“受支持类别宏 F1”，不能将缺失的六类视为已在 CARLA 测试中验证。分项召回率：

| 类别 | 召回率 |
|---|---:|
| keep_lane | 100.00% |
| decelerate | 95.62% |
| emergency_brake | 100.00% |

域适配前的 v5 在同一 CARLA 独立测试集上只有 26.10% 动作准确率和 31.27% 受支持类别宏 F1。v6 提升到 98.68% 和 98.89%；v7 保持该结果，并将速度 MAE 从 0.43 降至 0.38 km/h。

v6 与 v7 在同一组 24,361 条官方 validation 独立测试上的对比：

| 模型 | 动作准确率 | 宏 F1 | 速度 MAE | 车道准确率 |
|---|---:|---:|---:|---:|
| v6 | 76.18% | 65.67% | 9.37 km/h | 95.06% |
| v7 | 79.70% | 70.05% | 8.43 km/h | 95.52% |

v7 相对 v6 的动作准确率提高 3.53 个百分点、宏 F1 提高 4.39 个百分点、速度 MAE 降低 0.94 km/h。官方独立测试中安全类样本准确率为 91.70%、宏 F1 为 82.42%；非安全类为 66.62% 和 50.57%。各类召回率为：keep_lane 73.66%、accelerate 79.17%、decelerate 87.16%、stop 92.18%、emergency_brake 83.58%、change_lane_left 80.30%、change_lane_right 50.94%、turn_left 32.24%、turn_right 40.65%。这些长尾结果按真实数值保留，未用总体准确率掩盖转向类别不足。

### 时延

RTX 5090、FP16、batch size 1、预热 50 次、统计 500 次：

| 阶段 | 平均 | P50 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| 分词 | 0.22 ms | 0.22 ms | 0.23 ms | 0.66 ms |
| ModernBERT + Adapter | 13.53 ms | 13.39 ms | 14.02 ms | 18.12 ms |
| 文本到 proposal 特征总计 | 13.75 ms | 13.61 ms | 14.25 ms | 18.34 ms |

在最新 CARLA 分支上完成两轮捕获序列复测，Adapter 平均时延为 2.23–2.37 ms，P95 为 2.25–2.38 ms，观测最大值 5.40 ms。配置预算为 Adapter 35 ms、完整决策 150 ms，均满足。

### CARLA 0.9.16 联调

规则控制器闭环共运行 12 次：首次 9 次成功 8 次，随后直行场景热启动复测 3/3 成功，总计 11/12。所有成功运行均无碰撞、超速和非法车道侵入。唯一失败为直行 route manager 随机选择错误分支并达到时长上限，属于现有场景路线稳定性问题。

v7 对独立捕获序列使用同一条 `Keep the current lane.` 指令：

| 场景 | 帧数 | VLA proposal | 安全门后结果 |
|---|---:|---|---|
| straight_driving | 100 | keep_lane 100 | keep_lane 98，decelerate 2 |
| emergency_brake | 43 | decelerate 43 | decelerate 43 |

同一语言指令在普通直行和紧急场景产生不同 proposal，验证场景特征参与了决策。最新 `origin/lx-main-integration`（测试提交 `0adea58`）中，直行 100 帧 proposal 全部为 keep_lane，安全门后为 keep_lane 98 帧、decelerate 2 帧；紧急制动 43 帧 proposal 与最终结果均为 decelerate。

此前 v6 还完成了 pedestrian_crossing 30 帧联调：proposal 为 keep_lane 30 帧，安全门后为 keep_lane 29 帧、decelerate 1 帧。v6 使用场景一致的复合指令时：

- `Slow down and stop for the pedestrian ahead.`：VLA 输出 29 帧减速、1 帧停车；
- `Brake immediately to avoid the obstacle ahead.`：VLA 输出 32 帧减速、11 帧紧急制动。

这两条复合指令由上游解析为 `SAFE_STOP` 阻塞策略，FSM 首帧进入 STOP 后保持，最终控制均为 STOP。该结果验证了 VLA proposal 与确定性 FSM 的职责边界，也记录了复合持续指令在联调中的状态机行为。

### 分支链路接入核验

在独立 worktree 中将本模块覆盖到最新 `origin/lx-main-integration@0adea58` 后完成了真实代码联调，没有修改或合并当前 `zsz` 分支。该分支新增 ASR ingress、在线感知、非阻塞 Qwen、scene bridge 和 CARLA HUD；其 `ControlDecision 1.0` 与 `experiment/CARLA/control/protocol.py` 和本模块输出兼容。同步后再次运行直行 100 帧和紧急制动 43 帧，proposal 与安全门后的动作分布与首次联调完全一致。

支持两种接入方式：

1. 将本模块产生的根级 `ControlDecision 1.0` JSON 交给现有 `json_file` provider，再进入 CARLA 控制协议。
2. scene bridge 已产生 canonical decision 时，仅调用 `gate_vla_proposal()` 对该决策进行安全约束，不再调用 `advance_vla_control_plan()`。

第二种方式必须避免重复推进 FSM；scene bridge 与 VLA 各推进一次会造成状态跳变。按上述边界接入后，最新分支的 VLA、指令解析和场景理解联合测试共 `355 passed, 123 subtests passed`，CARLA 测试 `67 passed`。

### 回归测试

```bash
python -m pytest lightweight_vla_adapter/tests -q
python -m pytest structured_command_parser/tests -q
python -m pytest scene_understanding/tests -q
cd experiment/CARLA
python -m pytest tests -q
```

结果：

```text
lightweight_vla_adapter: 19 passed
structured_command_parser: 113 passed, 82 subtests passed
scene_understanding: 194 passed, 41 subtests passed
experiment/CARLA: 31 passed
最新 lx-main-integration 覆盖联调: 355 passed, 123 subtests passed
最新 lx-main-integration CARLA: 67 passed
```

## 实现边界

- 当前 CARLA 域训练和捕获联调使用 `StructuredBEVRasterizer` 处理 CARLA `WorldState` 真值实体；它验证的是多模态决策接口和高层决策，不是原始相机检测精度。
- 当前 SimLingo 张量使用真实 LiDAR 点云栅格，camera BEV 仍主要来自官方 3D 标注/结构代理；接入真实 BEVFusion 后无需改变本模块张量和输出接口。
- CARLA 独立测试仅覆盖直行、行人横穿和紧急制动三类场景，不等同于题目全部场景的闭环验收。
- 98.68% 是上述独立 CARLA 高层动作测试准确率，不是完整 ASR、感知、规划和控制链路的总成功率。
- 官方 validation 独立测试的 79.70% 是当前更严格的跨归档泛化结果；右变道和左右转向仍是主要误差来源。
- 大模型教师只用于离线研究和对照，不是运行本模块的依赖，也不计入实时链路时延。
