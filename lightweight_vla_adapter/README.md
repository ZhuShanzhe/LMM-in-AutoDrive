# 轻量多模态 VLA 决策适配模块

## 第一阶段 main 定位

本目录只保留可接入现有链路的运行时核心、配置、Schema、最小示例和离线推理入口。数据下载、张量构建、训练、蒸馏、教师模型、批量评测和开发测试继续保留在 `zsz` 分支，不进入当前 `main`。

模块位于结构化指令解析、场景理解和 CARLA 控制器之间：

```text
DrivingIntent 1.2
Camera BEV + LiDAR BEV
Candidate entities + Ego state
        |
        v
4-layer Cross-Attention Decision Adapter
        |
        v
VLADecisionProposal 1.0
        |
        v
TemporalProposalSupervisor
        |
        v
Deterministic safety gate + canonical fallback
        |
        v
ControlPlan FSM -> ControlDecision 1.0 -> CARLA controller
```

本模块只替代或增强原链路中的“名义高层动作建议”。以下部分均不被替代：

- `structured_command_parser` 的结构化指令；
- `scene_understanding` 的感知、目标语义对齐和风险判断；
- 原规则链路产生的 canonical decision；
- ControlPlan FSM、轨迹规划、PID 和 CARLA 控制协议。

因此，未安装 VLA 权重时可完整运行原规则链路；VLA 输入不完整、推理失败或安全门拒绝时也自动回退到原规则结果。

## 运行环境

已验证环境：

```text
Linux / Ubuntu 22.04
Python 3.12.13
PyTorch 2.11.0+cu130
CUDA 13.0
RTX 5090 / sm_120
CARLA 0.9.16
```

复用项目数据盘环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda create -p /root/autodl-tmp/conda_envs/vla_runtime \
  python=3.12.13 -y
conda activate /root/autodl-tmp/conda_envs/vla_runtime

cd /root/autodl-tmp/LMM-in-AutoDrive
python -m pip install --upgrade pip
pip install -r lightweight_vla_adapter/requirements.txt
pip install -r structured_command_parser/requirements-modernbert.txt
```

## 下载模型

最终 v10 权重：

```text
https://huggingface.co/UNIC0RN-Zhu/lightweight-vla-drive-decision-adapter-v10
```

该仓库采用自动批准访问门控。首次下载时：

1. 登录 Hugging Face；
2. 在模型页面申请访问；
3. 运行 `hf auth login`；
4. 下载到数据盘。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/vla_runtime

python -m pip install --upgrade huggingface_hub

if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi

hf auth login

MODEL_DIR=/root/autodl-tmp/models/lightweight_vla_adapter/v10
mkdir -p "$MODEL_DIR"

hf download \
  UNIC0RN-Zhu/lightweight-vla-drive-decision-adapter-v10 \
  --repo-type model \
  --local-dir "$MODEL_DIR"
```

校验权重：

```bash
cd /root/autodl-tmp/models/lightweight_vla_adapter/v10
sha256sum -c model.sha256
```

预期：

```text
model.pt: OK
```

权重 SHA256：

```text
0f813842ec36ef1b3d4d80a9013a83531c522dafc3f42723c3e22e55f6e567b6
```

本模块读取 ModernBERT 意图 token。若上游解析模块尚未安装，再按照 [structured_command_parser/README.md](../structured_command_parser/README.md) 下载：

```text
UNIC0RN-Zhu/modernbert-drive-command-base
```

推荐路径：

```text
/root/autodl-tmp/models/modernbert-drive-command-compositional
```

## 模型结构

`configs/student_base.json` 中的默认结构：

| 参数 | 值 |
|---|---:|
| camera channels | 8 |
| LiDAR channels | 4 |
| candidate feature | 12 |
| ego feature | 8 |
| intent token | 768 |
| hidden size | 256 |
| Cross-Attention layers | 4 |
| attention heads | 8 |
| BEV grid | 8 x 8 |
| max candidates | 32 |

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

## 时序稳定运行时

v10 模型本身仍是单帧 Decision Adapter。运行时在原始 proposal 与最终安全门之间加入 `TemporalProposalSupervisor`，不修改权重或 JSON 协议：

- 更保守的减速、停车和紧急制动立即生效；
- 普通动作切换需连续 3 帧确认，重新加速需连续 5 帧确认；
- 建议速度下降立即生效，上升默认限制为 `8 km/h/s`；
- 同车道前车闭合且 TTC 不足时禁止继续建议加速；
- `RiskAssessment` 要求减速或紧急制动时在 proposal 阶段立即约束；
- 目标实体切换需连续 3 帧确认，旧目标离开候选集合后立即失效；
- 时间戳倒退或帧间隔超过 2 秒时自动清理旧状态。

该监督器是确定性进程内状态，不是经时序数据训练的 RNN/Transformer。最终安全门和 ControlPlan FSM 继续作为不可绕过的下游边界。

## 输入接口

上游必须提供：

```text
DrivingIntent 1.2
WorldState
semantic alignment
RiskAssessment
candidate entity IDs
camera BEV
LiDAR BEV
ego features
candidate features and mask
ModernBERT intent tokens and mask
```

张量接口由 `SensorTensorBatch` 定义：

```python
from lightweight_vla_adapter.src.contracts import SensorTensorBatch

batch = SensorTensorBatch(
    camera_bev=camera_bev,
    lidar_bev=lidar_bev,
    ego_features=ego_features,
    candidate_features=candidate_features,
    candidate_mask=candidate_mask,
    intent_tokens=intent_tokens,
    intent_mask=intent_mask,
)
```

`examples/sensor_bundle.example.json` 和 `schemas/sensor_bundle.schema.json` 描述传感器侧结构。目标实体必须来自候选集合，模型不能自由编造对象。

## 加载与调用

```python
import json
import torch

from lightweight_vla_adapter.src.decision_adapter import (
    LightweightDecisionAdapter,
)
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline

model_dir = "/root/autodl-tmp/models/lightweight_vla_adapter/v10"

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
    f"{model_dir}/model.pt",
    model_name=config["model_name"],
    device="cuda",
    dtype=torch.float16,
)

pipeline.warmup(example_batch, iterations=30)

proposal, plan_state, control_decision = pipeline.decide(
    batch,
    driving_intent,
    world_state,
    semantic_alignment,
    risk_assessment,
    candidate_entity_ids=candidate_entity_ids,
    prior_state=prior_state,
    feedback=feedback,
)
```

`pipeline.decide()` 默认启用时序监督。新指令、路线结束或场景重置时清理对应状态：

```python
pipeline.reset_temporal_state(driving_intent["request_id"])
```

若调用方绕过 `decide()` 而直接调用 `predict_proposal()`，必须同时传入 `world_state`、`risk_assessment` 和稳定的 `stream_id`，否则返回的是原始单帧 proposal。最近一帧的监督原因可通过以下接口写入运行日志：

```python
diagnostics = pipeline.temporal_supervisor.diagnostics(
    driving_intent["request_id"]
)
```

在线服务必须先预热。随机初始化的 Pipeline 会拒绝在线推理。

完整离线入口：

```bash
python lightweight_vla_adapter/scripts/run_offline_inference.py \
  --config lightweight_vla_adapter/configs/student_base.json \
  --checkpoint /root/autodl-tmp/models/lightweight_vla_adapter/v10/model.pt \
  --request-json /path/to/request.json \
  --tensor-batch /path/to/tensor_batch.pt \
  --output /path/to/decision.json \
  --device cuda \
  --precision fp16
```

## 输出接口

学生模型先输出 `VLADecisionProposal 1.0`：

```json
{
  "schema_version": "1.0.0",
  "request_id": "request-1",
  "frame_id": "carla-100",
  "action": "lane_change_left",
  "target_speed_kmh": 18.0,
  "target_lane": "left",
  "target_location": null,
  "target_entity_id": "vehicle-front",
  "confidence": 0.91,
  "model": "lightweight-vla-modernbert-bev-base",
  "latency_ms": 2.5
}
```

`pipeline.decide()` 返回：

```text
vla_proposal
control_plan_state
control_decision
```

下游仍消费原有 `ControlDecision 1.0`，CARLA 控制器不需要修改。

## 已完成结果

v10 权重的完整原始报告位于 Hugging Face `evaluations/`：

| 测试集 | 样本数 | VLA 原始动作准确率 | 安全门后 canonical 动作准确率 |
|---|---:|---:|---:|
| 固定官方 validation 测试 | 24,361 | 79.90% | 91.77% |
| 广义官方 validation 测试 | 82,262 | 84.18% | 94.66% |
| CARLA 三类域测试 | 908 | 98.68% | 100.00% |

安全门后结果包含确定性 fallback，不是学生模型单独准确率。CARLA 集只覆盖 `keep_lane`、`decelerate`、`emergency_brake`，不代表九类动作均达到 98.68%。

RTX 5090、FP16、batch size 1、预热后 500 次：

```text
文本到 proposal 特征 P95：17.30 ms
最大值：21.22 ms
```

该时延不包含 ASR、原始相机/LiDAR 感知、规划器和控制器。

230 帧确定性连续压力序列用于单独验证时序监督层，不计作模型准确率：

| 指标 | 原始逐帧 proposal | 时序监督后 |
|---|---:|---:|
| 动作切换次数 | 83 | 4 |
| 目标实体切换次数 | 229 | 0 |
| 危险帧错误加速 | 51 | 0 |
| 紧急风险响应 | 不适用 | 同帧 |

时序监督层 CPU 平均额外耗时 `0.023 ms`，P95 `0.026 ms`，最大 `0.102 ms`。

## 文件范围

```text
lightweight_vla_adapter/
├── configs/student_base.json
├── examples/sensor_bundle.example.json
├── schemas/
├── scripts/run_offline_inference.py
├── src/
├── requirements.txt
└── README.md
```

`main` 不包含训练、蒸馏、教师、数据构建、批量评测、测试数据或模型权重。

## 许可与安全

- v10 权重使用 SimLingo 数据，采用自定义非商业研究许可。
- 仅用于非商业学术研究，不得用于真实车辆或机器人运行及高风险部署。
- 模型不能绕过紧急制动、目标车道安全、语义失败或 FSM 安全兜底。
- 时序监督层改善连续输出稳定性，但不改变单帧模型本身的分类准确率。
- 当前相机 BEV 主要来自官方三维标注或结构代理，不能替代真实 BEV 感知验收。
- 模型及当前实验结果不构成自动驾驶安全认证。
