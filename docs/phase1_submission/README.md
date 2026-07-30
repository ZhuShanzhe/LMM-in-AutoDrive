# XH-202602 第一阶段材料索引

本目录按照题目方案“基础赛道提交材料”整理当前 `main`。目标是让代码审阅、模块接入和后续材料补齐拥有统一入口，不把未完成内容写成已验收。

## 1. 技术方案文档

| 内容 | 当前入口 | 状态 |
|---|---|---|
| 总体架构和双链路 | 根目录 `README.md` | 已整理 |
| 语音解析 | `automatic_speech_recognition/README.md`、`structured_command_parser/README.md` | 已有模块说明 |
| 视觉理解与语义对齐 | `scene_understanding/README.md` | 已有模块说明 |
| 动作生成 | `scene_understanding/README.md`、`lightweight_vla_adapter/README.md` | 原链路和新链路均保留 |
| CARLA 执行 | `experiment/CARLA/README.md` | 已有环境与运行说明 |
| 完整正式技术方案 | 待新增独立文档 | 待补 |

## 2. 模型架构

- 指令解析：ModernBERT backbone、多任务头、实体/关系 token 头和确定性验证。
- 场景理解：YOLOP、YOLO11s、ByteTrack、CARLA 真值和可选异步 VLM。
- 语义对齐：结构化实体检索、目标唯一性、TTC 与风险规则。
- 原动作链路：canonical 规则动作、风险门控、ControlPlan FSM。
- 新动作链路：相机/LiDAR BEV、意图 token、候选实体、自车状态和 4 层 Cross-Attention Decision Adapter。

关键参数和模型下载方式记录在各模块 README；权重本体托管在 Hugging Face。

## 3. 多模态融合与场景适配

当前数据流：

```text
ASR text
  -> DrivingIntent
  -> camera/LiDAR/CARLA state
  -> entity grounding + risk
  -> canonical rule decision or gated VLA proposal
  -> FSM
  -> CARLA control
```

安全同步链路不等待异步 VLM 文本；碰撞、TTC、交通灯和车道安全不得只依赖视觉大模型回答。模糊指令、未解决指代、非法目标或过期感知结果由结构校验、拒绝和 canonical fallback 处理。

正式文档仍需补充动态权重公式、噪声实验、模糊指令分层统计和三类场景异常处理表。

## 4. 数据集构建

当前涉及的数据和报告：

- 指令解析：Talk2Car、SimLingo 及组合泛化增强；
- 场景理解：BDD100K、nuScenes、CARLA 域数据；
- baseline：DriveLM-CARLA、nuScenes-mini/full；
- CARLA：基础操控、行人横穿、紧急制动场景采集。

Git 只保留 Schema、配置、数据说明和少量示例。数据来源、规模、划分、增强、质量评估和随机样本需要在最终提交时形成独立数据集报告。

## 5. 模型与可执行代码

`main` 当前提供运行时源代码、依赖、配置和稳定接口。模型通过各模块 README 下载：

- `UNIC0RN-Zhu/modernbert-drive-command-base`
- `UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1`
- `UNIC0RN-Zhu/lightweight-vla-drive-decision-adapter-v10`

VLA 训练、蒸馏、数据构建和批量评测代码暂留 `zsz`，不进入本次 main 运行面。最终比赛复现包需要在审核后补充完整训练代码、参数统计及训练过程记录。

## 6. 仿真测试全量报告

当前已有：

- CARLA 0.9.16、Python 3.12.13 和 Linux/RTX 5090 环境说明；
- 基础直行、行人横穿和紧急制动代码入口；
- 指令解析、场景理解和轻量 VLA 的模块级结果摘要；
- `scene_understanding/PERCEPTION_EXPERIMENT_REPORT.md`；
- baseline 调研报告。

最终仍需补齐：

- 基础、复杂、极限场景完整时序数据；
- 任务完成率、解析准确率、对齐精度、碰撞、违规和端到端延时的统一计算；
- 异常记录与失败案例；
- 视频、指标曲线和决策流程可视化；
- 数据集适配性与泛化验证报告。

## main 内容边界

第一阶段 `main` 以运行和集成为主：

- 保留运行代码、接口、配置、最小示例和必要测试入口；
- 保留已完成报告和真实结果摘要；
- 不提交权重、数据集、缓存和逐样本大规模输出；
- 不把待补材料标记为完成；
- 不删除组员分支中的训练与实验历史。
