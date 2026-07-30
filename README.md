# XH-202602 基础赛道提交包

本分支 `basic-track-submission` 用于整理“面向智能驾驶的大模型应用场景研究”基础赛道材料。源码、配置、Schema、测试和正式文档受 Git 管理；服务器生成的最终 ZIP 另外包含根目录 `models/` 中的预训练与最终训练权重，以及后续补入的全量 CARLA 日志和视频。

## 材料入口

[基础赛道提交材料/README.md](基础赛道提交材料/README.md) 按题目第六章六项要求组织：

1. 技术方案；
2. 四大模块架构；
3. 多模态融合与场景适配；
4. 数据集构建；
5. 模型、源码与复现；
6. CARLA 全量报告和佐证。

[提交清单](基础赛道提交材料/00_提交管理/提交清单.md) 明确区分已有、部分和待补材料。未找到的材料只保留目录和要求，不写成已完成。

## 系统架构

```text
中文语音
  -> DeepFilterNet3 / Qwen3-ASR
  -> 中文文本
  -> Qwen2.5-3B 术语约束翻译
  -> ModernBERT + 规则校验
  -> DrivingIntent 1.2
  -> 相机 / LiDAR / CARLA 状态
  -> 场景理解 + 实体对齐 + 风险评估
  -> 规则决策或轻量 VLA proposal
  -> 时序监督 + 确定性安全门 + FSM
  -> ControlDecision 1.0
  -> CARLA 协议和 PID
```

## 代码目录

| 路径 | 内容 |
|---|---|
| `automatic_speech_recognition/` | ASR、降噪、方言优化和翻译 |
| `structured_command_parser/` | ModernBERT 结构化指令解析 |
| `scene_understanding/` | 感知、实体对齐、风险和规则决策 |
| `lightweight_vla_adapter/` | 轻量 VLA 数据、训练、推理、评测和安全门 |
| `experiment/CARLA/` | CARLA 场景、控制、日志、指标和视频 |
| `docs/baseline_research/` | 技术选型和 baseline 调研 |
| `program/` | 题目原文及团队阶段文档 |

## 环境与模型

统一环境为 Ubuntu 22.04、Python 3.12.13、CARLA 0.9.16、CUDA 13.0 和 RTX 5090/sm_120。模块依赖不同，应按模块 README 建立环境。

完整 ZIP 的模型目录：

```text
models/
├── Qwen3-ASR-1.7B/
├── Qwen2.5-3B-Instruct/
├── modernbert-drive-command-compositional/
├── external/YOLOP/
├── scene_understanding/yolo11s_specialized_carla_v1/
├── lightweight_vla_adapter/v10/
└── pretrained/
    ├── DeepFilterNet3/
    ├── ModernBERT-base/
    └── yolo11s.pt
```

源码默认从提交包根目录的 `models/` 读取权重，不兼容旧服务器绝对路径。
需要在命令行中引用同一组路径时可执行：

```bash
source submission_env.sh
```

模型文件可通过 `sha256sum -c models/SHA256SUMS` 完整校验。模型 ID、
许可证和空缺参数见
[模型权重与参数统计](基础赛道提交材料/05_模型与可复现代码/模型权重与参数统计.md)。

## 获取提交包

普通 Git 分支不跟踪大权重，不能仅使用 `git archive` 作为最终邮件附件。服务器完成权重和全量佐证归档后，直接下载生成的：

```text
/root/autodl-tmp/XH-202602_基础赛道提交包.zip
```

提交前必须补齐报名表、正式三场景全量数据、统一指标、异常记录和视频，并完成敏感信息扫描。
