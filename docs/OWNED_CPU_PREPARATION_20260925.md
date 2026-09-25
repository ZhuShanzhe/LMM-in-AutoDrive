# 朱善哲模块无卡准备与验证记录

## 范围

基于挑战联合基准 `169632263e21e32b84943607d537629f6df4e923`，仅处理语言解析、VLA 模型独立加载、权重分发及 CPU 检查入口。未修改其他组员算法，未运行 GPU 训练、CARLA 驾驶或 J6P 仿真。Python 3.12.13、PyTorch 2.11.0+cu130 在 CPU 上执行，线程数为 2。

## 权重分发

匿名下载成功，不需要额外链接或 Token。来源和不可变版本存于 `lightweight_vla_adapter/configs/challenge_asset_sources.json`，文件大小和 SHA256 仍使用原有 `challenge_assets.json`，没有降低校验要求。

- 挑战权重：`twlk666/lmm-autodrive-challenge-assets`。
- 解析模型：`UNIC0RN-Zhu/modernbert-drive-command-base`。
- 23 个文件共 346,686,908 字节，全部通过校验；关闭网络后再次执行，23 个文件均复用并校验通过。
- ModernBERT 最新 README 与冻结清单不一致，使用匹配清单的历史提交，不修改模型文件或原清单。
- HF 仓库里的旧 `sequence_policy_v2.pt` 不在下载清单中。
- 服务器模型目录：`/root/autodl-tmp/models/challenge-assets-pinned`。

下载及配置命令见 [模型说明](../models/README.md)。此清单不包含独立 ASR、感知所需的全部资产，不代表整个系统资产已经补齐。

## 检查结果

| 项目 | 实际结果 |
|---|---|
| 解析与 VLA 回归 | 360 项测试、144 项子测试通过，40.18 秒 |
| 真实 ModernBERT CPU 加载 | 通过；参数量 149,051,184 |
| 指定 40 km/h | 输出 SET_SPEED，目标速度 11.111 m/s |
| 在红色卡车之前停车 | 保留 target_ref、BEFORE 与 STOPPED_BEFORE_TARGET |
| 不要左转，继续直行 | 输出 PROCEED/STRAIGHT |
| 非英文入口和空输入 | 拒绝，无静默接受 |
| phase_recovery 权重 | 严格加载；合成输入下 3 秒、30 步序列及运动学、授权契约检查通过 |
| Town01/04/05 配置 | 根据本地模型目录生成绝对路径配置 |
| 只打包本人两个模块 | 独立目录内离线检查通过，不依赖外部控制模块的提前导入 |

独立加载检查曾发现包初始化会提前导入 `scene_understanding` 控制模块。修复为按需加载，完整链路的三个公开控制导出仍保持同一对象，兼容测试通过。此次修复不改变控制规则或驾驶策略。

三个解析样例不用于计算准确率；序列使用合成输入，不作为实际跟车、安全性或舒适性证据。

## 轻量优化核验

FP32 模型副本融合 34 对卷积/归一化，所有张量输出满足 `rtol=3e-4, atol=3e-5`，最大绝对差约 `2.19e-6`。参数量从 5,251,633 降至 5,245,577，默认模型及运行配置未更改。

PyTorch CPU profiler 可计数 FLOPs 前后均为 6,605,523,171。该数只覆盖指定输入下 adapter 的可计数算子，不包含语言模型、记忆模块及未计数算子，不能当作全链路 FLOPs，也不能声称本次已显著降低计算量。

## 容器与验证边界

已添加独立 CPU Dockerfile、依赖清单、离线入口及只读权重挂载说明。服务器没有 Docker，未构建或执行镜像；独立 CPU 依赖清单没有在全新环境中重新安装验证，实际检查使用已有环境。容器、J6P 和整链路验收不计入本次通过项。

GPU 环境依赖补充与现有 PyTorch 对应的 torchvision，未改动服务器已有训练环境。完整调用方式见 [部署说明](../lightweight_vla_adapter/deployment/README.md)。

## 原始记录

服务器目录：`/root/autodl-tmp/review_logs/20260925/`。

- `download_owned.json`：首次下载和 SHA256。
- `download_owned_resume.json`：离线重复校验。
- `owned_tests_v2.log`：最终回归。
- `owned_cpu_runtime/report.json`：初次真实权重检查及解析 JSON。
- `owned_isolated_runtime_v2/report.json`：修复后独立包检查及优化对比。
- `owned_isolated.log`：首次隔离加载失败，保留故障记录。

2026-09-25 的实现与验证在服务器独立审核工作区 `challenge-review-20260925` 完成。2026-09-26 已将代码提交 `4365199` 快进合并并推送至 `challenge-track`；提交不含模型、缓存或临时测试包。服务器开机后，联合工作区 `/root/autodl-tmp/worktrees/challenge-track` 已同步该提交。合并后再次回归：360 项测试、144 项子测试通过，40.78 秒，记录为 `merged_owned_tests_0926.log`。未修改 `main`、`zsz` 或默认权重。
