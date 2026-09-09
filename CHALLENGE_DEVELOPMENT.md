# 挑战赛道开发入口

## 当前联调版本

`challenge-track` 已接入朱善哲的事件记忆与纵向序列运行代码、独立配置和接口测试，原链路保持默认。组员统一参考 [序列策略联调说明](lightweight_vla_adapter/CHALLENGE_HANDOFF.md)；候选权重放在服务器约定路径，不在 Git。个人训练与中间实验仍留在 `zsz` 工作区。本轮只更新挑战赛道分支，不回补 `main`，也不提交个人工作区的其他未完成实验。

下方记录原分支建立方式和稳定链路复用方法；新增模块的状态、结果与接口以本节链接为准。

## 分支关系

- `main`：基础赛道稳定链路与公共规划；本次只增加任务文档和经过回归验证的 CARLA 修复。
- `challenge-track`：挑战赛道集成分支，由同步最新 main 后的 zsz 建立，保留开发所需的训练工具。
- `zsz`：朱善哲个人开发分支。后续指令解析、VLA 压缩和 J6P 适配先在这里开发，再通过 Pull Request 合并到 `challenge-track`。

规划见 [路线及人员分工](program/task_0908.pdf)。仓库只保留此次规划的PDF，不提交TeX源文件。当前只是建立共同开发基线，尚未进行挑战赛道训练或 J6P 适配。

## 内容整理边界

挑战赛道分支移除已结束的独立baseline调研材料、旧第一阶段材料索引及DriveLM实验记录；这些内容仍保留在 `main` 和Git历史中。`program/` 中既有题目、路线和任务PDF不删除；本系统三场景测试报告继续用于压缩前后对照。运行代码、训练工具、模型权重及数据不在本次清理范围内。

## 服务器工作目录

| 目录 | 分支与用途 |
|---|---|
| `/root/autodl-tmp/LMM-in-AutoDrive` | `main`，基础赛道稳定工作区 |
| `/root/autodl-tmp/worktrees/zsz` | `zsz`，个人开发与测试 |
| `/root/autodl-tmp/worktrees/challenge-track` | `challenge-track`，挑战赛道联合验证 |

三个目录是同一个 Git 仓库的独立 worktree，不需要在同一目录频繁切换分支。修改前确认当前目录和 `git branch --show-current`。不要同时让多个进程写入同一模型或数据输出目录。

## 个人工作范围

- `structured_command_parser/`：ModernBERT、指令规范化、可组合意图、训练及回归测试。
- `lightweight_vla_adapter/`：当前多模态决策、真实传感器编码、历史状态和训练入口。
- `scene_understanding/`：场景特征、实体对齐、安全判断及下游接口依赖。
- `experiment/CARLA/`：三场景运行、控制、风险门与联调测试。

本次合并以 main 的现有文件作为最新运行基线，保留 zsz 独有的旧训练、数据构建和测试工具。旧 `train_student.py`、SimLingo 构建工具等属于历史开发资源，不代表已经适配新模型，更不是已完成的 J6P 训练流程。当前三场景实现与训练入口以模块 README 为准。旧配置与旧测试不能直接替代当前基线评测。

## 复用现有环境与权重

```bash
cd /root/autodl-tmp/worktrees/zsz
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/command_parser
export MODEL_ROOT=/root/autodl-tmp/models
source submission_env.sh
git status --short --branch
```

个人开发和挑战赛道工作区的 `models/` 已通过本机软链接复用现有 ModernBERT、场景理解及 VLA 权重；链接与权重不提交 Git。共享权重作为基线读取，新训练结果应另设目录，不覆盖既有权重。

现有数据位于 `/root/autodl-tmp/datasets/`，本次未重新下载数据集，也未复制数据或模型。各工具所需的数据版本与完整性需在训练前核验。J6P 工具链应使用独立环境，不能将当前 Python/CUDA 环境直接当作已经配置好的 J6P 环境。

## 基线回归

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  structured_command_parser/tests/test_scene_report_regressions.py \
  structured_command_parser/tests/test_rule_parser.py \
  structured_command_parser/tests/test_compositional_frame.py \
  lightweight_vla_adapter/tests/test_raw_multimodal_adapter.py \
  lightweight_vla_adapter/tests/test_temporal_supervisor.py \
  experiment/CARLA/tests/test_carla_multisensor_vla.py \
  experiment/CARLA/tests/test_unified_architecture.py \
  experiment/CARLA/tests/test_scene2_town05.py \
  experiment/CARLA/tests/test_emergency_response_6km.py \
  experiment/CARLA/tests/test_camera_video_cadence.py
```

以上覆盖解析、传感器/VLA接口、统一架构和本次CARLA修复，不代表重新完成了三场景实跑、模型性能评测或J6P验证。

## 本次CARLA修复

- 场景二 `--competition-logs-only` 保留显式视频选择；关闭视频使用 `--no-video`。演示摄像头按视频帧率采样，不改变在线VLA传感器。
- 场景三失效的间隙控制车辆不再被访问，但对象失效本身不会被记录为目标车道已释放。
- 场景三默认保留原来的辅助图像采集。只有显式传入 `--skip-audit-cameras` 才关闭辅助审计摄像头；在线VLA、安全传感器和明确请求的直接录制不受该开关影响。

## 合并方向

在 `zsz` 提交并推送后，创建以 `challenge-track` 为目标分支的 Pull Request。验证后的改动进入挑战赛道集成分支；只把确实需要回补的通用修复单独同步到 `main`。不通过强制推送覆盖他人的历史。
