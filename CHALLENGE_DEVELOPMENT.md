# 挑战赛道协作与验证

## 分支

- `challenge-track`：本轮经过回归的运行基准，组员据此适配。
- `zsz`：朱善哲个人实验、训练及原始证据；本轮未提交该工作区。
- `main`：原共享链路与历史材料；本轮未修改。

本分支没有数据集、训练检查点、录屏、原始大日志或基础赛道提交包。题目与路线规划 PDF 保留；CARLA 中仍被调用的基础场景代码保留，作为可复用联调设施。

## 同步与运行

在自己的干净工作区拉取 `challenge-track`，不在有未提交改动时强行切换或重置。权重不随 Git 下载，按照 [模型目录](models/README.md) 获取固定文件，再执行 [配置和校验](lightweight_vla_adapter/README.md)。

服务器联合工作区为 `/root/autodl-tmp/worktrees/challenge-track`，现有个人实验位于独立的 `zsz` 工作区。共享模型只读使用，不能覆盖基准模型。

## 本轮交接变化

相对上一版仅交接序列候选的版本，本次同步了递归事件记忆、跟车恢复、0.5 秒常规操作间隔、目标追踪、完整意图步骤执行、车道授权、当前图像信号灯判别及停止线约束。新配置与权重是固定的一组，旧 `challenge_sequence_v2.json` 不再作为入口。

保留原 V6 配置作为架构对照，但其结果不代表新序列模型；旧三场景脚本也不是新版已经完成全场景适配的证据。新版权重仅在 x86/CARLA 仿真运行，不能直接控制真实车辆。

## 回归入口

整理后的独立工作区联合回归为 **833 项测试、207 个子测试通过**。固定 `phase_recovery_v1.pt` 在 CUDA 上完成严格权重加载、递归记忆上下文、30 步序列积分和未授权隔离检查。23 个模型及配套文件全部通过大小与 SHA256 校验。此轮没有重新运行 14 次物理用例；对应运行源码与已测 `zsz` 版本保持一致，变化为分支整理、配置路径和无数据检查入口。

首次整理回归有 4 项因旧测试依赖被移走而失败；已恢复当前跟车审计、保留多模态运行测试，并将序列烟雾测试更新到当前分层头后重新全量通过。训练专用的数据源标签用例仍在 `zsz`，没有更改驾驶评测阈值来获得通过。

```bash
cd /path/to/LMM-in-AutoDrive
export PYTHONPATH="$PWD/experiment/CARLA:$PWD"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  lightweight_vla_adapter/tests experiment/CARLA/tests \
  structured_command_parser/tests scene_understanding/tests
```

接口测试不替代物理闭环。上游必须提供真实同步的传感器、时间戳和有效掩码，下游必须保持停车、红灯、目标车道风险及过期序列限制。不得把缺失模态填零后当成完整观测，也不得绕过风险门获得表面任务完成率。

闭环证据的范围和时延边界见 [根说明](README.md) 和 [新版记录](lightweight_vla_adapter/CHALLENGE_SIGNAL_GENERALIZATION.md)。新的集成测试结果只按实际运行计数，不沿用未运行的旧成绩。
