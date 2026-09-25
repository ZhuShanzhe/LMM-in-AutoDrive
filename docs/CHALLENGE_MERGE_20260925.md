# 0925 挑战分支核验与合并记录

## 来源与处理

| 成员 | 已核对分支及版本 | 合并范围 |
|---|---|---|
| 黄皓星 | `hhx-challenge-track` / `2a94f4e` | 相机编码尺寸、候选实体契约和配置驱动的局部模型计时工具 |
| 李畅锦 | `lcj-challenge-track` / `c48f4bb` | 选择性导入 `automatic_speech_recognition/`；不覆盖其分支携带的旧根文档、模型说明及其他模块说明 |
| 刘旭 | `lx-challenge` / `6051728` | 三场景配置、38 项任务判据、独立真值、同步采集/回放、控制反馈及测试 |
| 王皓然 | `whr-adjust-speed-contract` / `5d46831` | ADJUST_SPEED 的相对目标速度映射与回归 |
| 王皓然 | `whr` / `d0ac69b` | 仅增加一个占位 README，没有新增挑战实现，不重复导入 |

目标是既有的 `challenge-track`，没有另建拼写为 `challeng-track` 的公共分支。合并在独立工作区完成，未覆盖 `zsz` 中未提交的实验及暂停的量化工作，未改 `main`。

## 集成修复

- 保留黄皓星重新实现的 `benchmark_latency.py`。其输入为合成张量，测量范围不含语言模型、记忆头、安全门或 CARLA，不视为全链路成绩。
- 三项测试从已删除的 `challenge_sequence_v2.json` 改用当前 `challenge_signal_generalization.json`。
- 核对三场景 38 项任务的动作、速度和激活位置后，更新判据对当前源配置的 SHA256 绑定；逐文件比较确认除指纹外所有判据、阈值、覆盖声明均未改变。保留严格指纹检查，并固定配置文件的 LF 换行。
- 补齐旧测试替身的路口字段；按刘旭现有生产实现更新测试，验证预置行人触发移动、禁止隐式更换已销毁演员、禁止用瞬移代替行走。没有修改生产控制逻辑来迁就测试。
- 修复 ASR 的包内导入，支持 README 中的仓库根目录公开入口，不再依赖全局 `src` 包名；新增不加载模型的接口测试。

## 验证边界

本轮在无卡 AutoDL 执行 CPU 逻辑/接口回归，未启动 CARLA、未下载语音权重、未恢复 J6P/x86 仿真，也未验证模型精度或目标平台时延。ASR 检查仅覆盖语法、公开包导入和模拟识别/翻译结果的接口传递，不等于实际音频识别通过。

首轮存在配置路径、过期指纹、旧测试契约及三个超过 50 ms 的规则解析计时子测试失败；修复集成问题后保留原计时阈值复测。首轮日志不删除，不把一次复测通过解释为无卡环境的时延保证。

最终完整回归为 **1346 项测试、207 个子测试通过，耗时 52.74 秒**。这是测试套件总耗时，不是模型推理时延。最终日志为 `/root/autodl-tmp/review_logs/20260925/merged_cpu_tests_v4.log`；此前失败日志保留在同目录。ASR 额外语法检查覆盖 59 个成员提交的 Python 文件。

测试使用原 `command_parser` 环境及独立的 `/root/autodl-tmp/validation_deps/asr_merge_0925` 导入依赖目录，未覆盖原环境依赖。该目录仅为本次接口检查准备 librosa、soundfile、cffi、pycparser、lazy_loader，不代表完整 ASR 运行环境。

```bash
cd /root/autodl-tmp/worktrees/challenge-track
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/root/autodl-tmp/validation_deps/asr_merge_0925:$PWD/experiment/CARLA:$PWD
/root/autodl-tmp/conda_envs/command_parser/bin/python -m pytest -q \
  structured_command_parser/tests scene_understanding/tests \
  lightweight_vla_adapter/tests experiment/CARLA/tests \
  automatic_speech_recognition/tests/test_package_contract.py
```

## 仍需负责人收尾的明确问题

1. **语音部署：**`src/asr/deployment/hb_convert.py` 的默认芯片为 `bayes`，检测到 `hb_compile` 后仍固定调用 `hb_mapper`；不能作为已完成 J6P/Nash-P 适配的证明。由李畅锦修正，当前不启用此路径作为提交默认部署。
2. **语音接口：**要求英文输出但没有译文时，`ASRPipeline._select_output` 会回退为中文；`success` 仅由输出是否非空决定，`guard_safe` 另行传递。正式接入须明确拒绝/澄清与语言检查，不能只检查 `success`。由李畅锦与朱善哲确认边界。
3. **仿真验收：**刘旭记录中多个验收由 Traffic Manager/BehaviorAgent 或独立夹具执行，不能算 ModernBERT/VLA 全链路通过；正式场景仍有 `benchmark_ready=false`、部分语义覆盖与未验收项。由刘旭组织真实公共链路测试，王皓然处理执行侧问题。
4. **模型分发：**刘旭提交的 9 月 20 日资产审计记录有三项指定模型缺失和一项 README 不匹配。服务器有模型不等于每个组员已拿到；由朱善哲提供同版本文件与哈希，不用旧模型替换来获得“加载成功”。
5. **压缩成绩：**朱善哲个人工作区的候选量化与暂停结果未纳入默认链路；黄皓星本次新增的是接口与测量能力，并无新的目标平台性能报告。不能据此宣称 FLOPs、功耗、512 MB 内存或板端响应已达标。

这次合并交付的是公共开发与收尾基线，不是最终挑战赛道验收结论。简明分工见 [0925 收尾任务](../program/task_0925.md)。
