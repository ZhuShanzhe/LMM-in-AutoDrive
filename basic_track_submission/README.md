# 面向智能驾驶的大模型应用场景研究（南京大学）

本提交使用一套统一的多模态 VLA 架构完成三个题目场景。当前闭环验收输入为**文本指令**，不使用声音；在线策略输入包括文本、4 路同步 RGB、LiDAR BEV、自车/环境状态及前后向物理雷达，输出动作、目标速度、目标车道和风险，再由通用时序安全监督器与 Route PID 写入 CARLA 车辆控制接口。三个场景使用同一 Stage‑8 checkpoint、同一配置和同一控制链，不按场景 ID、事件 ID 或命令 ID切换模型。

## 1. 提交文件

- `image.tar`：镜像 `lmm-autodrive-basic:final`，包含官方 Bench2Drive 框架、CARLA 0.9.16、Python/CUDA 运行环境、代码、权重和命令解析模型，可断网独立运行；
- `weights/`：镜像内权重的外置副本与 SHA256，便于审计或替换；
- `面向智能驾驶的大模型应用场景研究-南京大学_技术方案.pdf`：方案、架构、数据与实测说明；
- `metrics.zip`：三个场景的完整结构化日志、运行清单、汇总和测试报告；
- 不提交 `demo.mp4`（题目允许“如有”提交，团队选择以可审计日志为准）。

## 2. Bench2Drive 基线与兼容性

镜像固定官方 `Thinklab-SJTU/Bench2Drive` 提交：

```text
2645714eb1f3a100217928dd113093cae0779f36
```

保留其 `leaderboard/`、`scenario_runner/`、220 路线、指标脚本和标准 `AutonomousAgent` 生命周期。团队 Agent 位于：

```text
/workspace/Bench2Drive/leaderboard/team_code/universal_vla_agent.py
```

Agent 实现 `setup()`、`sensors()`、`run_step()`、`destroy()`，只消费框架声明的物理传感器和自车/地图路线。官方基准通常使用 CARLA 0.9.15；本题三条长路线已在 CARLA 0.9.16 完整验证，因此镜像保留 0.9.16，并使用 Bench2Drive 的兼容 Leaderboard API。镜像标签中同时记录 Bench2Drive 提交、CARLA 版本和项目代码提交。

## 3. 运行要求

- Linux x86_64；
- NVIDIA GPU，主机驱动支持镜像内 PyTorch CUDA；
- Docker 24+ 与 NVIDIA Container Toolkit；
- 建议内存 64 GB、共享内存 32 GB、空闲磁盘 80 GB；
- 使用 `--network host`，CARLA RPC 默认端口 2000，Traffic Manager 默认端口 8000。

## 4. 加载与静态校验

```bash
docker load -i image.tar
docker image inspect lmm-autodrive-basic:final >/dev/null
mkdir -p outputs
docker run --rm --gpus all --network host --shm-size=32g \
  -v "$PWD/outputs:/workspace/outputs" \
  lmm-autodrive-basic:final \
  python3.12 /workspace/submission/tools/verify_runtime.py
```

预期输出包含 `RUNTIME_VERIFICATION_OK`、模型 SHA256、Bench2Drive 提交和 `cuda_available=true`。镜像内复现 Notebook 为：

```text
/workspace/submission/REPRODUCTION.ipynb
```

## 5. Bench2Drive 标准入口

以下命令使用官方 `drivetransformer_bench2drive_dev10.xml`，Evaluator 自动启动镜像内 CARLA，并把结果写到挂载的 `outputs/`：

```bash
docker run --rm --gpus all --network host --shm-size=32g \
  -v "$PWD/outputs:/workspace/outputs" \
  lmm-autodrive-basic:final \
  /workspace/submission/container/run_bench2drive.sh
```

切换完整 220 路线时增加环境变量：

```bash
-e ROUTES=/workspace/Bench2Drive/leaderboard/data/bench2drive220.xml
```

该入口用于验证框架适配，不把 dev10 smoke 写成题目三个长程场景的成绩。

## 6. 题目三个长程场景复现

先进入容器并启动 CARLA：

```bash
docker run --rm -it --gpus all --network host --shm-size=32g \
  -v "$PWD/outputs:/workspace/outputs" \
  lmm-autodrive-basic:final bash

cd /workspace/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port=2000 > /workspace/outputs/carla_server.log 2>&1 &
cd /workspace/LMM-in-AutoDrive
```

统一资产路径：

```bash
VLA=../models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt
CFG=lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json
PARSER=../models/modernbert-drive-command-compositional
```

场景一（Town04，5 km，15 条文本指令）：

```bash
python3.12 -u experiment/CARLA/run_control_experiment.py basic_voice_urban_5km \
  --host 127.0.0.1 --port 2000 \
  --scenario-config experiment/CARLA/configs/basic_voice_urban_5km.json \
  --decision-source vla_scene_bridge --command-parser-model "$PARSER" \
  --vla-checkpoint "$VLA" --vla-config "$CFG" \
  --vla-device cuda --vla-precision fp16 --target-speed-kmh 45 \
  --duration-s 3600 --goal-distance-m 5000 --stop-when-goal-reached \
  --output-dir ../outputs/scene1
```

场景二（Town05，8 km，70 车、21 行人和 4 类事件）：

```bash
python3.12 -u experiment/CARLA/run_complex_avoidance_town05.py \
  --host 127.0.0.1 --port 2000 --timeout 120 --duration 0 \
  --traffic-hybrid-physics --traffic-hybrid-radius-m 100 \
  --competition-run --competition-logs-only --external-ego-control \
  --output-dir ../outputs/scene2 --vla-checkpoint "$VLA" \
  --vla-config "$CFG" --command-parser-model "$PARSER" \
  --vla-device cuda --vla-precision fp16 --vla-decision-every-n 3
```

场景三（Town05，6 km，雨夜与 7 类紧急事件）：

```bash
python3.12 -u experiment/CARLA/run_emergency_response_6km.py \
  --host 127.0.0.1 --port 2000 --duration 0 --require-complete-scene \
  --event-variant cautious_sparse --output-dir ../outputs/scene3 \
  --camera-mode chase-only --presentation-lighting official-rainy-night \
  --record-ground-truth --ground-truth-every-n 5 \
  --ego-controller vla-route-pid --vla-checkpoint "$VLA" \
  --vla-config "$CFG" --vla-parser-model "$PARSER" \
  --vla-device cuda --vla-precision fp16 --vla-decision-every-n 3
```

## 7. 输入与输出接口

`UnifiedSensorBatch` 包含文本 token/mask、4 路 `[B,4,3,H,W]` RGB、`[B,4,64,64]` LiDAR BEV、自车和环境特征、相机/模态掩码；前后雷达作为可审计的物理安全观测进入同一控制器。前车闭合风险优先制动；后车 TTC 风险仅在低于道路/路线限速时允许加速，到达限速后仅在地图允许且目标侧视觉风险为 low 时变道。

输出 `vla_control_decisions.jsonl` 逐决策记录文本、模型提议、视觉风险概率、雷达候选、通用安全门、最终控制和端到端延迟；场景脚本同时生成 `commands/events/runtime/summary`。Bench2Drive 入口另生成官方 checkpoint JSON 和每条路线的 `agent_manifest.json`、`controller_summary.json`。

## 8. 已提交自测结果

| 场景 | 完成情况 | 安全 | 链路 |
|---|---|---|---|
| 场景一 r30 | 5 km，15/15 指令 | 0 碰撞，0 非法车道 | fallback=0，P95 17.82 ms |
| 场景二 r19 | 8000.913 m，15/15 指令，4/4 事件 | 0 碰撞，0 禁行线 | fallback=0，P95 33.34 ms |
| 场景三 r32 + r33 启动复核 | 6000.404 m，7/7 事件 | 0 碰撞，0 禁行线 | r33 启动 fallback=0 |

场景三 r32 提供完整 6 km 行车证据；首帧同步 RGB 未到达在当次记录为安全驻车，最终代码改为独立 warmup 计数，r33 20 秒 smoke 验证 `fallback=0`。未把两次运行伪装成同一次。所有原始证据和已知不足见 `metrics.zip/TEST_REPORT.md`。

模型 SHA256：

```text
53e949b37c84d6010ab45bfd473cb9d39a88cd89cd7729f55d3e9bb1baddaad3
```

部署配置 SHA256：

```text
40164752c522779330a2a2f68a869968eaacb075eb409bc91813143a3ef9c39e
```
