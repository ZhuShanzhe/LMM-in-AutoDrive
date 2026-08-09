# 面向智能驾驶的大模型应用场景研究 - 基础赛道复现说明

本提交使用一套通用多模态 VLA 架构处理三个 CARLA 场景。当前验收仅使用文本指令，不使用声音输入。在线输入为文本、RGB 相机、车辆状态、环境状态；场景二额外使用 LiDAR BEV。模型统一输出驾驶动作、目标速度、目标车道、置信度和视觉风险，再由通用时序安全监督器、指令 FSM 与 Route PID 连接到 CARLA 车辆控制接口。在线策略不读取场景 ID、事件 ID、命令 ID 或 CARLA actor 真值。

## 1. 文件

- `image.tar`：基于 Bench2Drive/CARLA 0.9.16 的完整 Docker 镜像，包含运行环境和代码。
- `weights/`：仅当权重未封装进镜像时使用，目录结构见 `weights/README.md`。
- `面向智能驾驶的大模型应用场景研究-南京大学_技术方案.pdf`：技术方案、模型架构与数据说明。
- `metrics.zip`：三个场景的自测摘要、指令、事件、风险决策统计和关键运行日志；不含演示视频。

## 2. 环境要求

- Linux x86_64；
- NVIDIA GPU，驱动支持镜像内 CUDA；
- Docker 24+，已安装 NVIDIA Container Toolkit；
- 建议内存 64GB、共享内存 16GB 以上、可用磁盘 120GB 以上；
- CARLA RPC 使用主机端口 `2000`，Traffic Manager 使用 `8000`。

## 3. 加载镜像

```bash
docker load -i image.tar
docker image inspect lmm-autodrive-basic:final >/dev/null
mkdir -p outputs
```

若权重没有放入镜像，将 `weights/` 只读挂载到 `/workspace/models`：

```bash
docker run --rm -it --gpus all --network host --shm-size=32g \
  -v "$PWD/weights:/workspace/models:ro" \
  -v "$PWD/outputs:/workspace/outputs" \
  lmm-autodrive-basic:final bash
```

若权重已经放入镜像，去掉第一条 `-v` 即可。以下命令均在容器 `/workspace/LMM-in-AutoDrive` 下执行，使用 Linux 相对路径。

## 4. 启动 CARLA

```bash
cd /workspace/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port=2000 > /workspace/outputs/carla_server.log 2>&1 &
cd /workspace/LMM-in-AutoDrive
python3 - <<'PY'
import carla
c = carla.Client("127.0.0.1", 2000)
c.set_timeout(30)
print(c.get_server_version())
PY
```

## 5. 三场景复现

统一 VLA 权重路径：

```text
../models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt
```

统一配置：

```text
lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json
```

命令解析模型：

```text
../models/modernbert-drive-command-compositional
```

场景一（Town04，5km，15 条文本指令）：

```bash
python3 -u experiment/CARLA/run_control_experiment.py basic_voice_urban_5km \
  --host 127.0.0.1 --port 2000 \
  --scenario-config experiment/CARLA/configs/basic_voice_urban_5km.json \
  --decision-source vla_scene_bridge \
  --command-parser-model ../models/modernbert-drive-command-compositional \
  --vla-checkpoint ../models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt \
  --vla-config lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json \
  --vla-device cuda --vla-precision fp16 \
  --target-speed-kmh 45 --duration-s 3600 --goal-distance-m 5000 \
  --stop-when-goal-reached --output-dir /workspace/outputs/scene1
```

场景二（Town05，8km，密集交通与四类特殊事件）：

```bash
python3 -u experiment/CARLA/run_complex_avoidance_town05.py \
  --host 127.0.0.1 --port 2000 --duration 0 --competition-run \
  --external-ego-control --output-dir /workspace/outputs/scene2 \
  --record-ground-truth --ground-truth-every-n 5 \
  --vla-checkpoint ../models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt \
  --vla-config lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json \
  --command-parser-model ../models/modernbert-drive-command-compositional \
  --vla-device cuda --vla-precision fp16 --vla-decision-every-n 3 --no-video
```

场景三（Town05，6km，7 类紧急事件）：

```bash
python3 -u experiment/CARLA/run_emergency_response_6km.py \
  --host 127.0.0.1 --port 2000 --duration 0 --require-complete-scene \
  --event-variant cautious_sparse --output-dir /workspace/outputs/scene3 \
  --camera-mode chase-only --presentation-lighting official-rainy-night \
  --record-ground-truth --ground-truth-every-n 5 \
  --ego-controller vla-route-pid \
  --vla-checkpoint ../models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt \
  --vla-config lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json \
  --vla-parser-model ../models/modernbert-drive-command-compositional \
  --vla-device cuda --vla-precision fp16 --vla-decision-every-n 3 --no-video
```

## 6. 输入输出接口

每次决策输入为统一 `UnifiedSensorBatch`：原始文本、最多四路同步 RGB、物理 LiDAR BEV（可选）、自车状态和天气/限速等环境状态。输出日志 `vla_control_decisions.jsonl` 记录模型提议、风险概率、监督器决策、是否实际施加以及端到端延迟；场景脚本同时输出 `runtime.jsonl`、`events.jsonl`、`commands.jsonl` 和最终 `summary.json`。

验收重点为路线完成、任务闭环、碰撞/红灯/逆行/驶离道路/禁行线等严重安全事件、指令一致性、30 分钟稳定性和端到端延迟。提交的实际自测结论以 `metrics.zip` 内三个场景各自的 `summary.json` 和 `TEST_REPORT.md` 为准，不以视频代替日志。
