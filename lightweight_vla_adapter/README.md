# 挑战赛道分层 VLA 运行模块

当前固定版本：`signal-generalization-v2-phase-recovery-v1`。本模块用于 x86/CARLA 挑战赛道开发基准，不是已完成 J6P 量化部署的模型。旧 V6 配置仅供架构对照，新版必须使用本页的配置和权重组合。

## 架构与职责

1. **观测**：四视角 RGB、物理 LiDAR/雷达、自车状态及道路信息按时间戳组织；持续目标观测与有效性分开保存。灯头由静态地图几何定位，当前 RGB 由 56,148 参数的 LampStateNet 判别 UNKNOWN/RED/YELLOW/GREEN。
2. **记忆**：短窗口与事件历史结合；行为转换时压缩和筛选之前的状态，区分需要持续保留的跟车事件与已经结束的加速行为。目标离开视野后历史可保留 30 秒，但不是当前可直接执行的可靠观测。
3. **决策**：ModernBERT token、图像/点云状态和记忆输入四层 Cross-Attention VLA；序列头输出未来 3 秒、30 步速度与加速度。
4. **约束与执行**：意图图、目标车道授权、停止线与风险约束共同限制最终行为，FSM/路线 PID 执行。常规操作间隔 0.5 秒，危险情况下可立即制动。

真实视觉来自多视角图像编码器；当前候选实体通道关闭、camera BEV 兼容张量不代表已学习的完整几何 BEV。不要把本版本称作已经实现 BEVFusion 的完整传感器融合网络。灯色不在线读取 CARLA 动态信号真值，静态地图会核对 OpenDRIVE SHA256；换图或道路资源变更必须匹配静态几何。

## 与上一版的不同

- 从旧序列候选推进为递归事件/行为记忆与观测分层，使用跟车恢复训练后的 `phase_recovery_v1` 权重。
- 打通近期速度/加速度参考与低层执行，保留长期目标速度，新增序列有效期和 0.5 秒常规操作间隔。
- 保留组合指令各步骤、前置条件和真实完成反馈；危险换道不能通过等待超时自动放行。
- 新增跨灯头、天气、距离采集训练的 RGB 灯色模型，结合曝光时相机姿态、适用车道与停止线约束，替代单路口颜色阈值依赖。
- 初始化增加神经网络预热，同次决策复用方向风险结果；暂未进行全链路剪枝或量化。

## 环境

实际验证：Linux x86、Python 3.12.13、CARLA 0.9.16、RTX 5090、PyTorch 2.11.0+cu130、Transformers 4.57.6、NumPy 1.26.4。该 Python 环境是仿真运行环境，不是地平线工具链环境。

```bash
conda create -n challenge_runtime python=3.12.13 -y
conda activate challenge_runtime
python -m pip install -r lightweight_vla_adapter/requirements-challenge.txt
export PYTHONPATH="$PWD/experiment/CARLA:$PWD"
python -c "import torch, carla; print(torch.__version__, torch.cuda.is_available())"
```

依赖文件固定已验证的关键库；CARLA 服务端需另行安装 0.9.16。服务端使用非 root 用户、离屏渲染，客户端可以使用当前有权限的用户。不能因为驱动显示 CUDA 13.x 就把 J6P 编译器装进同一个环境。

## 固定模型

权重不提交 Git，位置与获取方式见 [模型目录](../models/README.md)。`configs/challenge_assets.json` 固定 23 个运行资产及说明文件的大小和 SHA256，配置生成前必须全部校验通过。

```text
models/
  modernbert-drive-command-compositional/       # 解析权重、分词器、标签/推理配置与许可
  lightweight_vla_adapter/
    universal_three_scene_v6_sensor_policy/model.pt
  challenge/
    phase_recovery_v1.pt                        # 微调 VLA + 递归记忆/序列头，约 23 MB
    lamp_state_v2.pt                            # 当前 RGB 灯色模型，约 224 KB
```

新版配置提供灯色权重时不会加载旧 YOLO 灯检测器，因此不需要为此模式另下载 `models/pretrained/yolo11s.pt`。独立场景理解模块使用的检测模型仍按该模块说明配置。

```bash
python lightweight_vla_adapter/scripts/prepare_challenge_runtime.py \
  --map Town04 --model-root "$PWD/models" \
  --output lightweight_vla_adapter/outputs/runtime/Town04.json
```

该命令校验权重后生成使用绝对运行路径的本机配置；Git 内模板没有个人实验目录依赖。Town01/Town05 使用对应 `--map`。其他地图先导出几何，再用 `--static-map`；导出工具只记录静态路口/灯头几何，不导出动态颜色。

```bash
python lightweight_vla_adapter/scripts/export_static_traffic_controls.py \
  --port 2000 --output lightweight_vla_adapter/outputs/runtime/custom_map.json
python lightweight_vla_adapter/scripts/prepare_challenge_runtime.py \
  --static-map lightweight_vla_adapter/outputs/runtime/custom_map.json \
  --output lightweight_vla_adapter/outputs/runtime/custom_config.json
```

`Town04_Opt` 和 `Town04` 不应仅凭名称相近视为同一地图，运行时以哈希为准。缺失或不匹配时不允许静默使用别的路口几何。

## 调用接口

入口仍为 `experiment/CARLA/universal_vla_controller.py` 的 `UniversalVLAController`。上游提供世界、自车、路线控制器、指令计划；在线摄像头、雷达和 LiDAR 由控制器管理。集成时沿用路线控制器协议，不把以下变量替换成假观测。

```python
from pathlib import Path
from universal_vla_controller import UniversalVLAController

runtime = UniversalVLAController(
    world=world,
    ego=ego,
    route_controller=route_controller,
    commands=[{"text": "Keep the current lane at 20 kilometers per hour.",
               "trigger_progress_m": 0.0}],
    checkpoint_path=Path("models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt"),
    config_path=Path("lightweight_vla_adapter/outputs/runtime/Town04.json"),
    parser_model_path=Path("models/modernbert-drive-command-compositional"),
    output_path=Path("lightweight_vla_adapter/outputs/runtime/decisions.jsonl"),
    precision="fp32",
    decision_interval_frames=2,
    fixed_delta_seconds=0.05,
    enable_lidar=True,
    default_speed_kmh=20.0,
    hold_seconds=2.0,
)
# world 使用同步模式、fixed_delta_seconds=0.05；仅由一个主循环推进。
# 先等待传感器缓冲就绪，随后每个物理帧调用：
control = runtime.run_step()
ego.apply_control(control)
world.tick()
# 场景结束或异常退出时放在 finally 中：
runtime.close()
```

未显式设置 `available_cameras` 时使用前、左、右、后四视角；本基准必须开启 LiDAR。正式主循环应复用现有场景管理器的资源回收，不在不同线程重复推进世界。日志父目录需提前建立。

**输出**：`run_step()` 返回 CARLA `VehicleControl`。日志保存原始 VLA 建议、语义/风险约束和最终决策。纵向序列位于 `risk_assessment.event_memory.longitudinal_sequence`，含 `schema_version=longitudinal_sequence/1.0`、`dt_s=0.1`、30 项 `speed_mps`/`acceleration_mps2`。授权序列的控制指令附带 `target_acceleration_mps2`、`sequence_valid_until_s`；过期、改写或拒绝时不得强行透传正加速度。

运行时的近期 `target_speed_kmh` 不等于原指令巡航速度，也不等于整个 3 秒序列末端速度。当前配置利用前 10 步的终点估计近期加速度，并滚动形成参考速度；下游不能重新用长期巡航速度覆盖它。

旧 `scripts/run_universal_vla.sh` 是原三场景对照入口，不能仅更换权重便宣称已接入本基准。新配置通过上述控制器接口接入各组员场景；本轮完整 5/8/6 km 三场景未重新执行。

## 验证结果

训练共 2,960 张仿真灯图：训练 1,600、验证 1,024、保留路口测试 336。第 37 轮按验证集选择，验证/测试四分类均 100%，红绿召回均 100%。保留路口未参与训练，但曾用于失败诊断，不称为从未查看的盲测。自动仿真标签不等于人工金标准。

14 次完成的 22 秒物理闭环全部通过：10 次红转绿、1 次绿黄红绿、1 次跟车恢复、1 次安全左换道、1 次危险左换道拒绝。完成用例的碰撞、闯红及危险/未授权换道均为 0。常规信号覆盖 Town01/Town04 共 9 个不同信号，4 种天气；另两次 Town05 引擎中断不计通过。

标准信号用例调用 P95 为 77.69–103.10 ms；安全/阻塞左换道为 130.41/134.63 ms。首次调用为 410.48–555.96 ms，不含 ASR/构造初始化/后续物理执行。有限闭环结果不能代替全路线完成率、长期可靠性或 J6P 性能。

旧 RGB 模型曾出现绿灯漏检、距离/天气变化误差和取图姿态不同步；本轮改为曝光时投影、发光成对自动标签与跨路口分组。强增亮扰动仍降至 91.96%、变暗为 97.32%，未出现非绿判绿。完整尝试与对比见 [信号泛化记录](CHALLENGE_SIGNAL_GENERALIZATION.md)，既有 V6 来源见 [模型说明](UNIVERSAL_THREE_SCENE_MODEL.md)。

接口与回归入口见 [联合验证](../CHALLENGE_DEVELOPMENT.md)。无权重测试只证明代码契约；部署设备、量化精度和闭环效果须分别记录，不能混用结果。

本次整理后联合回归 833 项、207 个子测试通过；固定权重 CUDA 序列检查通过，23 个运行资产校验通过。可单独执行新序列检查：

```bash
python lightweight_vla_adapter/scripts/smoke_sequence_runtime.py \
  --checkpoint models/challenge/phase_recovery_v1.pt --device cuda
```

此命令使用合成张量验证加载与接口，不输出驾驶准确率。
