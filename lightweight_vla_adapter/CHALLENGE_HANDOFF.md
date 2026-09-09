# 挑战赛道序列策略联调说明

## 分支与状态

本次向 `challenge-track` 交接事件记忆与短期序列策略的运行代码、配置、接口测试和已测结果，供各模块并行适配。原默认配置与公共权重保留，`main` 不变。新增训练程序、训练数据、中间检查点及原始大规模日志不进入此次提交，仍保留在服务器 `zsz` 工作区。

新模型是实验候选，不是已验收模型。当前仅验证 Linux x86、Python 3.12、PyTorch 2.11.0+cu130、CARLA 0.9.16、RTX 5090 上的 FP32。新增记忆/序列分支尚未完成 J6P 编译、量化和板端验证。原模块 README 中的旧性能数据不能替代本模型结果。

## 已实现架构

```text
指令解析与 ModernBERT token + 四视角 RGB + LiDAR + 车辆/环境状态
                 + 前后物理雷达观测及有效性
                              |
                 原 VLA 融合 + 事件驱动历史
                      0.4 / 2 / 8 秒
                              |
                 未来 3 秒、30 步加速度
                              |
                 0.1 秒积分得到速度序列
                              |
                 原风险门、语义约束和 FSM
                              |
               最新首步速度/加速度 -> 低层控制
```

仅 `KEEP_LANE`、`SET_SPEED`、`ADJUST_SPEED` 且没有换道方向时启用序列分支。换道、转向和其他指令保留原模型与控制路径。当前候选实体输入关闭、camera BEV 为兼容占位，不能称为已训练完成的实体条件化或完整视觉 BEV 序列模型；真实图像来自四视角图像编码器，LiDAR 使用既有物理点云编码。

直接网络试验跳过高层风险解码/监督/FSM 改写，仍有低层控制器，仅限隔离仿真诊断。公共运行入口不新增绕过安全门的默认开关。

## 输入与输出约定

- 上游仍输出 `DrivingIntent`、指令目标速度、ModernBERT token，以及既有同步传感器/车辆状态。ASR 和中文翻译接口不因本次改变。
- 历史由 `EventMemoryBuffer` 维护，使用真实时间戳、帧号、有效掩码和 episode 标识；不要把未来帧或仿真风险标签作为输入。时序中断、回退或 episode 变化会清空历史。
- 原始序列位于运行日志 `risk_assessment.event_memory.longitudinal_sequence`：`schema_version=longitudinal_sequence/1.0`，`dt_s=0.1`，`horizon_s=3.0`，`speed_mps` 与 `acceleration_mps2` 各 30 项。对应观测帧为同级风险记录的 `sensor_frame_id`，决策帧为 `frame_id`。
- `target_speed_kmh` 在序列执行指令中表示首步速度乘 3.6，不是长期巡航目标，也不是 3 秒末端速度。长期目标仍来自上游指令。
- 只有明确接受且动作、速度没有被改写的已授权序列，执行指令才附加 `longitudinal_sequence_schema` 和 `target_acceleration_mps2`。其余情况保持原控制方式。
- 低层 `EgoPIDController` 仅在 schema 匹配时启用加速度前馈与速度/加速度反馈。停车、紧急制动、下游速度上限优先；禁止通过补回正加速度绕过上游否决。原单步指令不变。

高层决策适配的边界是：区分长期目标与近期采样点，不因正常起步的首步速度低就套用旧速度下限。控制模块的边界是：按最新有效观测滚动执行首步，不开环执行整段；显式处理过期序列和动作模式切换。感知/对齐模块保持时间戳、目标连续性与缺失状态一致。现有接口尚不构成通用带有效期的序列执行协议。

## 配置与权重

在仓库根目录运行。沿用团队现有 `command_parser` 环境；新分支没有增加训练环境要求。无 GPU、无 CARLA、无权重也可用安装了 PyTorch、Transformers、NumPy、pytest 的环境运行接口测试。

```bash
python lightweight_vla_adapter/scripts/smoke_sequence_runtime.py
python -m pytest lightweight_vla_adapter/tests/test_sequence_runtime_contract.py -q
```

上述使用随机权重和合成张量，只证明接口、维度、授权隔离及积分一致性，不代表驾驶性能。

候选配置：`lightweight_vla_adapter/configs/challenge_sequence_v2.json`。约定权重路径为 `models/challenge/sequence_policy_v2.pt`，约 21 MB，SHA256：

```text
0b7cb441ad2c0050da041a4a6426432d6899fb9788efc7903f1fa3d940f6e5b6
```

服务器已放置于 `/root/autodl-tmp/worktrees/challenge-track/models/challenge/sequence_policy_v2.pt`。Git 不包含该文件，本轮没有上传 Hugging Face；外部机器需通过负责人分发或使用自己获准的 SSH 访问下载，不在仓库保存登录凭据。

```bash
mkdir -p models/challenge
# PORT、USER、HOST 使用本人获准的服务器访问配置。
scp -P PORT USER@HOST:/root/autodl-tmp/worktrees/challenge-track/models/challenge/sequence_policy_v2.pt models/challenge/
sha256sum models/challenge/sequence_policy_v2.pt
python lightweight_vla_adapter/scripts/smoke_sequence_runtime.py \
  --checkpoint models/challenge/sequence_policy_v2.pt --device cuda
```

候选文件包含微调后的 VLA 与序列头，不需要下载训练数据。完整 CARLA 入口仍需要旧 VLA 基线权重和 ModernBERT，供非纵向指令及原链路使用；下载方法沿用本模块及指令解析模块 README。初始化 `UniversalVLAController` 时使用新 `config_path`，`checkpoint_path` 仍指向旧基线，`precision="fp32"`，提供四视角与 LiDAR，不要直接套用仅前视角的旧场景一配置。启动位置必须是仓库根目录，使配置中的相对权重路径正确解析。

原配置 `universal_three_scene_v6_sensor_policy.json` 不带 `event_memory_checkpoint`，继续运行原链路。普通 `run_offline_inference.py` 不负责维护序列历史；新分支验证使用 smoke 入口或 `UniversalVLAController`，不能仅改该离线脚本的配置便宣称启用时序策略。

## 已有结果与限制

两轮训练解冻四层融合网络，65 个 VLA 参数张量发生变化。新 Town05 固定权重测试 1,200 帧：3 秒速度序列 MAE 3.72 km/h、加速度 MAE 0.705 m/s²、风险准确率 85.58%、高风险召回率 97.60%。监督来自简化动力学反事实弱标签，不是专家真实未来轨迹。

最终直接网络与完整安全链路各完成六类配对仿真，共 24 次运行无碰撞、无运行回退。这里只证明这些已测用例，不是所有场景安全保证。前车制动、静止前车、重新起步等用例是瞬态条件，不替代持续稳态跟车测试。

| 巡航车速稳定性，固定最后 5 秒 | 参考速度 | ±3 km/h 帧占比 | 结果 |
|---|---:|---:|---|
| 直接网络 | 30 km/h | 54% | 未通过 |
| 完整链路 | 30 km/h | 100% | 通过 |

用户确认的工程标准为参考速度 ±3 km/h 内至少 95% 帧满足。加减速指令切换次数仅诊断，不单独判失败。车速稳定不豁免碰撞、不安全车距或停驻不跟车。制动、停车、重新起步不按恒速标准计分。

完整链路的 900 次候选决策中，加速度前馈透传为 0 次；旧解码/监督层会改写近期目标，不能将完整链路成绩当成序列前馈已打通。直接网络的车速稳定性、旧监督层与近期目标的适配、持续跟车覆盖、风险标签可信度、跨重规划一致性、序列有效期处理、跨车型控制标定和 J6P 部署，均仍是明确存在的限制。

本次交接工作区回归与已知旧失败记录见下方验证记录。原始训练及仿真数据保留在服务器 `zsz/lightweight_vla_adapter/outputs/sequence_policy_v2/`，不随此次代码提交分发。

## 本次验证

独立 `challenge-track` 工作区已通过无权重 CPU 和候选权重 CUDA 的运行契约测试。训练标签生成测试与训练脚本留在个人工作区，交接测试不依赖这些文件。VLA 与 CARLA 范围回归为 337 项通过、38 个子测试通过，另有一项既有失败：`experiment/CARLA/tests/test_continuous_manager.py:57` 期待路线进度 10 米、现有实现返回 11 米，本次未修改或隐藏该失败。该问题与序列控制无关，但不能据此声称整个仓库全绿。
