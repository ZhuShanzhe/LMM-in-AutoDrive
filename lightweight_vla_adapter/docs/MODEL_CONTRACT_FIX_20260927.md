# 朱善哲：模型输出契约修复与核验

## 范围

基于五人问题分支 `a609334`，在个人分支 `fix/zsz-model-contract-20260927` 修复。此次只修改轻量VLA模型模块及其测试，不修改感知生产端、计划状态机、场景或独立评测。未训练、替换或重新发布权重。

## 已确认并修复的问题

| 问题 | 修复前同一故障注入 | 修复后 |
|---|---|---|
| 无有效历史但允许纵向推理 | 实际保留基础输出，却记录 `applied=true` | `authorized`、`available`、`applied` 分开记录 |
| 未应用记忆头仍发布序列 | 基础输出被保留，但仍有 `longitudinal_sequence` | 未实际应用时不发布序列；不泄露上次序列 |
| NaN风险输出 | `argmax`后得到 `low` | 抛出明确错误，由上层已有异常处理接管，不生成低风险结论 |
| 无效历史中的NaN | 乘零掩码不能消除NaN | 用布尔选择屏蔽；有效历史中的非有限值仍拒绝 |
| 模型输出来源混淆 | 日志中的network proposal可能已经过记忆/序列头 | 同时保留基础头与有效头输出，标明序列首样本动作来源 |
| 综合风险被误读成纯前方视觉风险 | 前后综合代理风险仍沿用visual字段名 | 补充风险scope、前后方向的时域分数及非校准语义，不改变保护动作 |

这些故障注入是接口鲁棒性测试，不代表真实日志里曾出现NaN，也不证明它们是全部驾驶失败的原因。

实际部署的Behavior/Layered子类也在Runtime入口屏蔽无效事件与行为历史，避免只修复基础SequenceEventHead而遗漏其子类。有效历史中的非有限数值仍视为输入错误。

## 接口

`pipeline.last_visual_risk_assessment`新增 `prediction_provenance`，现有风险字段保持不变。现有控制器会随风险日志保存，无需改其代码。

```json
{
  "prediction_provenance": {
    "schema_version": "vla_prediction_provenance/1.0",
    "base": {"action": "keep_lane", "target_speed_kmh": 30.0, "risk_probabilities": [0.9, 0.08, 0.02]},
    "effective": {"action": "accelerate", "target_speed_kmh": 31.0, "risk_probabilities": [0.1, 0.2, 0.7]},
    "residual_present": true,
    "residual_applied": true,
    "effective_action_source": "sequence_first_sample_encoding",
    "scope": "model stages before execution and safety gates; not ground-truth correctness"
  }
}
```

上面是字段示例，不是真实测试结果。`base`指当前发布模型的基础网络前向结果，不是基础赛道旧模型。`effective`指记忆/序列头处理后的结果。对于没有应用状态诊断的其他残差实现，`residual_applied`可以为null，不虚构已应用结论。

记忆诊断的 `authorized` 表示指令允许纵向头参与，`available` 表示当前记忆输入有效，`applied` 表示两者同时满足。只有实际应用才提供可执行序列。原有消费者已经使用可选序列字段，无需为此替换控制接口。

模型预测中的动作、速度、车道/对象指针、置信度及风险数值出现NaN/Inf时拒绝解码；负速度或越界置信度也拒绝。不会通过改成零值来掩盖无效模型输出。

记忆头实际应用时还提供 `risk_semantics`：`scope=joint_front_rear_rollout_surrogate`、`calibrated=false`、`horizon_seconds=[1,2,4]`、`directional_horizon_scores.front/rear`。原风险等级、推荐动作及理由码没有放宽，避免未经观测核验便减弱保护。

## 风险含义核验

检查现有 `counterfactual_motion_targets.rollout_targets`：风险标签来自前后方向恒速假设的碰撞时域及所需前向减速度，动作标签则选择候选加速度的综合代价最小值。两者不是“高风险必然要求刹车”的同一标签。

用现有生成器核验：自车10m/s、前方150米同速、后方15米以20m/s逼近时，生成 **high + accelerate**；前方15米静止、无后车时，生成 **high + emergency_brake**。这是合成训练目标语义检查，不是实测安全保证。

历史场景二示例frame 3686的记忆头前方4秒分数约0.0020、后方约0.2259，综合high概率约0.7506。各分数来自不同输出头且未经概率校准，不能直接互换或据此取消制动。该记录说明必须区分方向、核实观测，不能把全部high/accelerate组合都当作错误伪标签重训。

## 验证

- 增加26个测试实例；原工作区回归合计 **1363 passed，79 subtests passed**。9月28日仅导出待提交文件再次测试，**1355 passed，79 subtests passed**；另外8项早期实验用例未纳入本次提交。干净副本结果见 `model_contract_20260927/submission_clean_tests.log`。
- 已发布 `phase_recovery_v1.pt` 在CUDA上完成加载、授权、序列形状及运动学一致性检查。该项使用合成输入，不是驾驶准确率。
- 使用问题包中的8帧真实多视角RGB、LiDAR、车辆状态和实际ModernBERT编码，比较修复前后基础网络动作、目标速度和风险概率，结果完全一致。没有重建截取片段之前的事件/行为记忆，因此不把该项说成完整时序链路回放。
- 原始权重未变化；不宣称模型准确率或全程完成率已经提高。新增数值检查与来源记录的全链路时延尚未独立标定。
- 首轮核心补丁完成20秒场景二接入检查：400条有效决策、400条分阶段来源记录，0条模型契约异常，末条路线进度114.230米；首条速度任务未完成，退出3（证据不足），不是场景验收通过。
- 额外90秒窗口因仿真推进慢主动中止，实际记录451条决策、约22.5秒仿真时间，末条进度145.872米，未覆盖历史冲突段；退出-15，不使用它声称90秒测试完成。
- 随后追加的分层无效历史覆盖和风险语义字段通过最终1363项回归及实际权重检查，没有再为追加项重复闭环。

回归命令（仓库根目录、原有command_parser环境）：

```bash
PYTHONPATH=.:experiment/CARLA python -m pytest -q \
  lightweight_vla_adapter/tests scene_understanding/tests experiment/CARLA/tests
python lightweight_vla_adapter/scripts/smoke_sequence_runtime.py \
  --checkpoint /root/autodl-tmp/models/challenge-assets-pinned/challenge/phase_recovery_v1.pt \
  --device cuda
```

新增只需Python标准库的日志核验入口：

```bash
python lightweight_vla_adapter/scripts/audit_model_output_log.py \
  /path/to/vla_control_decisions.jsonl --output /path/to/model_audit.json
```

该工具检查分阶段来源、实际应用标志和序列授权，不把风险分数当真值；错误记录单独计数。它的通过不能替代任务成功或驾驶安全评测。

## 当前证据边界

场景二79帧、场景三19帧的高风险/加速冲突来自先前日志。此次没有把这些帧自动当成真实危险训练标签，也没有改写历史日志。新记录能够在后续同源复测中区分基础网络、时序头与执行层。

风险输入真实性依赖黄皓星的H1/H2观测核验；45km/h目标与步骤推进依赖王皓然的W1/W2修复；场景几何及独立评测边界由刘旭处理。此处不代改这些模块，也不将其问题标记为已解决。文本配置模式与真实解析模式的区分仍按李畅锦的L1任务核验。

完整本轮日志保存在服务器 `/root/autodl-tmp/review_logs/20260927/zsz_model_fix/`。个人修复尚未合并到公共分支。

可移交的精简证据位于同目录 `model_contract_20260927/`，包含故障注入前后结果、基础网络真实输入一致性、闭环模型契约检查及最终回归日志。
