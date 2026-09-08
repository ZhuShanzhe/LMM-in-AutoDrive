# 场景三文本到 VLA 闭环测试报告（2026-08-05）

## 1. 结论

场景三已经接通以下在线闭环，并完成一次严格 6 km 实跑：

`中文文本指令 -> ModernBERT -> VLA adapter -> 风险/活性门控 -> route-PID -> CARLA VehicleControl`

- 路线完成 `6000.969 m`，`route_completed=true`。
- 7/7 个事件全部进入 `RESOLVED`，碰撞 0，非法车道采样 0。
- `model_output_used=true`，VLA fallback 0。
- 1,463 次在线决策中，597 次模型动作可采用，采用率 `40.8066%`。
- 全决策延迟平均 `6.037 ms`，P95 `7.412 ms`，最大 `19.005 ms`，低于场景 120 ms 预算。
- 完整 H.264 视频 29,288 帧，0 丢帧，1280x720，20 FPS，时长 1464.4 秒。
- CARLA/VLA 单元测试 `95/95` 通过。

本轮只测试文本指令，不生成、不播放、也不解析声音。配置和部分事件字段仍沿用历史 `voice_*` 命名，但 VLA 控制器读取的是配置中的 `text` 字段；没有音频或 ASR 数据进入推理。

当前闭环不能描述为“原始传感器端到端 VLA”。VLA 的视觉输入是从实时 CARLA actor/车道状态生成的结构化 BEV 代理，日志标记为 `carla_state_structured_bev_proxy`，不是由 RGB、LiDAR 或摄像头感知模型估计出的 BEV。该方式适合验证文本、VLA adapter、门控和车辆控制接口，但会高估真实感知条件下的输入质量。

## 2. 本轮修改

### 2.1 文本/VLA/控制接口

- 新增 `scene3_vla_controller.py`，在同一进程加载 ModernBERT 和 VLA 权重。
- 根据实时 CARLA 状态构建结构化 BEV，调用真实 VLA adapter 权重产生动作、速度、置信度和目标实体。
- 将安全审查后的高层动作传给场景三 route-PID，最终调用 CARLA `VehicleControl`。
- 每次决策同时记录模型原始输出、风险判断、最终控制意图、覆盖原因和全链路延迟。
- `FrameGroundTruth.model_output_used` 在该模式下记录为 `true`；严格完成条件要求决策数大于 0 且 fallback 为 0。

### 2.2 相邻车道误制动修复

第一次长测在约 268 m 停滞。VLA 日志为 `candidate_count=0`、`risk_level=low`，但 route-PID 的障碍兜底仅以横向距离小于 3 m 判定，会把相邻车道车辆当成本车道障碍。

修复后：

- 优先比较 CARLA `road_id/section_id/lane_id`；
- 只在路口、道路分段边界或 waypoint 不明确时使用 1.65 m 的几何兜底；
- 保留同车道车辆和行人的紧急制动；
- 新增同车道、相邻车道和道路边界三类单元测试。

400 m 回归实际到达 `695.066 m`，碰撞 0、非法车道采样 0、fallback 0，确认跨过旧死锁点。最终 6 km 也成功跨过此前发生锥桶误判的约 3446 m 位置。

### 2.3 事件文本生命周期

原调度器会永久保持最近一次触发的文本。例如横穿人员在约 3425 m 清空后，“有人突然横穿”仍会一直生效到 4700 m，使车辆长期限速 10 km/h，既污染评估，也可能导致 6 km 超时。

现在每条事件型文本具有 `end_progress_m`：

- 只在对应事件窗口内参与决策；
- 重叠事件结束后回退到仍有效的上一条文本；
- 没有有效事件文本时回到 `scene3_cruise`；
- 1550 m 和 3450 m 后的真实闭环日志均确认已恢复巡航文本。

## 3. 场景覆盖与事件结果

| 事件 | 激活进度 | 解除进度 | 结果 |
|---|---:|---:|---|
| 突发车辆切入 | 1250.448 m | 1551.386 m | 合流完成，无碰撞 |
| 施工提前预警 | 2851.071 m | 3051.426 m | 顺序正确 |
| 锥桶渐变收窄 | 2976.148 m | 3150.043 m | 未再产生邻道锥桶死锁 |
| 右侧施工区 | 3100.043 m | 3451.994 m | 正常通过 |
| 临时人员横穿 | 3251.261 m | 3425.612 m | 人员清空后恢复 |
| 维护车辆阻塞 | 4701.847 m | 5001.306 m | 等待安全间隙后左换道 |
| 驶离施工区 | 5001.306 m | 5257.577 m | 恢复正常速度 |

现有 7 类连续小事件已经覆盖切入、预警、锥桶渐变、施工车辆、人员横穿、阻塞换道和恢复行驶。当前不需要仅为增加数量再添加第 8 类事件。后续更有效的是对现有事件做多 seed 参数矩阵，例如 TTC、目标距离、左右位置、车流密度、雨雾强度和文本同义改写。

## 4. VLA 输出统计

| 指标 | 结果 |
|---|---:|
| 在线决策 | 1,463 |
| 模型动作可采用 | 597（40.8066%） |
| safety/canonical 修正 | 866 |
| fallback | 0 |
| 原始低风险误停 | 550 |
| 原始速度低于场景下限 | 583 |
| 与当前文本意图不一致 | 315 |
| 风险层紧急覆盖 | 1 |
| 平均/P95/最大延迟 | 6.037/7.412/19.005 ms |

原始 VLA 动作分布：

| 动作 | 次数 |
|---|---:|
| keep_lane | 563 |
| stop | 551 |
| emergency_brake | 139 |
| accelerate | 125 |
| lane_change_left | 63 |
| decelerate | 22 |

最终高层动作分布：`keep_lane=863`、`decelerate=313`、`accelerate=212`、`lane_change_left=73`、`emergency_brake=1`、`stop=1`。

分指令采用情况：

| 指令 | 决策数 | 模型可采用 | 主要问题 |
|---|---:|---:|---|
| 巡航 | 860 | 440 | 416 次 stop；速度头普遍过低 |
| 通用危险减速 | 123 | 0 | keep/stop 与减速意图不一致 |
| 切入避让 | 68 | 0 | 主要输出 keep_lane |
| 施工预警 | 22 | 0 | 只输出 keep_lane |
| 并道至左侧 | 22 | 13 | 9 次 emergency_brake |
| 通过施工区 | 27 | 3 | 24 次 emergency_brake |
| 人员横穿 | 64 | 0 | stop/emergency 方向合理，但未按风险解除 |
| 阻塞左换道 | 65 | 16 | 36 次 emergency_brake；29 次左换道 |
| 恢复行驶 | 212 | 125 | 仍有 stop/emergency/错误换道 |

## 5. 模型不足与后续修改优先级

1. **速度头失准。** 583 次动作需要提升到文本/场景速度下限，部分巡航预测只有约 1--5 km/h。优先使用本次真实闭环 BEV 分布重训速度回归头，并报告 MAE、分位误差和危险/非危险分桶结果。
2. **低风险 stop 误报严重。** 550 次低风险误停会在没有活性门控时造成大面积死锁。训练集需要增加无障碍巡航负样本，并对 `stop` 做时序确认和概率校准。
3. **文本与视觉融合不稳定。** 离线“左换道”样本可正确输出，但实时 BEV 下阻塞场景经常变为 `emergency_brake`。应检查训练/推理 BEV 通道含义、朝向、尺度、归一化和实体类别是否一致。
4. **事件解除缺少模型记忆。** 人员和障碍清空后仍会连续预测 stop/emergency。可加入短时状态、目标跟踪和事件解除样本，但不能用永久锁存危险指令代替时序建模。
5. **当前安全层占比较高。** 40.81% 的采用率说明接口已闭环，但不能据此宣称 VLA 已独立完成 6 km；改模型后应以模型采用率、门控原因和闭环完成率共同验收。
6. **感知仍是特权代理。** 下一阶段应把真实相机/LiDAR感知结果转换为同一 BEV schema，再与当前 actor-state proxy 做 A/B 对照。原始传感器链路未接通前，不报告端到端感知准确率。

## 6. 安全审计边界

- 碰撞：0。
- 非法车道采样：0。
- CARLA 原始 lane-invasion 回调：209。

209 次回调不能直接解释成 209 次违章：只有 5 次位于受控换道窗口，146 次发生时 `abs(steer)<=0.1`，且触发分布于 Town05 路口和重复路线连接处；视频抽样中车辆大多保持在合法车道内。不过该计数仍偏高，后续应增加车道线 ID、路口状态、持续时间聚类和允许换道窗口过滤，形成可评分的压线指标，并对 route-PID 横向误差做专项统计。

风险层在 1,463 次决策中仅产生 `high=1`、`medium=1`，其余为 low。这说明当前规则风险层偏稀疏，安全成功不能全部归因于 VLA 风险识别；应利用 TTC、横穿轨迹和同车道占用做连续风险标签校准。

## 7. 视频与结果文件

推荐查看压缩版：

```text
experiment/CARLA/outputs/scene3_vla_full_20260805_r7/scene3_vla_6km_compact.mp4
```

- H.264，1280x720，20 FPS，29,288 帧，1464.4 秒；
- 592,706,840 bytes；
- SHA-256：`53fa518aceb0ba1813e342f4d4ce9f22de071a93bb2a4ea122a6a7b7f1ee6823`。

原始高码率版：

```text
experiment/CARLA/outputs/scene3_vla_full_20260805_r7/scene3_vla_6km.mp4
```

- 2,633,109,814 bytes；
- SHA-256：`4845983fae7ce1850b43ee57619a8f9e5f6c4fd9aef6b65cb4f281798af45aa1`。

同目录还包含 `scene_summary.json`、`event_timeline.jsonl`、`frame_ground_truth.jsonl`、`vehicle_state.jsonl`、`vla_control_decisions.jsonl`、`vla_control_summary.json`、配置快照和抽帧联系图。

抽取 12 个时间点检查，建筑、湿路、车道线、车辆和 HUD 均可辨，没有旧配置中高光整片裁白的过曝。整体画面仍偏亮，属于 Town05 夜间环境照明与湿路反射观感；如需更真实的夜景，应单独标定太阳高度、路灯、后处理曝光和显示 gamma，不应再次用高正曝光补偿。

## 8. Linux/Docker 相对路径复现

从仓库根目录运行；Docker 镜像负责提供 CARLA Python API、PyTorch、Transformers、FFmpeg 和模型运行环境，权重通过相对路径或环境变量挂载：

```bash
export VLA_CHECKPOINT="${VLA_CHECKPOINT:-../models/lightweight_vla_adapter/v10/model.pt}"
export MODERNBERT_MODEL="${MODERNBERT_MODEL:-../models/modernbert-drive-command-compositional}"
export OUTPUT_DIR="${OUTPUT_DIR:-experiment/CARLA/outputs/scene3_vla_reproduction}"

python experiment/CARLA/run_emergency_response_6km.py \
  --ego-controller vla-route-pid \
  --vla-checkpoint "$VLA_CHECKPOINT" \
  --vla-parser-model "$MODERNBERT_MODEL" \
  --vla-decision-every-n 20 \
  --duration 1800 \
  --camera-mode chase-only \
  --record-ground-truth \
  --ground-truth-every-n 20 \
  --video-output "$OUTPUT_DIR/scene3_vla_6km.mp4" \
  --video-overlay \
  --video-fps 20 \
  --camera-width 1280 \
  --camera-height 720 \
  --output-dir "$OUTPUT_DIR" \
  --require-complete-scene

python experiment/CARLA/tools/summarize_scene3_vla_control.py \
  "$OUTPUT_DIR/vla_control_decisions.jsonl" \
  --output "$OUTPUT_DIR/vla_control_summary.json"
```

输出目录和权重路径不写死服务器绝对路径，适合随 Docker 镜像与模型权重一起交付。
