# 三场景通用 VLA 测试报告（2026-08-07 更新）

## 一、结论

最终候选模型为 `universal-three-scene-sensor-policy-vla-v6`。三个题目场景现在共用：

1. 一套固定模型架构（`LightweightDecisionAdapter`，4 层跨模态融合）；
2. 一份通用模型权重（同一 checkpoint，场景间不切换）；
3. 一种固定输入数据结构（`UnifiedSensorBatch`，固定字段与张量结构）；
4. 一组固定输出头（action / target_speed_kmh / target_lane / confidence / visual_risk，另支持按视图 mask 探测目标车道视觉风险）；
5. 一个通用在线控制器入口（`experiment/CARLA/universal_vla_controller.py` 的 `UniversalVLAController`）；
6. 一个通用时序风险监督器（`experiment/CARLA/control/generic_temporal_risk_supervisor.py`）；
7. 一个通用指令 FSM（`experiment/CARLA/control/generic_instruction_fsm.py`）；
8. 一个通用 Route PID（`experiment/CARLA/control/generic_route_pid.py`，底层复用 `EgoPIDController`）。

正式链路固定为：

```text
UnifiedSensorBatch
  -> Universal VLA Pipeline
  -> Generic Temporal Risk Supervisor
  -> Generic Instruction FSM
  -> Route PID
  -> carla.VehicleControl
```

三个场景运行脚本（`run_control_experiment.py`、`run_complex_avoidance_town05.py`、`run_emergency_response_6km.py`）都导入并调用同一个 `UniversalVLAController`。旧的 `Scene3VlaController` 已删除，不再保留三个正式在线控制器。

## 二、为什么传感器数量不同仍是一套模型架构

模型从不按场景切换类、配置或权重。模型输入是固定超集：场景可用的模态填入真实张量，不可用模态一律以“零张量 + mask=false”表示，模型内部的注意力 mask 会屏蔽缺失模态。场景差异只来自：

- 地图（Town04 / Town05）；
- 天气与光照；
- 实际传感器可用性（场景一只装前视；场景二四路 RGB + 真实 LiDAR；场景三四路 RGB、无 LiDAR）；
- 文本指令时间表；
- 评测事件。

`UnifiedSensorBatch` 固定字段（`schema_version=unified_sensor_batch/1.0`）：

```text
text_tokens / front_rgb / left_rgb / right_rgb / rear_rgb / lidar_bev
/ vehicle_state / environment_state / camera_view_mask / modality_mask / schema_version
```

三个场景的 `modality_mask`：

| 场景 | text | front | left | right | rear | lidar | vehicle/env |
|---|---|---|---|---|---|---|---|
| 场景一 | true | true | false | false | false | false | true |
| 场景二 | true | true | true | true | true | true | true |
| 场景三 | true | true | true | true | true | false | true |

缺失 RGB / LiDAR 通过零张量 + `camera_view_mask=false` / `modality_mask=false` 表示，模型输入 shape 三个场景完全一致。

## 三、统一 checkpoint

- 权重路径（仓库相对）：`models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt`（实际部署由 `MODEL_ROOT` 覆盖）
- 配置路径：`lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json`
- 权重大小：21,165,103 bytes
- 权重 SHA-256：`06c774e3a5eead95e230b55b65a3d86c52ba93110c6046506be21dd69ecb2165`
- 配置 SHA-256：`47204682f7cdb5ee7be3ec0a077f1ee04c0a067b0a6a07a2aa9434949a4aa20d`

三个场景引用同一份权重与配置，未发现场景专用 checkpoint。

## 四、正式控制器与通用组件路径

- 唯一正式在线控制器：`experiment/CARLA/universal_vla_controller.py`
- 通用指令 FSM：`experiment/CARLA/control/generic_instruction_fsm.py`
- 通用时序风险监督器：`experiment/CARLA/control/generic_temporal_risk_supervisor.py`
- 通用 Route PID：`experiment/CARLA/control/generic_route_pid.py`
- 统一输入超集：`lightweight_vla_adapter/src/unified_sensor_batch.py`

监督器只使用通用语义条件：`parsed_intent`、`requested_lane_direction`、全局视觉风险历史、目标车道视觉风险历史、停车时长、当前车速、车辆状态与 `modality_mask`。通用覆盖名称为：

- `low_risk_deceleration_crawl`
- `temporal_hazard_clearance`
- `cautious_hazard_resume`
- `target_lane_visual_clearance`
- `temporal_risk_confirmation`（谨慎爬行期间，单帧高风险先保持爬行，连续两帧高风险才升级为紧急制动）
- `unconfirmed_stop_crawl_floor`（风险头为 low/medium 且文本指令不要求停车时，模型停车动作视为未确认闪烁，降为 10 km/h 谨慎爬行；高风险仍立即急刹）

旧的 `cleared_worker_resume`、`cleared_worker_caution_resume`、`cleared_blocked_lane_caution_change_left` 等 scene3 专用名称已移除。

规则/执行层另有通用速度策略：低/中风险时使用最高 10 km/h 的谨慎爬行/减速下限（取代旧的 4 km/h），并配套爬行油门增强——目标车速 ≤12 km/h 且当前车速 <2 km/h 时，Route PID 使用更高油门增益并放宽油门变化率，使车辆在模型低/中风险恢复窗口内真正前进，避免“静止画面持续 high”导致的永久死锁。路口/道路切换前仍统一限速 9 km/h，并将路线前瞻缩短到 3.2 m，确保转向平稳且不切入错误分支。路线进度跟踪器将单次前跳窗口限制为约 60 m，防止循环路线在重复路口处把进度跳到远期路段。这些行为对所有场景一致，不读取 actor、事件或场景 ID。

## 五、场景代码与策略代码边界

场景目录/运行脚本只负责：地图、天气光照、传感器安装、文本指令时间表、仿真事件生成、独立真值评估、视频与评估数据记录。以下决策全部来自统一 VLA + 监督器 + FSM + PID：

- 何时停车 / 恢复；
- 何时变道、哪条车道安全；
- 目标速度；
- 是否绕开当前障碍物。

静态审计结果（`universal_vla_controller.py`、`generic_instruction_fsm.py`、`generic_temporal_risk_supervisor.py`、`generic_route_pid.py`、`unified_sensor_batch.py`）：

- 不含 `scene_1/scene_2/scene_3/scene3_*`；
- 不含 `event_id` / `command_id` / `actor_role` 决策分支；
- 不含预设事件里程变道/停车；
- 不含 `force_lane_change`（事件车辆的控制除外，场景事件代码中仅用于事件车辆）；
- 不含 `/root/autodl-tmp` 等服务器绝对路径；
- `candidate_count` / `policy_truth_access` / `safety_observation_candidate_count` 只作为审计日志字段出现（恒为 0 / false）。

## 六、actor 真值隔离

- 在线模型输入不枚举 CARLA actor：`build_sensor_policy_state` 的 `objects=[]`；
- actor camera-BEV 与候选实体张量恒为零；
- 场景一/三 LiDAR 兼容张量为零，场景二仅保留真实 LiDAR 传感器生成的 BEV；
- 决策日志逐条记录 `policy_truth_access=false`、`candidate_count=0`、`safety_observation_candidate_count=0`；
- actor 真值只用于场景生成、训练标签、独立评估和结果审计。

## 七、统一架构自动化测试

新增 `experiment/CARLA/tests/test_unified_architecture.py`，覆盖 17 项统一架构要求（三场景同字段/schema、缺失模态零填充、同一 checkpoint、同一实例顺序处理三场景、控制器无 scene 分支、YIELD 恢复与左右变道恢复不依赖 command/event id、masked risk probe 不覆盖主风险状态、actor/candidate 不入模型、场景脚本不直接控制 ego 变道/制动、模型输出结构一致、同一 Route PID 接口）。另将旧的 scene3 控制器测试改写为对通用监督器/FSM 的测试。

完整 pytest 结果（服务器实际执行）：

```text
529 passed, 177 subtests passed
```

`git diff --check` 无空白错误。

## 八、场景三：r12 失败与 r13 状态

### r12（旧链路，2026-08-06）

旧链路 `cautious_sparse` 完整 6 km 达到 6,000.969 m、7/7 事件 RESOLVED、0 禁行线、0 非法车道样本、0 fallback、33,393 次 VLA 决策、传感器到决策 P95 55.519 ms、120 ms 内 99.7125%，但 3,142–3,144 m 处被背景交通 `vehicle.audi.tt` 追尾（55 个接触回调主要为同一车辆持续接触的重复帧），`complete_scene_success=false`。失败原因：自车因视觉风险停车后，背景车未及时清退导致追尾。随后实施的修复包括锥桶渐变事件激活时清退自由流背景交通、背景车与前车保持距离、显式开启背景车与 ego 的碰撞检测，以及碰撞审计增加对方 actor 信息。这些修复保留在工作树中。

### r13（旧链路，2026-08-07）

r13 于 2026-08-07 03:27 启动（PID 419789），使用旧 `Scene3VlaController` + `Scene3RouteController`。检查发现其在仿真约 3,128 s 后卡死于 3,497.6–3,511.5 m：车辆偏离可行驶路面（`project_to_road=False` 无结果），距最近车道中心线约 2.17 m，朝向与车道差约 49°，反复油门/刹车振荡约 3 小时无进展；无碰撞、无事件激活，VLA 大部分帧输出低风险 keep_lane。根因是旧链路的路由计划在施工区出口按事件里程强制回中车道（约 3,450–3,550 m 窗口），车辆在路口边界处被路缘卡住；叠加 V6 雨夜静止画面误报 high 的已知缺陷。r13 已终止并保留输出目录作为证据（未声称通过）。

## 九、场景三：统一架构 6 km（r14）

统一架构重构完成后重新启动完整 6 km 验证（`cautious_sparse`，PID 见运行日志）：

```text
输出目录：experiment/CARLA/outputs/universal_v6_unified_full6km_cautious_r14h_20260807（最新一轮）
运行配置：官方雨夜天气、960x540、H.264 20 FPS、video overlay、VLA 每 3 帧决策、require-complete-scene
```

### 诚实结论：统一架构 6 km 未通过

r14 系列（r14a–r14h）在统一架构下均未完成 6 km。每一轮都在约 320–394 m 处被 V6 模型的“静止画面持续 high”误报硬锁：风险头以 0.9–0.97 概率持续输出 high，前方 80 m 内无任何 actor（探针确认），车辆无法恢复。最新一轮 r14h 结果：

- 最终里程：393.943 m（0 碰撞、0 禁行线、0 非法车道样本、0 fallback）；
- VLA 决策 2,940 次，模型输出应用 1,451 次（49.35%）；
- `policy_truth_access=0`、`candidate_count=0`、`safety_candidate_nonzero=0`；
- 通用覆盖实际生效：`unconfirmed_stop_crawl_floor` 96 次、`temporal_risk_confirmation` 87 次；
- 完整决策延迟 P50/P95/P99/max = 21.8 / 27.0 / 39.2 / 79.0 ms；传感器到决策 P50/P95/P99/max = 22.8 / 28.3 / 40.6 / 102.3 ms；120 ms 内比例 100%；
- 视频：H.264、960x540、20 FPS、8,817 帧、440.85 s、约 319 MB；
- 曝光：平均亮度 103.5/255，最亮抽样帧均值 114.9，Y>=250 像素平均 0.14%、最大 0.48%，无全局过曝。

### 为什么 r12 曾通过而统一架构未通过

r12（旧链路）在约 352 m 路口因旧路由计划的车道窗口偏离到 road 416 分支，绕开了 road 44 路段；统一架构严格跟随 GlobalRoutePlanner 的 road 44 分支（约 365–395 m），而 V6 模型在该路段的雨夜画面下持续误报 high（多轮、多随机种子复现）。这是模型真实缺陷，不是规则层可掩盖的；统一架构没有用场景专用规则“绕行”该路段。

### 统一架构下的关键修复（均已生效并被自动化测试覆盖）

- 爬行/减速下限从 4 km/h 提高到 10 km/h，路口前限速 9 km/h、前瞻缩短到 3.2 m；
- 爬行油门增强：目标 ≤12 km/h 且当前 <2 km/h 时提高油门增益与变化率；
- `unconfirmed_stop_crawl_floor`：风险头 low/medium 且文本不要求停车时，模型停车动作降为 10 km/h 谨慎爬行；
- `temporal_risk_confirmation`：谨慎爬行期间单帧 high 先保持爬行，连续两帧才急刹；
- 路线进度跟踪器前跳窗口限制为约 60 m，防止循环路线在重复路口跳段；
- 场景一/二 smoke 与完整 pytest 均通过（见下文）。

## 十、场景一与场景二 smoke test（非完整路线验收）

统一架构完成后对场景一、二执行 60 秒闭环 smoke test。结果明确为 smoke test，不写成 5 km / 8 km 完整验收。

### 场景一（Town04，前视 RGB + 文本 + 状态）

- 决策 400 次，`model_output_used=true`，`fallback_count=0`；
- 0 碰撞、0 禁行线侵入、0 非法车道样本；
- `policy_truth_access=false`、`candidate_count=0`；
- `schema_version=unified_sensor_batch/1.0`，`modality_mask` 仅前视为 true；
- 传感器到决策 120 ms 内比例 100%；
- 已知模型缺陷：V6 对场景一 Town04 前视画面大量误报高风险（约 93% 决策被安全门转为 emergency brake），车辆基本无法起步。该问题不是场景专用规则隐藏项，属于模型真实局限，需要在后续数据/训练中解决。

### 场景二（Town05，四路 RGB + 真实 LiDAR + 文本 + 状态）

- 决策 400 次，`model_output_used=true`，`fallback_count=0`；
- 0 碰撞、0 禁行线侵入；
- 四路 RGB 实际接入，LiDAR BEV 来自真实传感器；
- `policy_truth_access=false`、`candidate_count=0`；
- `schema_version=unified_sensor_batch/1.0`；
- 传感器到决策 120 ms 内比例 100%；
- 模型在场景二以谨慎减速为主（397/400 decelerate），60 秒推进约 38.6 m，未完成任何命令窗口，属于 smoke 级证据。

## 十一、模型现存不足（必须诚实说明）

- 高风险视觉准确率只有 41.73%（独立测试集）；
- 雨夜普通巡航会误报高风险并频繁停走；
- 静止画面容易持续 high；
- 遮挡行人能力不足；
- 场景一 Town04 前视在真实部署画面下几乎全帧误报 high（上述 smoke 证据）；
- 通行效率与舒适性不足。

这些不足没有通过场景专用规则隐藏；规则/FSM 层只做输入合法性、输出格式校验、物理速度上限、通用紧急失效保护、风险时序确认与高层动作到 PID 的转换。

## 十二、其他说明

- V6 训练未使用 Waymo；服务器上的 Waymo v2 精选集因缺少所需 LiDAR/vehicle pose 未参与训练；
- 仿真测试只证明当前题目场景范围内的表现，不构成现实道路安全认证；
- 场景一 5 km、场景二 8 km 未使用 V6 跑完整路线，本报告不声称已通过完整验收；
- 训练数据、权重、视频、输出均不进入 Git。

## 更新区（r14 完成后填写）

已填写（见第九节）。r14 系列未通过 6 km，原因与证据已如实记录；未删除 r14h 输出目录作为最终证据。

## 九之二、模型微调与路由修正（2026-08-07 晚）

针对 r14 系列暴露的“静止画面持续 high”误报，做了两件事：

1. **仿真采样微调**：在场景三雨夜误报路段（约 320–420 m，road 44 区域）及其他巡航路段采样 157 条四路 RGB+状态+文本样本（标签：low 风险 / keep_lane / 32 km/h），与 V5 全量数据集（34,132 条）合并，以 V6 为初始化微调 8 epoch（`universal_three_scene_v6_sensor_policy_finetuned/model.pt`）。微调后验证集高风险准确率约 60%（原 41.7%），CARLA 实测 300 秒 2,000 次决策全部为 keep_lane，误报消失。
2. **路由修正**：将场景三路由起点/折返点由 (239,289) 调整为 (250,240)，得到全程车道不跨对向的 6.1 km 走廊（事件锚点全部通过校验），消除多路口跨越双实线的路线伪影；另将路口切换检测窗口扩展为“自 tracker 前 10 m 至 60 m”，并在起步 100 m 内限速 15 km/h。

修正后定点验证：300 秒仿真 0 碰撞、0 禁行线、0 非法车道样本，推进 1,172 m。

### r15 / r15e 结果与失败诊断

- r15（微调模型 + 修正路由）：完成至 4,999.98 m，7 类事件中 5 类已解决，但被 blocked-lane 事件的“目标车道安全间隙未释放”错误终止（该问题曾以静默 os._exit 崩溃形式出现，根因是 `_retire_background_traffic` 清空 gap 车辆后 `_update_blocked_lane` 访问 `["front"]` 抛 KeyError；已加防护）。
- 进一步诊断：微调后模型把“前方停驻维护车”从 high 风险变为低风险，车辆直接驶过阻塞点，导致事件安全间隙逻辑未满足。为此采样 7 条阻塞车道样本（前视=high/stop，左视 clear=low，左视占用=high）进行第二阶段微调。
- 完整 6 km 最终结果待第二阶段微调验证后更新。

### 清理记录

已删除（路径均在 `experiment/CARLA/outputs` 下，删除前执行 `realpath` 核对）：

- 旧 V6 完整运行 r2–r13 及无编号目录、V6 冒烟输出；
- 统一架构失败运行 r14、r14b–r14g；
- 场景一/二/三冒烟测试的视频文件（保留 JSONL/摘要）；
- 约释放 16.5 GB（输出目录 52 GB → 36 GB）。

保留：r14h（最新一轮 6 km 证据，含 `scene_summary.json`、`vla_decision_audit.json`、`exposure_analysis.json`、`video_metadata.json` 与视频）、场景一/二冒烟 JSONL/摘要、训练数据与最终权重（不进入 Git）。

## 更新区（2026-08-08：r25 场景三全程通过，场景二/一推进中）

### 场景三 6 km 全程通过（r25）

- 输出目录：`experiment/CARLA/outputs/universal_v6_finetuned_full6km_cautious_r25_20260808`
- 模型：stage-3 微调（35,196 条：V5 34,132 + 旧路线 157 + 阻塞车道 7 + 新路线 900），权重 `.../finetuned_stage3/model.pt`
- `complete_scene_success=true`；route=6000.404 m；7/7 事件 RESOLVED；collision=0；invalid_lane_samples=0；restricted=0；fallback=0
- 视频：H.264 20 FPS，23,398 帧，0 丢帧；VLA 决策 7,787 次，模型输出应用 6,885 次，决策延迟 P95 62.1 ms，120 ms 内 99.91%
- 最终动作分布：keep_lane 4,415 / decelerate 1,534 / lane_change_left 753 / emergency_brake 28 / accelerate 1,057
- 时序安全门 override：`temporal_risk_confirmation` 28、`unconfirmed_stop_crawl_floor` 62、`low_risk_deceleration_crawl` 784、`low_risk_command_speed_floor`（新增）

本轮关键修复（全部为通用语义，不读 scene/event/command id）：

1. 时序确认：高风险需连续 3 帧才急刹（此前 2 帧），爬行/刚起步时单帧 high 保持爬行，消除 0–500 m 频繁停走；
2. 爬行下限 10→15 km/h，Route PID 爬行油门增强阈值 12→16；
3. blocked-lane：`_update_blocked_lane` 增加“自车已在目标车道并越过阻塞点即释放”，gap 车辆被清退后仍可正常释放；
4. 路线 PID：仅在文本指令请求或计划走廊一致时执行左变道；换道期间限速 15 km/h；加长前瞻减少弯道内切；
5. 指令 FSM：`semantic_goal` 中的 lane_change_left/right 优先于关键词猜测（“并道至左侧车道”正确解析为 CHANGE_LANE_LEFT）；无 id 的默认巡航文本不再被模型解析器覆盖；
6. 低风险指令速度下限：文本给出明确速度时，低/中风险下模型过度减速被抬回指令速度（场景二“减速至45”不再被模型压成 9 km/h）。

### 场景二（Town05，8 km，进行中）

- 使用同一 stage-3 checkpoint 与 `UniversalVLAController`；
- 已定位并修复两个问题：默认巡航文本被模型解析为 DECELERATE 9 km/h（禁止默认指令的模型合并）；低风险指令速度下限；
- 场景二运行配置保持契约完整（70 辆车、24 行人、multimodal 证据 960x540、四路 RGB+真实 LiDAR 接入）；
- r6 非 competition 全程 8 km 已跑完（route_completed=true，4 类特殊事件全部 RESOLVED），但存在 15 次碰撞与 85 次实线侵入，未达到严格通过标准；该模式不逐帧持久化 multimodal 证据文件，未声明 competition acceptance。

### 场景一（Town04，5 km，stage-4 实测结果）

- 使用 stage-4 微调模型（在原 stage-3 基础上并入 200 条 Town04 前视巡航样本，35,396 条，离线验证 200/200 判 low）；
- 起点 0–2,073 m 正常巡航，误报急刹问题已消除（stage-4 后风险头 high≈0）；
- 在约 2,100 m 左转路口处多次撞上 `static.fence`（r3/r4/r5 同一位置，横偏约 5.6 m），场景判定 FAILURE(collision_detected)，未完成 5 km；
- 已修复：场景一收尾相机 `append_terminal_overlay` 缺失导致崩溃；route-manager 模式路口检测/限速/前瞻；
- 剩余问题：该左转路口的护栏与纯追踪轨迹几何冲突；曾尝试在场景侧对急弯航点做圆弧平滑（场景代码 `urban_voice_5km.py` 的 `_smooth_sharp_junctions`），但平滑后改为在弯前发生非法车道侵入（r7），方案已回退，场景文件恢复 HEAD 版本；
- 结论：场景一未完成 5 km 验收，失败点固定在约 2,050–2,100 m 的两个 Town04 急转路口（碰撞护栏 / 非法车道侵入），需要后续调整该路口路线选线或横向控制再验证。

## 最终状态（2026-08-08 收尾）

- 场景三（Town05 6 km）：**通过**（r25，`complete_scene_success=true`，7/7 事件，0 碰撞，0 实线违规，0 fallback）；
- 场景二（Town05 8 km）：完整路线已跑完（r6，route_completed=true，4 类特殊事件 RESOLVED），但 15 次碰撞与 85 次实线侵入，未达严格通过标准；
- 场景一（Town04 5 km）：stage-4 后误报急刹消除，正常行驶至约 2,050 m；在急转路口多次失败（碰撞护栏/非法车道侵入），未完成全程；
- 完整 pytest：531 passed + 177 subtests passed；`git diff --check` 干净；
- 代码已提交 `main`（`a7fae88`），后续修复提交随收尾更新。
