# 场景三 6 km 全链路测试与模型问题报告（2026-08-04）

## 1. 结论

本轮在 `main` 上完成了场景三校准、6 km 实跑、视频录制和模型旁路评测。

- CARLA 路线实际完成 `6000.969 m`，7/7 事件全部进入 `RESOLVED`，碰撞 0 次。
- 直接录制视频无丢帧：25,664 帧，1280×720，20 FPS，时长 1283.2 秒。
- 修复后的雨夜前视图不存在过曝：7 个事件样本平均亮度 112.100–120.091，最高亮部截断仅 0.042%。
- 84 项 CARLA 单元测试全部通过。
- 当前不是“模型闭环驾驶”：6 km 由 route-PID 控制器完成，模型只做离线/旁路评测，`scene_summary.json` 中 `model_output_used=false`。

当前 7 个连续小事件已能覆盖切入、施工预警、锥桶渐变、施工车辆、临时行人、阻塞换道和驶离施工区，不建议为了数量继续增加第 8 类事件。当前更需要的是同一事件的参数化变体重复测试；默认配置仍是固定种子、固定雨夜、固定事件位置，因此不能从本次单次成功推导模型具备泛化能力。

## 2. 本轮修改

### 2.1 雨夜曝光校准

四路 RGB 相机统一改为手动曝光：

- exposure compensation: 0.0
- shutter speed: 100
- ISO: 800
- f-stop: 2.0
- gamma: 2.2
- bloom: 0.1
- motion blur: 0.1
- lens flare: 0.05

过曝根因不是 CARLA 雨夜本身，而是相机先继承了 `exposure_compensation=+3`，随后又使用 ISO 1600、shutter 80 和默认 f/1.4；增益叠加后，湿路反光、路灯和雾把大量像素推到饱和区。旧前视样本均值约 246，`>=250` 的高光像素约 51.65%；修复后真实事件帧的最大高光截断只有 0.042%。因此本轮模型漏检不能继续归因于过曝。

### 2.2 场景和录制稳定性

- 修复视频采样节奏：`--record-every-n` 只控制 PNG 抽帧，H.264 视频仍逐 tick 写入。
- Town05 偶发回收横穿工人 actor 时，允许一次受控重建；再次失败仍会抛错，避免静默伪造成功。
- HUD 中硬编码事件说明改名为 `SCENARIO INTENT (REFERENCE)`，防止被误认为模型输出。
- 增加语音指令批量评测和场景事件图像 manifest 工具。
- manifest 支持 `--selection observed-target`：从同步真值中选择有证据 actor、且距离目标范围最近的帧；同时明确它只是几何可观测代理，不等同于图像中无遮挡可见。

第一次完整运行在约 3.36 km 因工人 actor 被 CARLA 回收而终止，且旧录制逻辑只有 78 帧。修复后第二次运行完成 6 km，视频 25,664 帧且丢帧为 0。

## 3. 场景三内容与可评测性

默认路线包含以下 7 个有序事件：

1. 突发车辆切入与紧急避让；
2. 施工区提前预警；
3. 锥桶渐变/收窄；
4. 施工车辆与施工区域通行；
5. 临时施工人员横穿；
6. 前方车道阻塞并向左避让；
7. 驶离施工区并恢复正常行驶。

这些事件已足够检查场景三要求中的环境感知、模糊语音理解、安全减速、让行、避障、换道及恢复驾驶。全程真值 1,281 条，其中 `OBSERVED=183`、`SCHEDULE_ONLY=1098`，7 类事件最终证据模式均为 `OBSERVED`。

仍需注意三个评测边界：

- 固定随机种子 `20260729`，只有一组交通流与 actor 行为；
- 固定雨夜配置，尚无白天、不同雨雾强度或逆光的成组对照；
- 切入车在有效真值阶段选到的最近可用前视样本仍约 38.06 m，目标很小；几何真值虽为 observed，图像中不一定清晰可辨。

后续最有价值的变体不是增加新事件类型，而是对现有 7 类事件做参数矩阵：actor 距离/速度/TTC、目标左右位置、降雨雾强度、昼夜、语音同义改写、SNR 和背景交通密度。每个组合至少多 seed 重复，才能形成召回率、误报率和成功率，而不是单次演示结果。

## 4. 6 km 仿真结果

| 指标 | 结果 | 解释 |
|---|---:|---|
| 实际路线长度 | 6000.969 m | 达到 6 km 要求 |
| 事件完成 | 7/7 | 全部 RESOLVED |
| 碰撞 | 0 | 本次 route-PID 基线 |
| 原始 lane-invasion 回调 | 197 | CARLA 车道线接触的原始回调，不能直接等同 197 次违规 |
| 非法车道采样 | 0 | 基于道路/车道合同的有效违规为 0 |
| chase 视频帧 | 25,664 | 0 丢帧 |
| 完整场景成功 | true | 仿真与事件合同通过 |
| 模型参与控制 | false | 模型未闭环 |

`lane_invasion_event_count=197` 与 `invalid_lane_samples=0` 并不冲突。前者会在换道、贴线及重复路线连接处被频繁触发，后者才检查是否进入合同外车道。后续评分若要使用压线指标，应先做时间去抖、同一车道线聚类和允许换道窗口过滤。

## 5. 模型与链路测试

### 5.1 规则语音解析器

输入是配置中的中文文本，并非带雨声的真实音频或 ASR 输出。8 条指令中 5 条有效：

- “前方进入施工区域，注意观察”返回空；
- “危险路段结束，保持安全并逐步恢复车速”返回空；
- “前方车道受阻，确认安全后向左避让”触发 `AVOID requires a target`。

问题集中在监控型提示、恢复车速语义，以及“向左避让”在 AVOID 与 CHANGE_LANE 之间的消歧。

### 5.2 ModernBERT 指令模型

该项使用人工核对的英文语义输入评测，不包含中文 ASR/翻译耗时，因此不是中文端到端结果。

- 7/8 为 `VALID`，1/8 为 `NEEDS_CLARIFICATION`；
- 预热 4.152 s；服务平均 50.559 ms，最大 94.882 ms；
- 明确“向左换道”的阻塞指令仍被判需澄清，置信度 0.9023；
- “通过施工区”和“恢复正常行驶”只输出 `KEEP_LANE`，遗漏安全速度变化/恢复速度；
- 切入指令可输出 `ADJUST_SPEED + YIELD + AVOID`，但置信度仅 0.5159。

结论：时延基本可用，语义槽位和恢复动作训练不足；高置信度错误澄清尤其需要优先修正。

### 5.3 MiniCPM-V 4.6

使用前视相机、几何 observed-target 事件帧，高分辨率/多 slice 配置：

- JSON schema 7/7 有效；
- 平均推理 3.677 s，中位 3.578 s，最大 4.2493 s；
- 7/7 均输出 `objects=[]`；
- 文本摘要能提到红色车辆、黄色厢式车等，但没有生成结构化检测框；
- 7/7 把雨夜/湿路误判成雪景。

低分辨率配置只做到 5/7 schema 有效；提高分辨率改善了格式稳定性，但没有改善目标落框。约 3.7 s/帧也远高于 120 ms 级在线预算。当前 MiniCPM 只能作为慢速语义旁路，不能承担实时主感知。

### 5.4 YOLO 专项模型

`conf=0.1`、`imgsz=640` 下首帧冷启动 1849.278 ms，之后约 13.494–16.028 ms，热态时延合格。

- 施工红车：正确检测 vehicle，置信度 0.82556；
- 阻塞黄色厢式车：正确检测 vehicle，置信度 0.86134；
- 临时行人：漏检 pedestrian，只检测到车辆；
- 锥桶：漏检 traffic_cone；
- 切入：未检出车辆，并产生多个低置信度 traffic_light 误报；
- 施工/出口帧也存在 traffic_light、traffic_sign 低置信度误报。

结论：速度够，但锥桶、行人和远距离切入召回不足，低阈值下误报明显。应补充 CARLA 雨夜小目标数据，按类别分别校准阈值，并报告每类 PR/召回而不是只展示个例。

### 5.5 VLA 与控制闭环

仓库有 VLA adapter 和结构化 BEV rasterizer，但当前场景 runner 没有把实际相机/LiDAR转换成 VLA 所需 tensor，也没有把 VLA 输出接到车辆控制。因此本轮不能给出“VLA 完成 6 km”的结论。

此外，现有权重说明带有 SimLingo 衍生使用限制。在许可证和训练来源明确前，建议仅做离线/仿真 shadow 评测；若需要提交为可驾驶链路，应使用许可兼容数据重新训练并补齐安全门控、超时降级和控制仲裁。

## 6. 当前全链路的真实边界

本轮已验证的链路是：

`CARLA 场景与传感器 -> 同步真值/视频 -> 中文文本规则解析 + 人工核对英文的 ModernBERT -> YOLO/MiniCPM 离线事件帧评测 -> 问题报告`

尚未验证的链路是：

`真实含噪语音 -> ASR -> 中文语义模型 -> 融合感知 -> VLA -> 安全仲裁 -> CARLA 闭环控制`

配置虽然声明了雨声、道路噪声和 SNR 18 dB，但 runner 当前只输出文本指令 schedule，没有合成音频和 ASR，因此不能宣称完成噪声语音测试。

## 7. 视频与产物

推荐提交压缩版：

```text
experiment/CARLA/outputs/scene3_full_chain_20260804_r2/scene3_6km_full_compact.mp4
```

- H.264，1280×720，20 FPS
- 25,664 帧，1283.200 s（21:23.2）
- 518,757,049 bytes
- SHA-256: `e50bbbf06cee4c4506bf5557755975ce6013c198a13554ebb219e0534704ce76`

原始高码率版：

```text
experiment/CARLA/outputs/scene3_full_chain_20260804_r2/scene3_6km_full.mp4
```

注意：本次视频在 HUD 文案修正前开始录制，画面内旧标签 `STRUCTURED INTENT (RULE)` 是硬编码场景参考意图，不是模型输出。代码已改为 `SCENARIO INTENT (REFERENCE)`，但没有为只改标签而重跑 21 分钟视频；提交说明中必须保留这一解释。

## 8. Linux 相对路径复现

从仓库根目录执行，镜像内通过变量挂载模型权重，不把服务器绝对路径写进脚本：

```bash
export MODEL_ROOT="${MODEL_ROOT:-../models}"
export SCENE3_OUTPUT_DIR="experiment/CARLA/outputs/scene3_reproduction"

bash experiment/CARLA/tools/run_scene3_linux.sh \
  --camera-mode four-view-plus-chase \
  --record-images \
  --record-every-n 20 \
  --record-ground-truth \
  --ground-truth-every-n 20 \
  --video-output "$SCENE3_OUTPUT_DIR/scene3_6km.mp4" \
  --video-overlay \
  --video-fps 20

python experiment/CARLA/tools/evaluate_scene3_voice_schedule.py \
  --schedule "$SCENE3_OUTPUT_DIR/voice_command_schedule.jsonl" \
  --output "$SCENE3_OUTPUT_DIR/command_parser_results.json"

python experiment/CARLA/tools/prepare_scene3_event_manifest.py \
  --run-dir "$SCENE3_OUTPUT_DIR" \
  --image-dir "$SCENE3_OUTPUT_DIR/rgb/front_rgb" \
  --camera-name front_rgb \
  --selection observed-target \
  --target-distance-m 20 \
  --output "$SCENE3_OUTPUT_DIR/front_observed_target_manifest.jsonl"
```

Docker 镜像应包含 CARLA Python API、FFmpeg 和各模型环境；模型目录由 `MODEL_ROOT` 或容器 volume 注入。不要把 `/root/autodl-tmp/...` 固化到配置或提交包中。

## 9. 后续优先级

1. 先补真实音频合成/回放和 ASR，消除“配置有噪声但链路无音频”的断层。
2. 修规则解析与 ModernBERT 的左换道、恢复速度和监控型提示语义。
3. 为 YOLO 增加雨夜行人、锥桶、远距离切入训练样本并做类别阈值校准。
4. 让 MiniCPM 的自然语言摘要与结构化 objects 一致，并评估更小模型或异步低频使用。
5. 接通真实传感器到 BEV/VLA，再用安全仲裁做 shadow 对比；在此之前继续保留 route-PID 基线。
6. 对现有 7 类事件做多 seed 参数矩阵，不以新增事件数量替代泛化测试。
