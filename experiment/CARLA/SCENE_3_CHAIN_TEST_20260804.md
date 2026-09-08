# 场景三现有链路测试记录（2026-08-04）

## 结论

当前 `main` 可以稳定完成场景三的 CARLA 场景基线：6 km 路线完成，7/7 事件均激活并结束，0 碰撞，四路 RGB 与逐帧车辆状态、事件真值可以同步落盘。

本次通过的是 `route-pid` 场景基线，不是模型端到端结果。输出明确记录 `model_output_used=false`；现有 `main` 没有把 ASR、多模态模型输出和外部车辆控制接入场景三，因此不能据此给出语音准确率、多模态对齐率、120 ms 应急响应或模型安全决策成绩。

## 测试环境

- Git 提交：`d7569735a3833cb86bebaab3797bfd9a2bd6326c`
- CARLA：0.9.16，官方 `Town05_Opt`
- GPU：NVIDIA GeForce RTX 5090 32 GB
- 可用解释器：`/root/miniconda3/bin/python`（Python 3.12）
- 场景配置：`experiment/CARLA/configs/scene_3_emergency_6km_runtime.json`
- 控制器：`route-pid`
- 天气：雨天夜间、湿滑路面、雾密度 35、摩擦系数 0.68

完整测试命令：

```bash
CARLA_ROOT=/root/autodl-tmp/CARLA_0.9.16 \
SCENE3_OUTPUT_DIR=experiment/CARLA/outputs/scene3_chain_test_20260804 \
PYTHON_BIN=/root/miniconda3/bin/python \
bash experiment/CARLA/tools/run_scene3_linux.sh \
  --duration 1400 \
  --fixed-delta-seconds 0.05 \
  --camera-mode four-view \
  --record-images \
  --record-every-n 200 \
  --camera-width 640 \
  --camera-height 360 \
  --record-ground-truth \
  --ground-truth-every-n 20 \
  --require-complete-scene
```

## 测试结果

### 静态与单元测试

- 场景配置校验：通过。
- 场景三测试：58/58 通过。
- 覆盖范围：7 个事件、官方地图和 6 km 路线、四路相机配置、低照/湿滑参数、动态演员、碰撞与车道审计、真值和影子评测。

### 6 km 严格运行

| 指标 | 结果 |
| --- | --- |
| 路线 | 6000.969 / 6000 m，完成 |
| 仿真时长 | 1341.7 s（22 min 21.7 s） |
| 实际运行时间 | 约 15 min 11 s |
| 事件 | 7/7 已解决 |
| 碰撞 | 0 |
| 非法车道采样 | 0 |
| 原始 lane-invasion 事件 | 203 |
| 四路 RGB | 前/左/右/后各 1341 帧 |
| chase RGB | 134 帧 |
| 车辆状态 | 26834 条 |
| 帧真值 | 1342 条，其中 OBSERVED 182、PARTIAL 1、SCHEDULE_ONLY 1159 |
| 模型输出参与决策 | 否 |
| 严格场景结构判定 | `complete_scene_success=true` |

7 个事件均有物理证据帧：预警 33、阻塞车道 40、锥桶收窄 23、加塞 23、横穿行人 25、施工区 46、施工区出口 35。

主要事件行为：

- 加塞车辆在约 54 m 间距触发并完成并入，主车无碰撞。
- 施工预警、锥桶渐变封道、施工车辆和横穿工人均正常生成并清理。
- 阻塞车道首先创建不安全左侧间隙，间隙释放时前/后距离约为 221.5/25.1 m，随后基线控制器向左变道。
- 横穿行人窗口内最低速度约 9.9 km/h；事件在边界清理，没有验证模型是否理解语音或主动让行。
- 最长近静止持续约 49.6 s，位置约 3760.6–3764.1 m。日志未记录交通灯状态，无法区分合法等灯与控制器停滞。

## 已发现问题

### P0：四路模型输入严重过曝

在横穿行人事件附近检查同步帧 `03090392`，四路画面均大面积发白，路面、车道线、锥桶与行人难以辨识。像素统计如下：

| 相机 | RGB 均值 | `>=250` 像素比例 |
| --- | ---: | ---: |
| front | 246.28 | 51.65% |
| left | 246.61 | 55.00% |
| right | 243.96 | 44.80% |
| rear | 246.72 | 52.13% |
| chase | 195.76 | 1.67% |

四路相机默认保留了 `exposure_compensation=3.0`，随后低照配置又设置手动曝光、`iso=1600`、`shutter_speed=80`，组合后造成明显高光裁剪。该输入不能用于可靠的视觉理解或多模态对齐评测。

证据帧保留在忽略目录：`experiment/CARLA/outputs/scene3_chain_test_20260804/evidence/`。

### P0：模型、ASR 与控制未形成端到端闭环

- 本次控制器为 `route-pid`，真值来源标记 `model_output_used=false`。
- `voice_command_schedule.jsonl` 只记录预设文本、语义目标和噪声配置，没有实际音频文件、ASR 结果或注入时间。
- `parse_completed_at`、`policy_started_at`、`action_started_at` 只存在于配置声明，运行输出未产生这些时间戳。
- `external` 模式只让场景释放 `role_name=hero` 的控制权，仓库中没有场景三模型控制进程负责接管。

因此当前无法验证 ASR 准确率、ASR 50 ms、应急响应 120 ms、语义对齐 97% 或四模态融合有效性。

### P0：严格成功判定只证明场景完成，不证明语义动作正确

`complete_scene_success` 当前只检查路线、事件是否按进度结束、碰撞和非法车道采样。事件可以由 PID 在不读取语音和模型输出的情况下全部“解决”，也没有逐事件断言允许/禁止动作是否被模型遵守。

### P1：默认 Linux 命令在当前服务器不可复现

文档示例使用系统 `python3`，但该解释器缺少 `numpy`。运行在构建 Town05 路线时失败：

```text
ModuleNotFoundError: No module named 'numpy'
```

Miniconda Python 3.12 可以运行。Docker 镜像必须显式包含 `numpy`、CARLA Python API及其导航依赖，并在文档中指定解释器。

### P1：运行器会吞掉未列入捕获范围的异常

主函数 `finally` 中直接调用 `os._exit(result)`。例如上述 `ModuleNotFoundError` 不属于当前捕获列表，正式入口只返回 1 且不显示异常栈，增加复测和模型接入诊断难度。

### P1：车道违规指标仍存在审计歧义

CARLA lane-invasion 传感器记录 203 次事件，而 `invalid_lane_samples=0`，最终仍判定成功。现有输出没有把203次事件分成合法变道、地图拓扑/路口触发和真实违规，暂时不能直接支撑“无违规”结论。

### P1：900 m 模糊指令没有事件级评测闭环

“前方路况危险，保持安全车速”在 900 m 被调度，但不属于7个正式事件，没有 active/resolved 窗口、模型动作或响应延时。900–1050 m 基线自身出现停车和较高制动比例，后续即使接入模型也难以区分语音响应和道路控制行为。

### P1：尚未达到30分钟稳定性评分要求

本次仿真持续 1341.7 s，约22分22秒。路线完成没有异常，但不足以证明连续30分钟无崩溃、无推理失败。

### P2：日志和存储行为容易误导

- 未配置 `--video-output` 且 `scene_summary.json` 中 `direct_video=null`，日志仍输出 `FRONT RGB DIRECT H.264 CAPTURE: PASS`。
- 四路相机按默认1 Hz保存时，完整运行产生约4.8 GB PNG；`--record-every-n 200` 主要限制 chase 记录，并未限制四路传感器按 `camera_tick` 保存。应为模型测试和证据留存分别配置采样率。

## 保留的测试证据

服务器保留约19 MB结构化输出，路径为：

```text
experiment/CARLA/outputs/scene3_chain_test_20260804/
```

其中包含运行日志、配置快照、事件时间线、车辆状态、帧真值、语音调度表、汇总文件和5张代表性图像。输出目录由 `.gitignore` 忽略，不随代码提交；本报告随 `main` 提交。
