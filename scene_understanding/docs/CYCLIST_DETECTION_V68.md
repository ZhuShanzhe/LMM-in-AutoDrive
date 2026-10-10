# 自行车通用检测增强（2026-10-08）

## 已完成的工作

保留原 COCO 通用检测器的车辆、行人、灯牌等类别，增加可复用的自行车检测融合接口。检测网络只接收图像，不读取指令、地图编号、任务演员 ID 或任务完成真值。显式检测到 `cyclist` 才能增加自行车候选，不能把普通行人或摩托车规则性改名为自行车。

本轮从已发布的 `UNIC0RN-Zhu/yolo11s-drive-scene-carla-v1` 权重微调，而不是直接替换整套感知模型。权重类别按文件内 `names` 解码；原网页对第 6、7 类的说明与文件不一致，不能按网页类别编号硬编码。使用模型及衍生权重须遵守对应的 AGPL-3.0 授权与训练数据使用条款；本文不授予超出原许可的数据或模型权利。

## 数据与训练

- 采集 1,872 张 CARLA 图像：训练 1,044 张、验证 324 张、测试 504 张。
- 三种自行车，加入摩托车、行人和汽车；覆盖 8/15/25/35/45/55 米、三个朝向偏移、晴天/阴湿/落日/雨天。
- 训练与验证使用 Town05 的不重叠位置，测试使用 Town04。这里的“独立”指本次微调的数据划分，不能声称原始预训练从未见过这些地图。
- 同帧语义掩码与目标三维投影区域联合生成可见边界框；RGB 与标签帧号一致。CARLA 实例颜色编号不是客户端演员 ID，未将两者混用。最初两轮不合格采集未进入训练。
- 混入 600 张既有 nuScenes 训练图像回放，训练 8 个 epoch。数据顺序、初始权重 SHA256、实际训练参数与日志均保留。
- 标签真值只用于离线训练与评测，不输入在线控制。

## 图像检测结果

同一组 Town04 测试图像，置信度阈值 0.25、匹配 IoU ≥ 0.5，自行车真阳性标注 243 个，负例图像 261 张。负例包括其他参与者与目标不在有效视野的情况。这里的指标针对完整骑行者/自行车目标框，与 COCO 的单独自行车框定义有差异。

| 权重 | TP / FP / FN | 精确率 | 召回率 | 负例图像误检数 |
| --- | --- | --- | --- | --- |
| 原 COCO YOLO11s | 10 / 48 / 233 | 17.24% | 4.12% | 8 / 261 |
| 已发布 CARLA 权重 | 64 / 25 / 179 | 71.91% | 26.34% | 20 / 261 |
| 本轮微调候选 | 183 / 14 / 60 | 92.89% | 75.31% | 12 / 261 |

新候选按距离的召回率：8 米 45/45、15 米 38/45、25 米 40/45、35 米 28/36、45 米 22/36、55 米 10/36。远距离小目标仍有漏检，不能写成“全部距离识别达标”。

8～25 米合计召回 123/135，即 91.11%。天气覆盖包含落日与雨天，不包含真正夜间；本次测试不能证明夜间或真实道路泛化已经达标。

曾尝试直接合并基础模型与新候选的自行车预测：同组回放召回率仍为 75.31%，精确率降至 88.83%，负例误检图像增至 20/261。因此当前增强接口由新专家负责自行车类别，其他基础类别保留；只消除高度重叠的同一骑行者重复框，不把其他行人或摩托车统一改为自行车。这一融合设计使用了测试回放诊断，不能将它描述为完全未参与方案选择的最终盲测。

原全程失败记录中的三张自行车漏检画面，新候选均给出自行车类别，置信度约 0.687/0.642/0.688；这是诊断回放，不计入上述独立测试集，也不能替代闭环验证。

以上是单模型图像指标，不是融合检测器指标、指令准确率、三场景通过率或 98.5% 决策准确率。批量离线网络计时不作为流式或全链路延迟。

## 接口与使用

输入为 BGR NumPy 图像或现有图像入口使用的 PIL RGB 图像；输出沿用 `Detection`，包含 `bbox_xyxy`、`confidence`、`category`、`subtype` 与 `class_id`。自行车输出使用统一类别，保留实际置信度。`detect_many` 保留视角顺序，数量不匹配时拒绝输出，不能静默截断。

```python
from pathlib import Path
from scene_understanding.realtime_perception.ultralytics_backend import UltralyticsTrafficDetector
from scene_understanding.realtime_perception.cyclist_detection import CyclistAugmentedDetector

base = UltralyticsTrafficDetector(Path("models/pretrained/yolo11s.pt"), device="cuda")
expert = UltralyticsTrafficDetector(Path("models/challenge/cyclist_detector_v68.pt"), device="cuda")
detector = CyclistAugmentedDetector(base, expert)
detections, detector_ms = detector.detect(bgr_image)
```

需要现有 `scene_understanding/requirements-realtime.txt` 环境、Ultralytics、Torch 与 NumPy；采集还需 CARLA 0.9.16。训练与评测脚本分别为 `scene_understanding/scripts/collect_cyclist_detection.py` 与 `train_cyclist_detector.py`，参数由 `--help` 列出。生成的数据、输出与权重不加入 Git 源码。

在线条件观察器通过 `pedestrian_condition_observer.cyclist_expert` 指定 `weights` 和 `sha256`，加载前强制核验。通用融合类可在其他感知入口复用，不依赖自行车指令或指定路线。

已有通用图像入口可直接启用，无需修改指令或地图：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
cd /root/autodl-tmp/worktrees/challenge-track
python -m scene_understanding.realtime_perception.run_dataset \
  --manifest /absolute/path/images.jsonl \
  --output /absolute/path/perception.jsonl \
  --summary /absolute/path/perception_summary.json \
  --backend yolop_yolo11 \
  --yolop-root /root/autodl-tmp/models/external/YOLOP \
  --yolo11-weights /root/autodl-tmp/models/challenge-assets-pinned/pretrained/yolo11s.pt \
  --cyclist-weights /root/autodl-tmp/models/challenge-assets-pinned/challenge/cyclist_detector_v68.pt \
  --device cuda
```

输入 JSONL 每行示例：`{"image_path":"/absolute/path/front.jpg","frame_id":"frame-0001","source":"carla"}`。该入口对独立图像重置跟踪器，不作为连续追踪评测。连续 CARLA 采集入口 `python -m scene_understanding.realtime_perception.run` 同样接受 `--cyclist-weights`，使用原有 `--capture-index`；两个入口均要求 `--backend yolop_yolo11`，不支持的组合立即报错。

生成带自行车候选的原生挑战运行配置：

```bash
python lightweight_vla_adapter/scripts/prepare_challenge_runtime.py \
  --model-root /root/autodl-tmp/models/challenge-assets-pinned \
  --map Town05 --output /absolute/path/runtime.json \
  --cyclist-weights challenge/cyclist_detector_v68.pt \
  --cyclist-sha256 c8ea834feebb1840a0ce0f77f1e505e6c73c0c5b8046c29e8be5a6f465d5b5bd
```

将生成的 `runtime.json` 作为现有原生运行器的 `--vla-config`。相对权重必须位于 `--model-root` 内；路径、软链接越界或哈希不匹配均拒绝加载。未指定两个自行车参数时不自动启用，避免旧下载包缺少权重而静默降级。

候选 SHA256：`c8ea834feebb1840a0ce0f77f1e505e6c73c0c5b8046c29e8be5a6f465d5b5bd`。当前权重保存在服务器模型目录，尚未发布至 Hugging Face，也未替换已发布下载包；不能把旧下载包描述为已包含该候选。

## 与执行链路的边界

图像识别不是超越完成证据。在线仍需要同步 RGB/LiDAR 测量、物体表面支持、唯一身份、当前运动与道路几何，后方实际净距和持续确认均保留。指令明确让车辆进入某个方向的安全车道时，不额外虚构“必须先识别自行车”的前置条件；换道自身引用目标、或车辆超车任务依赖特定目标时，仍要求当前帧已获取对应目标。任何自行车超越步骤都必须独立绑定实际目标并获得实测后方净距，不能把换道或经过路线终点当作超越成功。

弯道目标获取使用当前身体测量和局部道路投影，不把自车直线坐标中的大横向偏移直接视为目标离开道路。实际完成判定仍基于当前目标身体，不使用路线位置预测证明完成。

V68 最终回归：3,355 项测试与 207 项子测试通过。训练完成后的闭环记录与独立安全判据另外保存；本次图像改进不等于三个场景全程已通过。未执行 Git 提交、推送或 Hugging Face 上传。

V68 原生自行车片段 `s2_cyclist_v68_fragment29` 已结束，结果为 `CHECK_FAILED`，源码冻结检查无变化。车辆未获得自行车超越对象，因此未发起要求目标已解析的换道。独立快照显示近处背景车与更远处弯道自行车共存：示例帧 124270，近车约 14.73 米，自行车在自车坐标前方 39.27 米、横向 9.92 米。横向值包含弯道形状，不能直接当成“自行车不在道路上”；同时前车会遮挡目标。保留这一失败，不把图像检测改善写成闭环超越已完成。

V69 增加局部道路获取后仍未完成：近车遮挡和不必要的自行车预绑定条件使车辆等待，部分远处/图像边界检测没有足够点云身体支持，不允许据此伪造超越证据。V70 修正显式方向换道与目标依赖换道的边界，保留这些失败轮次。

## 闭环与接口验证结果

V70 原生片段 `s2_cyclist_v70_fragment31`：从约 1700 米行驶至 2255 米，150 秒仿真、3,000 个决策帧；减速、向左避让、超越自行车、返回右侧四项注册准则全部通过，运行时计划 `COMPLETED`，原生退出状态 `RECORDED_CHECKS_PASSED`。源码运行期间没有变化。

独立同传感器帧核查：首次绑定身体到指定自行车演员中心 0.544 米，超越完成时 0.125 米；同一身体 ID 的实测后方净距 10.445 米。真值只在独立审计中读取，不反馈给策略。没有记录碰撞、实线违规、超速、红灯违规或接口回退；15 次合法虚线跨线事件未计为违规。红灯审计没有发生停止线跨越，不能据此声称已充分覆盖红灯任务。

完整策略墙钟 P50 88.1965 ms、P95 148.3234 ms、最大 1190.693 ms，122 帧超过 150 ms（4.07%），没有删除离群帧。计时不含 ASR、J6P 或控制命令之后的车辆物理响应。

注册四步通过不是全语义验收：独立评测仍为 `coverage=PARTIAL`、`instruction_status=UNVERIFIED`，没有完整覆盖动态并道间隙、返回最右车道等语义要求。这个冷启动片段也不代表三个场景全程通过。

V71 补齐通用 CLI、哈希验证和批量视角一致性回归；3,379 项测试与 207 项子测试通过。实际通用入口 12 张图像冒烟运行通过，包含自行车跟踪结果且道路/历史接口正常；该小规模运行不作为新的精度或全链路性能验收。

完整证据保存在 `/root/autodl-tmp/review_logs/20261007/full_routes/` 下的 `cyclist_detector_v68/`、`s2_cyclist_v70_fragment31/`、`cyclist_generic_cli_v71/` 与 `v71_regression.log`。训练数据位于 `/root/autodl-tmp/datasets/perception/cyclist_semantic_v68/`。新模型、数据和日志未提交到 Git，未执行推送或 Hugging Face 上传。
