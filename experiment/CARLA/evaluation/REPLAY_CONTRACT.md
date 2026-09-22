# 同源回放接口

`SynchronizedReplayDataset.replay(consumer)` 按记录帧序向回调传递 `ReplayFrame`。
每帧包含四视角 RGB 和 LiDAR 文件路径、帧号、仿真时间及指令请求 ID。
`vehicle_state` 为直接的自车遥测对象，不再是完整 WorldState；旧调用方访问
`vehicle_state["ego"]` 时应改为直接读取 `vehicle_state`。

允许字段：speed_kmh、location、rotation、velocity_mps、acceleration_mps2、
angular_velocity_deg_s、control。嵌套字段同样按白名单复制。
不传递 actor_id、地图车道、道路限速、天气真值、其他 actor 或违规统计。
自车位姿是仿真遥测，不应宣称为感知模型预测的定位结果。

原始 `world_state.jsonl` 不修改，完整原始状态仍参与 integrity_manifest 的
SHA-256 计算，用于同源验证和离线评测。接口白名单不是进程级安全沙箱；
部署时若需禁止模型读取原始真值，应隔离评测目录权限，避免向模型挂载完整数据集。

回放要求完整同帧传感器、有限且严格递增的时间戳，以及 bundle 与状态时间一致。
缺失帧不会补邻帧。遥测至少需要 speed_kmh；其他字段若存在则检查其分量。
回放只提供传感器文件引用，不完成图像解码、点云预处理、相机标定适配或模型推理。
每帧的 driving_intent 按请求 ID 从 driving_intent.jsonl 关联，携带 input 原文、
intent 结构及 parse_result 来源。无请求 ID 时为 null，不自动沿用上一指令。
引用缺失、重复 ID、发布时间或帧号晚于回放帧均拒绝加载。历史数据只有请求 ID
而没有对应指令记录时不能作为完整指令链路回放，不会自动伪造指令。
competition_schedule 表示脚本构造，不能计为解析模型的正确预测。
指令顶层的 route_s_m 等额外字段不传递；input、intent 和 parse_result 属于
上游指令接口内容，此处保留其语义，不代替上游完整 schema 或可信来源校验。
同源 manifest 1.3 覆盖各帧指令、时间、场景上下文、标定及声明的原始LiDAR文件；不可与旧版哈希直接比较。

新采集由 ExactFrameSensorSuite 写出 sensor_calibration.json，包括实际传感器属性、
sensor_to_ego 齐次矩阵、相机光学坐标变换及理想针孔内参。镜头畸变及后处理属性
原样保留，不宣称已完成畸变校正。LiDAR记录安装矩阵及实际属性，PLY字段以文件头为准。
ReplayFrame.sensor_calibration 暴露该静态标定；旧数据没有此文件时为null。
命令行 --require-calibration 可拒绝缺失标定的旧数据，不能猜测或补造标定。
加载器目前检查标定版本及模态齐全性，尚未完成矩阵数值和真实图像投影一致性验收。

新采集同时保存lidar/<frame>.xyzi.bin，直接保留CARLA回调的float32四列
x/y/z/intensity，小端编码、传感器局部坐标。PLY仍用于可视化，不代替原始强度。
只有PLY与原始文件均完成落盘才标记LiDAR帧可用。回放artifacts中的lidar_raw
提供该路径，evaluation.raw_lidar.load_xyzi读取为N×4数组，空点云为0×4。
旧数据缺少原始文件不补零或猜测强度；依赖强度的模型适配器须明确拒绝这种输入。

## 团队模型接入差异

当前UniversalVLAController使用物理前后雷达、四相机、LiDAR、静态道路/信号几何、
路线和有状态指令执行反馈；五路传感器回放尚不足以复现该完整控制器。
carla_multiview_sensor中的LiDAR栅格器实际使用intensity生成第四通道，不能用
缺少强度的XYZ点云替代。该rig的相机安装高度/俯仰与ExactFrameSensorSuite也不同，
仅调整图片尺寸不能视为等价输入。新标定和原始点云记录解决数据描述/保存问题，
不自动解决相机域差异、缺少雷达或控制器与在线CARLA对象的耦合。
正式适配还需统一采集rig或直接记录模型rig的输入，并保存雷达、静态地图标识、
路线及执行反馈；不得把这些缺项伪造成空场景后声称完整模型回放成功。

### 直接录制团队模型传感器组

UniversalVLAController运行配置可添加sensor_recording_dir，值为一个尚不存在的输出目录。
默认不录制；不另生成传感器、不改变相机安装参数及模型预处理。关闭控制器时完成日志收尾。
目录包含sensor_frames.jsonl、sensor_calibration.json、capture_summary.json和各传感器原始bin。
相机保留原始BGRA（包括高分辨率前视图，尚未resize），LiDAR保留XYZI；雷达保留原始
velocity/azimuth/altitude/depth四列，其中角度是弧度。每条记录保留原始帧号、时间及可用的
曝光时sensor_to_world矩阵。雷达不因录制而强制补齐到相机帧。

此录制格式model_rig_raw_capture/1.0不同于ExactFrameSensorSuite的PNG/PLY格式，
由ModelRigReplayDataset读取，不传入SynchronizedReplayDataset。
RECORDED仅表示有回调记录且没有已捕获写入错误，不证明所有模态、全程帧或模型决策齐全。
缺失capture_summary.json应视为未完成收尾。错误/重复帧将标INVALID，不补帧、不覆盖旧目录。
同步写盘会增加回调耗时，完整原始相机流也会占用较多磁盘；当前适合短段联调取证，
长程录制前还需增加容量控制/异步写入并量化开销。不得混用开关录制两种条件下的延迟。

consumed_inputs.jsonl进一步记录每次已完成决策实际选择的相机/LiDAR帧及前后雷达帧，
不以日志写入时的最新回调覆盖选择结果。context包含本次车辆/环境特征向量、视角和模态
掩码、指令及当前步骤、雷达原始观测和用于雷达走廊过滤的路线点；decision沿用完整
决策日志，包括高层动作、约束改写和上一物理步的执行反馈。
关闭录制时检查所有消费引用是否实际保存；雷达不可用或引用缺帧会标INVALID。
零决策但有传感器数据的录制仍可能标RECORDED，必须另外检查decision_records，
不能将它当作模型已完成推理的证明。早于决策帧的雷达会保留真实帧号，不伪装成同帧。
该记录不是完整循环网络状态快照，也尚未包含后续实际apply_control的确认。
记录工作发生在现有决策延迟计时之后；新日志不能用于隐去采集开销的端到端延迟声明。

原生回放使用evaluation.replay_benchmark的--format model-rig选项，仍通过--adapter
指定模型工厂。适配器收到NativeReplayFrame，包含artifacts、sensor_metadata、context
及标定；不包含录制时的decision答案。context只取已声明的指令、特征及路线字段。
decode_sensor(frame,name)将原始相机解码为HWC RGB uint8，点云/雷达为N×4 float32；
不自动resize、栅格化或伪造缺失模态，这些仍须采用模型自己的预处理。
加载时要求录制已收尾且有效、有决策记录、消费引用存在、时间不来自未来、文件大小
和编码匹配，路径不能逃逸数据目录。原生同源哈希覆盖消费输入而不覆盖历史模型答案，
允许在相同输入上比较不同输出。它不证明记录的context满足模型全部输入语义。
历史执行产生的context仍是离线固定输入，不会因新模型输出不同而改变仿真世界。

evaluation.model_rig_preprocessing.build_sensor_batch(frame,text_tokens=...,text_mask=...)
复用在线camera_rgb_tensor和_rasterize_lidar，构造团队的UnifiedSensorBatch。
录制context新增camera_preprocessing以明确尺寸和前视resize模式，旧录制缺少时拒绝猜测。
相机保持uint8，只有明确禁用的模态才用零张量及false掩码；开启却缺失的模态报错。
车辆/环境特征使用当时记录的向量；语言特征必须由调用方的真实编码器提供，
该函数不生成假token、不读取录制答案，也不恢复事件记忆或闭环控制器。
已经用合成输入验证在线/离线LiDAR特征一致及RGB处理契约，未进行真实传感器模型推理。

## 控制提交与后续响应

三个正式runner的UniversalVLA分支通过控制器apply_control提交，不重复调用ego.apply_control。
独立的<决策日志名>_execution.jsonl记录control_id、最近决策帧、提交帧/时间和完整控制参数。
SUBMISSION_REQUESTED是调用前记录，SUBMITTED仅代表API返回未报错，SUBMISSION_FAILED保留异常。
后续snapshot帧严格大于提交帧时，记录PHYSICS_OBSERVED及实际自车位置、航向、速度和
客户端报告的控制量；同帧覆盖、缺少actor或结束前没有后续观测分别标注，不伪造确认。
最近决策帧可能早于本次控制提交，尤其是复用决策或安全兜底；应结合决策/兜底日志分析，
不能由单条关联断言该油门直接来自某次模型输出。物理响应不是任务成功，也不证明精确执行时刻。
执行日志单独保存，不混入模型回放输入。该接入尚未经过真实CARLA运行验证。
离线回放无法验证动作改变环境后的闭环效果，不能替代真实闭环测试。

## 模型回放评测入口

```powershell
python -m evaluation.replay_benchmark D:/CARLA/outputs/capture --adapter team_adapter:create --config D:/CARLA/configs/model.json --output D:/CARLA/outputs/replay_run_001 --deadline-ms 150
```

上例的team_adapter:create是待接入模型的工厂接口示意，不是已提供的模型。
工厂接收配置字典，返回具有predict(ReplayFrame)方法的适配器；输出必须为可序列化
且无NaN/Inf的JSON对象。可选reset()用于清空历史，synchronize()用于等待GPU任务完成，
close()释放资源。不同模型的图像/点云解码和专用预处理放在适配器内部。

默认必须存在标定；--allow-missing-calibration仅用于不依赖标定的明确对照实验。
--max-frames可限制前N帧。新目录保存input_manifest.json、predictions.jsonl和summary.json，
已有目录拒绝覆盖。每帧输出保留原始JSON，不假定模型输出一定是油门或高层动作。
失败帧记录异常并终止，避免有状态模型跨过错误后仍被视为连续有效结果。

统计P50/P95、最大调用耗时和超过预算的有效输出数量，不计模型初始化、输入哈希、
日志写入；不自动预热，首帧纳入统计。同步钩子是否覆盖全部设备由适配器负责。
没有同步钩子时只能解释为主机调用耗时；GPU准确延迟不能由该结果直接认定。
COMPLETED仅表示指定范围逐帧调用成功，不表示指令正确、动作安全或整段闭环完成。
这不是自动接入现有团队模型的实现，具体适配器仍需按团队接口提供并验证。

命令行默认额外记录resources.jsonl，可用--no-resources关闭；Python run接口需显式
collect_resources=True。采样线程约每秒记录当前模型宿主进程RSS/CPU、nvidia-smi整卡
显存与利用率，以及已初始化PyTorch CUDA分配器的allocated/reserved值；不为统计主动初始化CUDA。
ResourceSampler可通过pids参数显式增加已知CARLA进程，但不会猜测哪一个进程属于当前实例。
进程CPU可能超过100%（多个核），第一样本不提供CPU增量。缺驱动、权限或psutil时标不可用。
summary.resources汇总采样次数、可用性与采样峰值；不是硬件计数器的精确瞬时峰值。
整卡显存不可当作模型显存，PyTorch分配器也不包括全部CUDA库；两者不相互替代。
模型构造/加载发生在采样前，不计在当前统计窗口；外部采样进程及写盘有额外开销。
# 重复回放

## 正式场景采集入口

三个正式运行器支持`--vla-record-sensors`，向当前运行输出目录下的`model_inputs/`
保存现有VLA传感器的原始回调数据及模型实际消费帧索引，不额外创建一套相机。
场景一需选择`--decision-source vla_scene_bridge`，场景二需指定`--vla-checkpoint`，
场景三需选择`--ego-controller vla-route-pid`；其余权重/配置参数仍按模型原要求提供。
显式命令行采集目录优先于模型配置sensor_recording_dir。目录已存在时拒绝覆盖。

该格式回放使用`--format model-rig`，不是标准PNG/PLY采集格式。
未指定开关时沿用模型配置，配置也未指定目录时不采集。原始多视角数据体积较大，
应先使用短时采集核对磁盘余量；同步写盘开销会影响运行速度，不能忽略该开销宣称部署性能。
capture_summary记录成功不代表已经完成完整模型离线重放；仍需适配器和完整权重。

原始采集默认上限20 GiB，剩余磁盘空间保留2 GiB；模型配置可通过
`sensor_recording_limits: {"max_raw_bytes": 21474836480, "min_free_bytes": 2147483648}`调整。
上限统计原始传感器载荷，不含索引/其他运行日志；磁盘余量每秒检查，不能保证其他进程同时写入时的硬性配额。
触发限制后停止本次传感器和消费帧记录，写capture_stopped.json，最终状态INVALID，
不将截断采集当作完整同源输入，也不改变模型控制动作。已有原始帧保留供诊断。

`evaluation.replay_benchmark`支持`--repeat N`，N大于1时每轮通过factory创建新适配器，
依次生成run_001等独立目录及repetitions.json。每轮结束调用适配器close，不复用模型内部时序状态。
使用同一输入数据及配置，不隐式修改随机种子；适配器负责按配置初始化随机数。
输入哈希跨轮变化、初始化失败或推理失败会停止后续轮次。每轮包含首帧推理，不隐式预热。
汇总提供逐轮P50/P95与预测JSON精确一致性；预测中若含自身计时字段，也会参与精确比较。
精确一致不代表语义正确或闭环成功。同一进程多轮并非冷进程/GPU冷启动性能测试。
