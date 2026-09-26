# 指令调度模式

UniversalVLAController的运行配置支持command_dispatch_mode：

- route_latest：既有行为，按路线位置选择最新指令。
- completion_serial：按原始顺序派发结构化指令；当前执行计划COMPLETED后才派发下一条。

未显式配置时，控制器按命令契约自动选择：全部命令都携带有效 DrivingIntent 时使用
completion_serial，否则使用 route_latest。当前场景 2 的 VLA 调度会自动使用
completion_serial；未结构化的场景 1、3 保持 route_latest。显式配置仍优先于自动选择。

completion_serial要求每条指令携带非空driving_intent，且request_id唯一。
command_timeout_s设置单条指令从实际派发开始的超时，默认120秒。
无对应请求ID的反馈不能释放当前指令；BLOCKED继续等待；FAILED、CANCELLED或超时
中止调度，经控制器已有异常通道安全停止，不自动跳过失败项。显式结束里程已错过的
待执行指令标MISSED_COMMAND_WINDOW，不挪到错误位置执行。

结束里程字段为end_progress_m（优先）或deactivate_at_m，必须有限且大于触发里程，
在创建队列时校验。有效窗口为[触发里程,结束里程)，运行中的指令越过结束边界标
ACTIVE_COMMAND_WINDOW_EXPIRED并停止调度。到期或超时优先于同次收到的COMPLETED，
因为当前反馈没有独立完成时间/位置证据，不能用迟到的反馈追认窗口内完成。
精确窗口应预留反馈延迟余量；窗口未配置时仍仅使用时间超时，不自动猜测地图边界。

决策日志command_dispatch保存当前请求、累计完成数及状态事件。
来源为模型执行计划，不是独立仿真评测，也不证明模型的完成反馈一定正确。
benchmark侧验收仍独立计算。announce日志仍表示到达播报条件，不代表已派发执行，
后续分析应以command_dispatch为准；录音播放尚需与派发事件统一。

此模式仅完成代码及离线单元验证，尚未进行真实模型/仿真验证。
它不解决任务地点本身过近，也不自动补充缺少的结束窗口。延迟指令到达过晚时，
仍需重新设计路线间距或显式窗口，不能靠队列保证路口仍在前方。
当前不将场景真值事件完成状态传给模型决定计划完成；前序条件仍由模型执行器负责。
