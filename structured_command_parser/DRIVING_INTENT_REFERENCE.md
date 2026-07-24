# DrivingIntent JSON 完整字段参考

本文档对应 `schemas/driving_intent.schema.json` 的 `1.1.0` 版本，用于指令解析、语义对齐、风险判断、有限状态机和日志模块共同查阅。`1.1.0` 根据 Talk2Car 与 SimLingo 全量英文语料扩展动作、目标和参数，原 `1.0.0` 动作语义保持不变。

JSON Schema 是机器校验依据；本文档用于解释字段语义、字段组合和下游处理约定。二者冲突时，以 Schema 为准并提交接口问题，不得由各模块自行解释。

## 1. 完整结构

```text
DrivingIntent
|-- schema_version
|-- request_id
|-- input
|   |-- modality
|   |-- language
|   |-- raw_text
|   `-- normalized_text
|-- intent
|   |-- category
|   |-- urgency
|   |-- steps[]
|   |   |-- step_id
|   |   |-- action
|   |   |-- purpose                 可选
|   |   |-- target                  可选
|   |   |   |-- type
|   |   |   |-- relation
|   |   |   |-- description         可选
|   |   |   `-- coordinates         可选，仅保存原指令明确坐标
|   |   |-- parameters
|   |   |   |-- direction           可选
|   |   |   |-- change              可选
|   |   |   |-- target_speed_mps    可选
|   |   |   |-- speed_delta_mps     可选
|   |   |   |-- distance_m          可选
|   |   |   |-- start_distance_m    可选
|   |   |   |-- transition_distance_m 可选
|   |   |   |-- following_distance_m 可选
|   |   |   |-- duration_s          可选
|   |   |   |-- lane_count          可选
|   |   |   |-- lane_index          可选
|   |   |   |-- lane_reference      可选
|   |   |   |-- parking_maneuver    可选
|   |   |   |-- source_value        可选
|   |   |   `-- source_unit         可选
|   |   |-- trigger
|   |   |   |-- type
|   |   |   |-- step_id             可选
|   |   |   |-- distance_m          可选
|   |   |   `-- description         可选
|   |   |-- completion              可选
|   |   |   `-- type
|   |   |-- depends_on[]
|   |   |-- preconditions[]
|   |   `-- on_blocked
|   `-- constraints
|       |-- safety_first
|       |-- obey_traffic_rules
|       |-- driving_style
|       `-- max_speed_mps            可选
`-- parse_result
    |-- status
    |-- method
    |-- model
    |-- confidence
    |-- missing_slots[]
    |-- warnings[]
    |-- clarification_question       可选
    `-- latency_ms
```

## 2. 顶层字段

| 字段 | 类型 | 必填 | 允许内容 | 说明 |
| --- | --- | --- | --- | --- |
| `schema_version` | string | 是 | 固定为 `1.1.0` | 接口版本，不是模型版本 |
| `request_id` | string | 是 | 1-128 个字符 | 一条用户指令的唯一标识，贯穿所有模块日志 |
| `input` | object | 是 | 见第 3 节 | 原始输入与规范化文本 |
| `intent` | object | 是 | 见第 4 节 | 用户驾驶意图，不是最终车辆决策 |
| `parse_result` | object | 是 | 见第 13 节 | 解析过程及质量信息 |

不允许出现 Schema 未声明的顶层字段。

## 3. `input` 输入字段

| 字段 | 类型 | 必填 | 允许内容 | 说明 |
| --- | --- | --- | --- | --- |
| `modality` | enum | 是 | `TEXT`、`VOICE` | 当前输入来源；VOICE 表示文本来自 ASR |
| `language` | string | 是 | 推荐 `zh-CN` | 使用 BCP 47 风格语言代码 |
| `raw_text` | string | 是 | 非空字符串 | 用户输入或 ASR 原始结果，禁止改写 |
| `normalized_text` | string | 是 | 非空字符串 | 清洗空格、中文数字和单位后的文本 |

规范化可以将“三百米”“60公里每小时”等统一为便于解析的表达，但不能改变动作、方向、目标和先后顺序。

## 4. `intent` 意图字段

| 字段 | 类型 | 必填 | 允许内容 | 说明 |
| --- | --- | --- | --- | --- |
| `category` | enum | 是 | 见第 5 节 | 指令整体所属赛题场景 |
| `urgency` | enum | 是 | `NORMAL`、`URGENT`、`EMERGENCY` | 表达用户语义紧迫度，不代表允许越过安全规则 |
| `steps` | array | 是 | 0 个或多个 Step | `VALID` 时至少一个；其他状态可以为空 |
| `constraints` | object | 是 | 见第 12 节 | 全局安全和驾驶风格约束 |

组合指令必须按原文顺序拆成多个 Step。不得补充用户没有提出的动作，例如原文没有“回到原车道”时，不得自行增加 `RESUME` 或反向变道。

## 5. `category` 场景类别

| 值 | 使用条件 | 示例 |
| --- | --- | --- |
| `BASIC_CONTROL` | 基础操控或简单组合 | 保持车道、提速、停车、左转、变道 |
| `COMPLEX_OBSTACLE_AVOIDANCE` | 需要结合目标对象完成多步避障 | 避让行人后变道超车 |
| `EMERGENCY_RESPONSE` | 突发危险或应急动作 | 突发加塞，紧急避让 |
| `NAVIGATION` | 以位置、道路或目的地为目标 | 前方 300 米路口右转 |
| `META_CONTROL` | 取消、恢复等对当前任务的控制 | 取消刚才的变道指令 |

## 6. Step 通用字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `step_id` | string | 是 | 当前指令内唯一，推荐 `step_1`、`step_2` |
| `action` | enum | 是 | 高层动作，见第 7 节 |
| `purpose` | string | 否 | 简短目的，例如 `YIELD`、`OVERTAKE`、`REDUCE_RISK` |
| `target` | object | 否 | 语言中明确提到的目标对象，见第 8 节 |
| `parameters` | object | 是 | 动作参数；没有参数时使用 `{}` |
| `trigger` | object | 是 | 何时开始考虑执行该动作，见第 10 节 |
| `completion` | object | 否 | 动作完成的语义条件，见第 10 节 |
| `depends_on` | string[] | 是 | 当前动作依赖的先前 Step ID；无依赖时为 `[]` |
| `preconditions` | enum[] | 是 | 交给场景/风险模块验证的条件，见第 11 节 |
| `on_blocked` | enum | 是 | 前置条件不满足时的处理策略，见第 11 节 |

`depends_on` 只能引用当前 Step 之前已经出现的 ID，不允许循环依赖。

## 7. `action` 全部取值及参数搭配

| action | 语义 | 必需或推荐内容 | 示例指令 |
| --- | --- | --- | --- |
| `KEEP_LANE` | 保持当前车道 | 通常无参数 | 保持当前车道 |
| `SET_SPEED` | 设置明确速度 | 必须有 `target_speed_mps` | 提速至 60 km/h |
| `ADJUST_SPEED` | 定性或定量调速 | `change`；有明确差值时可用 `speed_delta_mps` | 开慢一点、加速 5 km/h |
| `STOP` | 正常制动直至停车 | 可带目标或距离，推荐完成条件 `VEHICLE_STOPPED` | 前方停车 |
| `WAIT` | 保持安全状态等待 | 必须有 `duration_s` 或条件触发 | 等行人通过后再走 |
| `FOLLOW` | 持续跟随动态目标 | 必须有 `target`；可带 `duration_s`、`following_distance_m` | 跟随蓝色车辆 5 分钟 |
| `APPROACH` | 缩短与目标的距离 | 必须有 `target` | 缓慢接近右侧行人 |
| `NAVIGATE_TO` | 行驶至目标地点或目标所在位置 | 必须有 `target` | 开到银色汽车所在位置 |
| `CHANGE_LANE` | 横向换道 | 必须有左右方向，或 `lane_index` + `lane_reference` | 向左变道 |
| `MERGE` | 汇入目标车道或交通流 | 必须有左右方向，或目标车道序号 | 向右汇入主路 |
| `TURN` | 路口转向 | 必须有 `direction`；通常带路口目标或距离触发 | 前方 300 米路口右转 |
| `U_TURN` | 调转行驶方向 | 推荐在允许掉头的路口执行 | 前方允许处掉头 |
| `PROCEED` | 开始或沿当前路线继续前进 | 可带 `direction=STRAIGHT` | 行人通过后继续直行 |
| `YIELD` | 给对象让行 | 必须有 `target` | 给前方行人让行 |
| `PULL_OVER` | 靠边行驶或停车 | 推荐 `direction` 和目标速度 | 靠边减速到 30 km/h |
| `PARK` | 驶入并占用停车位置 | 推荐停车目标和 `parking_maneuver` | 倒车停入右侧车位 |
| `OVERTAKE` | 超越目标 | 必须有 `target`；方向未明确时由规划器决定 | 超越前方慢车 |
| `PASS_BY` | 从目标旁经过，不表达取得领先 | 必须有 `target` | 慢速经过路边行人 |
| `AVOID` | 绕开目标 | 必须有 `target`；只有用户明确时才填写方向 | 避让施工锥桶 |
| `REVERSE` | 独立倒车动作 | 可带距离和目标 | 向后倒车 2 米 |
| `ENTER_AREA` | 进入明确区域 | 必须有区域目标 | 进入右侧停车场 |
| `EXIT_AREA` | 驶出明确区域 | 必须有区域目标 | 驶出停车场 |
| `EMERGENCY_BRAKE` | 紧急制动 | 目标可选，紧急程度应为 `EMERGENCY` | 前方危险，紧急停车 |
| `RESUME` | 临时动作后恢复先前路线、车道或驾驶状态 | 通常无参数 | 避障后回到原车道 |
| `CANCEL` | 取消待执行用户指令 | 通常无参数；具体取消对象可在 purpose 中说明 | 取消刚才的变道 |

这里的“必须”属于语义校验规则。Schema 负责结构校验，解析器和 `validate_examples.py` 后续还应逐步补全动作级语义校验。

## 8. `target` 目标对象

### 8.1 `target.type`

| 值 | 对应目标 |
| --- | --- |
| `VEHICLE` | 普通车辆或前车 |
| `SLOW_VEHICLE` | 明确描述为慢车的车辆 |
| `PEDESTRIAN` | 行人 |
| `CYCLIST` | 自行车或非机动车参与者 |
| `OBSTACLE` | 未进一步分类的障碍物 |
| `TRAFFIC_CONE` | 施工锥桶 |
| `CONSTRUCTION_ZONE` | 施工区域或施工路段 |
| `TRAFFIC_LIGHT` | 交通信号灯 |
| `TRAFFIC_SIGN` | 交通标志 |
| `CROSSWALK`、`STOP_LINE` | 人行横道和停止线 |
| `JUNCTION` | 路口或交叉口 |
| `LANE` | 语言中的左车道、右车道等语义车道 |
| `ROAD`、`AREA` | 道路和通用区域 |
| `PARKING_AREA`、`PARKING_SPACE`、`CURB` | 停车场、停车位和路缘 |
| `LANDMARK` | 树、建筑等可用于定位的地标 |
| `DESTINATION` | 目的地或停车点 |
| `PICKUP_POINT`、`DROPOFF_POINT` | 明确接客点和下客点 |
| `ROAD_HAZARD` | 路况危险、落物、积水等泛化危险 |
| `COORDINATE` | 用户原话明确给出的坐标位置 |
| `UNKNOWN` | 文本明确存在目标，但无法分类 |

### 8.2 `target.relation`

| 值 | 含义 |
| --- | --- |
| `AHEAD` | 自车前方 |
| `BEHIND` | 自车后方 |
| `LEFT` | 自车或参考目标左侧 |
| `RIGHT` | 自车或参考目标右侧 |
| `FRONT_LEFT`、`FRONT_RIGHT` | 自车前左或前右 |
| `REAR_LEFT`、`REAR_RIGHT` | 自车后左或后右 |
| `AHEAD_CROSSING` | 正在前方横穿 |
| `ADJACENT` | 相邻位置或相邻车道 |
| `NEXT_TO`、`IN_FRONT_OF`、`NEAR` | 在目标旁、目标前或附近 |
| `AT_JUNCTION` | 位于路口 |
| `NEAR_DESTINATION` | 位于目的地附近 |
| `INSIDE`、`PAST` | 区域内部或已通过目标的位置 |
| `UNSPECIFIED` | 原文未给出相对关系 |

`target.description` 保存无法完全枚举的原文短语，例如“公交站旁边穿蓝衣服的人”。它只用于语义对齐，不允许填写 CARLA actor ID。`target.coordinates` 只允许保存用户明确说出的 `x_m`、`y_m` 与 `frame`，禁止由解析器或感知模块猜测。

## 9. `parameters` 全部字段

| 字段 | 类型 | 允许内容 | 使用方式 |
| --- | --- | --- | --- |
| `direction` | enum | `LEFT`、`RIGHT`、`STRAIGHT`、`FORWARD`、`BACKWARD` | 变道、转弯、前进和倒车方向 |
| `change` | enum | `INCREASE`、`DECREASE`、`HOLD` | 定性速度变化 |
| `target_speed_mps` | number | `>=0` | 明确目标速度，单位 m/s |
| `speed_delta_mps` | number | `>0` | 速度变化绝对值，方向由 `change` 表示 |
| `distance_m` | number | `>=0` | 指令明确给出的距离，单位 m |
| `start_distance_m` | number | `>=0` | 距离动作开始点的距离 |
| `transition_distance_m` | number | `>=0` | 换道或横向过渡距离 |
| `following_distance_m` | number | `>=0` | 用户明确要求的跟车距离 |
| `duration_s` | number | `>=0` | 跟随或等待时长，统一为秒 |
| `lane_count` | integer | `1-8` | 需要横跨的车道数 |
| `lane_index` | integer | `1-16` | 从道路一侧起算的自然序号；0、负数和 unknown 必须澄清 |
| `lane_reference` | enum | `LEFT_EDGE`、`RIGHT_EDGE` | `lane_index` 的起算侧 |
| `parking_maneuver` | enum | `FORWARD`、`REVERSE`、`PARALLEL`、`UNSPECIFIED` | 明确的停车方式 |
| `source_value` | number | 任意数值 | 原始指令数值，用于日志追溯 |
| `source_unit` | enum | `km/h`、`m/s`、`m`、`s`、`min` | 原始数值单位 |

运行字段必须使用 SI 单位。`source_value` 和 `source_unit` 只能作为追溯信息，规划器应读取换算后的 `target_speed_mps`、`speed_delta_mps` 或 `distance_m`。

## 10. 触发和完成条件

### 10.1 `trigger.type`

| 值 | 含义 | 搭配字段 |
| --- | --- | --- |
| `IMMEDIATE` | 立即进入该动作 | 无 |
| `AFTER_STEP` | 指定前序动作后执行 | 必须有 `step_id` |
| `AT_DISTANCE` | 到达指定距离点时执行 | 必须有 `distance_m` |
| `AT_JUNCTION` | 到达路口时执行 | 可带目标路口 |
| `OBJECT_PRESENT` | 对应目标被场景模块确认时执行 | Step 应有 `target` |
| `WHEN_SAFE` | 风险模块判定安全后执行 | 应有相应 `preconditions` |
| `CONDITION` | 其他可机器判断的语义条件 | 必须有 `description`，后续需映射为规则 |

### 10.2 `completion.type`

| 值 | 含义 |
| --- | --- |
| `ACTION_REACHED` | 通用动作目标已达到 |
| `TARGET_CLEARED` | 已避让、超过或通过目标 |
| `TARGET_SPEED_REACHED` | 已达到目标速度 |
| `LANE_CHANGE_COMPLETED` | 已稳定进入目标车道 |
| `JUNCTION_EXITED` | 已完成路口动作并驶出路口 |
| `VEHICLE_STOPPED` | 车辆已停止 |
| `WAIT_CONDITION_MET`、`DURATION_ELAPSED` | 等待条件已满足或时长结束 |
| `FOLLOWING_ESTABLISHED`、`TARGET_REACHED` | 已建立跟随或到达目标 |
| `AREA_ENTERED`、`AREA_EXITED` | 已进入或驶出区域 |
| `PARKING_COMPLETED` | 已完成停车 |

`trigger.description` 只能描述后续模块能够实现的条件。禁止将任意自然语言直接作为可执行代码或表达式。

## 11. 前置条件和阻塞策略

### 11.1 `preconditions` 全部取值

| 值 | 验证模块 | 含义 |
| --- | --- | --- |
| `LEFT_LANE_EXISTS` | CARLA/地图 | 左侧车道存在 |
| `RIGHT_LANE_EXISTS` | CARLA/地图 | 右侧车道存在 |
| `LEFT_LANE_SAFE` | 风险判断 | 左车道当前允许安全进入 |
| `RIGHT_LANE_SAFE` | 风险判断 | 右车道当前允许安全进入 |
| `TARGET_LANE_SAFE` | 风险判断 | 已选目标车道安全 |
| `LANE_CHANGE_LEGAL` | 地图/规则 | 当前路段允许变道 |
| `JUNCTION_REACHED` | CARLA/地图 | 已到达目标路口 |
| `TARGET_VISIBLE` | 感知/语义对齐 | 语言目标已匹配到场景对象 |
| `PATH_CLEAR` | 风险判断 | 预计行驶路径无阻挡 |
| `TARGET_REACHABLE` | 地图/语义对齐 | 目标可由当前道路网络到达 |
| `AREA_ACCESSIBLE` | 地图/规则 | 目标区域允许驶入或驶出 |
| `PARKING_SPACE_AVAILABLE` | 感知/地图 | 停车位置存在且可用 |

### 11.2 `on_blocked` 全部取值

| 值 | 含义 |
| --- | --- |
| `WAIT_FOR_SAFE` | 保持安全状态并等待条件满足 |
| `SAFE_STOP` | 无法继续时安全停车 |
| `REQUEST_CLARIFICATION` | 请求用户补充目标、方向或其他信息 |
| `ABORT_COMMAND` | 终止当前用户指令 |
| `SKIP_STEP` | 跳过当前动作并继续后续动作，仅用于明确允许跳过的步骤 |

安全相关动作默认使用 `WAIT_FOR_SAFE` 或 `SAFE_STOP`。不得为了提高任务完成率滥用 `SKIP_STEP`。

## 12. `constraints` 全局约束

| 字段 | 类型 | 必填 | 允许内容 | 说明 |
| --- | --- | --- | --- | --- |
| `safety_first` | boolean | 是 | 固定 `true` | 安全策略优先于用户动作 |
| `obey_traffic_rules` | boolean | 是 | 固定 `true` | 禁止通过指令关闭交通规则 |
| `driving_style` | enum | 是 | `NORMAL`、`CONSERVATIVE` | “稳一点”“保持安全车速”使用 CONSERVATIVE |
| `max_speed_mps` | number | 否 | `>=0` | 用户明确提出的最高速度，不替代道路限速 |

## 13. `parse_result` 解析结果

| 字段 | 类型 | 必填 | 允许内容 | 说明 |
| --- | --- | --- | --- | --- |
| `status` | enum | 是 | `VALID`、`NEEDS_CLARIFICATION`、`UNSUPPORTED`、`INVALID` | 当前解析状态 |
| `method` | enum | 是 | `RULE`、`LLM`、`HYBRID` | 生成结果的方法 |
| `model` | string/null | 是 | 模型名称或 null | 纯规则解析使用 null |
| `confidence` | number | 是 | `0-1` | 由解析管线计算，不能使用模型自述值 |
| `missing_slots` | string[] | 是 | 字段路径列表 | 缺失但影响执行的字段 |
| `warnings` | string[] | 是 | 警告文本 | 记录单位假设、ASR 疑似错误等 |
| `clarification_question` | string | 否 | 非空文本 | NEEDS_CLARIFICATION 时推荐提供 |
| `latency_ms` | number | 是 | `>=0` | 仅统计本模块处理延时 |

状态使用约定：

| 状态 | `steps` | 后续处理 |
| --- | --- | --- |
| `VALID` | 至少一个 | 进入语义对齐和风险判断 |
| `NEEDS_CLARIFICATION` | 可以为空 | 暂停执行并询问用户 |
| `UNSUPPORTED` | 通常为空 | 告知不支持，不进入车辆控制 |
| `INVALID` | 通常为空 | 记录错误，不进入车辆控制 |

## 14. 典型指令映射

| 用户指令 | category | steps 摘要 |
| --- | --- | --- |
| 保持当前车道 | `BASIC_CONTROL` | `KEEP_LANE` |
| 提速至 60 km/h | `BASIC_CONTROL` | `SET_SPEED(16.667 m/s)` |
| 前方 300 米路口右转 | `NAVIGATION` | `TURN(RIGHT)` + `AT_DISTANCE(300 m)` |
| 向左变道 | `BASIC_CONTROL` | `CHANGE_LANE(LEFT)` + 左车道安全条件 |
| 跟随蓝色车辆 5 分钟 | `NAVIGATION` | `FOLLOW(VEHICLE, duration_s=300)` |
| 缓慢接近右侧行人 | `NAVIGATION` | `ADJUST_SPEED(DECREASE)` -> `APPROACH(PEDESTRIAN)` |
| 等行人通过后继续直行 | `COMPLEX_OBSTACLE_AVOIDANCE` | `WAIT(CONDITION)` -> `PROCEED(STRAIGHT)` |
| 前方掉头后进入右侧停车场 | `NAVIGATION` | `U_TURN` -> `ENTER_AREA(PARKING_AREA)` |
| 倒车停入右侧第一个车位接人 | `NAVIGATION` | `PARK(parking_maneuver=REVERSE, purpose=PICK_UP)` |
| 前方公交站有行人上下车，靠边减速至 30 km/h，确认安全后继续行驶 | `COMPLEX_OBSTACLE_AVOIDANCE` | `PULL_OVER` -> `SET_SPEED` -> `RESUME` |
| 避让施工锥桶并回归原车道 | `COMPLEX_OBSTACLE_AVOIDANCE` | `AVOID` -> `CHANGE_LANE` -> `RESUME` |
| 突发车辆加塞，紧急避让 | `EMERGENCY_RESPONSE` | `AVOID` 或 `EMERGENCY_BRAKE`，urgency 为 EMERGENCY |
| 前方路况危险，保持安全车速 | `EMERGENCY_RESPONSE` | `ADJUST_SPEED(DECREASE/HOLD)` + CONSERVATIVE |
| 从那边绕过去 | `NAVIGATION` | `NEEDS_CLARIFICATION`，目标与方向缺失 |

应急指令中究竟采用 `AVOID` 还是 `EMERGENCY_BRAKE`，只能根据用户明确语义解析；实际风险出现时的系统强制制动属于风险/决策模块，不需要用户先说出该动作。

## 15. 禁止写入 DrivingIntent 的内容

| 禁止字段或内容 | 应由哪个模块产生 |
| --- | --- |
| CARLA `actor_id`、感知推断坐标、边界框 | CARLA 感知与语义对齐；仅用户明确说出的坐标可进入 `target.coordinates` |
| 实际 `lane_id`、道路 ID、地图 waypoint | CARLA 地图与规划 |
| TTC、安全距离、相对速度、风险等级 | 风险判断 |
| 最终是否允许执行、决策原因 | 决策规划 |
| FSM 当前状态和状态迁移结果 | 有限状态机 |
| 油门、刹车、方向盘数值 | 车辆控制 |
| 图像、点云、音频二进制内容 | 数据和传感器模块，只能传引用 |
| Prompt、Token、模型内部推理过程 | 指令解析内部日志，不进入公共接口 |

## 16. 下游模块读取建议

| 模块 | 应读取的字段 |
| --- | --- |
| 语义对齐 | `target`、`action`、`trigger`、`input.normalized_text` |
| 风险判断 | `action`、`target`、`preconditions`、`constraints` |
| 有限状态机 | `steps`、`depends_on`、`trigger`、`completion`、`on_blocked` |
| 车辆控制 | 不直接读取 DrivingIntent；只读取决策规划输出 |
| 日志评估 | `request_id`、`parse_result`、动作序列及各模块关联结果 |

## 17. 开发检查清单

1. 输出能通过 JSON Schema 校验。
2. `VALID` 输出至少有一个 Step。
3. `step_id` 唯一，依赖只引用此前动作。
4. 数值已经转换为 SI 单位。
5. 动作顺序与用户原话一致。
6. 模糊目标和方向没有被模型凭空补全。
7. 安全和交通规则字段始终为 true。
8. 意图中不含风险结果、CARLA ID 或控制量。
9. 组合指令没有动作遗漏。
10. 日志能够用 `request_id` 串联完整链路。
