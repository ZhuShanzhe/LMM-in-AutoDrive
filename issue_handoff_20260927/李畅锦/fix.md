# 李畅锦 修改说明与测试结果

范围：文本入口模式、指令语义契约与输入回归（L1 / L2）。
代码在原模块修改，未改动 `source_snapshot/` 取证副本，旧日志未覆盖。

## 一、改动清单（原模块，已落盘）

### L1 [P0] 配置计划与真实文本解析分开标记

1. `experiment/CARLA/scene2_runtime_interface.py`
   - 新增常量 `PARSE_SOURCE_CONFIGURED = "competition_schedule"`、`PARSE_SOURCE_TEXT_MODEL = "structured_command_parser"`。
   - `build_scheduled_driving_intent` 的 `parse_result` 增加
     `source_kind: "CONFIGURED_PLAN"`、`model_prediction: false`，
     使配置计划的来源可机读、且不会被当作模型预测计分。

2. `experiment/CARLA/control/generic_instruction_fsm.py`
   - `ParsedInstruction` 新增字段 `parse_status: str = "VALID"`、`parse_source: str | None = None`（带默认值，向后兼容）。
   - `_merge_parser_result`：非 `VALID` 状态不再被静默丢弃，改为写入 `parse_status` / `parse_source`（不臆造动作，意图保持 `KEEP_LANE`）；`VALID` 分支同样回填来源。
   - `driving_intent`：文本解析产生的多步文档标记
     `source_kind: "TEXT_MODEL_PARSE"`、`model_prediction: true`、`source` 默认 `structured_command_parser`。

3. `experiment/CARLA/run_complex_avoidance_town05.py`
   - `overlay_payload` 的 HUD 日志新增
     `parse_mode: "CONFIGURED_PLAN"`、`intent_source: "competition_schedule"`、`model_prediction: false`。

### L2 [P1] 速度/条件/顺序/指代测试矩阵

新增测试文件（不改动被约束的 `scene2_runtime_interface.py` 返回车道段，该段由王皓然负责）：

- `experiment/CARLA/tests/test_text_entry_source_regression.py`
- `experiment/CARLA/tests/test_command_semantics_matrix.py`

## 二、回归入口

- L1：`experiment/CARLA/tests/test_text_entry_source_regression.py`
  断言文本模式确实调用解析器并记录真实来源；配置模式显式标记且仍可执行（执行隔离保持可用）；拒绝/澄清透传不入预置答案；预置计划 `model_prediction == false`。
- L2：`experiment/CARLA/tests/test_command_semantics_matrix.py`
  覆盖速度单位/目标字段、数字变体（mps→kmh）、否定、回原车道、等待条件、右转同义改写、缺失方向歧义；每例含原文/期望/实际/来源。

## 三、测试结果（本机实测）

运行命令：

```bash
PYTHONPATH=.:experiment/CARLA python -m pytest -q \
  experiment/CARLA/tests/test_text_entry_source_regression.py \
  experiment/CARLA/tests/test_command_semantics_matrix.py
```

实测输出：

```bash
test_text_entry_source_regression.py ......          [100%]  -> 6 passed
test_command_semantics_matrix.py    .........x        [100%]  -> 9 passed, 1 xfailed
```

- L1：6 passed。
- L2：9 passed，1 xfailed。

## 四、发现的解析偏差（交 朱善哲 定位）

- 用例 `test_compound_order_reference_is_not_collapsed`（标记 `xfail`）：
  规则解析器 `_parse_text_rules` 对复合顺序/指代句（如「先右转，然后直行通过路口，再次右转」）
  只取首个关键词，未保留完整顺序与指代。
  期望语义：`("TURN_RIGHT", "PROCEED_STRAIGHT", "TURN_RIGHT")`。
  状态：`xfail`，待朱善哲定位并给出修复方向；本目录不作偏差定性结论。

## 五、边界与声明

- 语音未接入，未对 ASR 作任何覆盖或声称。
- 语义不同的句子未共用同一期望答案。
- 输入缺少真值的现象不在此定性为感知误检。
- 配置契约（15 条 `configured_commands.json`）只作配置计划，不作为独立人工金标准或模型预测计分。