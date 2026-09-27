# 李畅锦：文本入口模式、指令语义契约与输入回归

本目录用于定位和修改，不表示问题已修复。代码在仓库原模块中修改；source_snapshot仅是带行号的取证副本。

## 快速开始

从仓库根目录运行：

```bash
python issue_handoff_20260927/check_package.py --owner 李畅锦
PYTHONPATH=.:experiment/CARLA python -m pytest -q experiment/CARLA/tests/test_scene2_first_task_contract.py
```

第一条只校验证据并展示问题基线，不将已知问题算作通过。第二条是已有回归测试；它们通过不意味着本目录的新验收条件已满足。

共用输入位于 `../_common/`：三场景摘录、环境、完整场景二路线/地图、配置与额外采集。传感器压缩包解压到任意新空目录后可用原有ModelRigReplayDataset读取。

## 问题与任务

### L1 [P0] 配置计划与真实文本解析分开标记

**证据：** 场景二通过build_scheduled_driving_intent直接提供competition_schedule计划，GenericInstructionFSM优先返回该计划；本轮不是ModernBERT解析准确率测试。

**修改任务：** 梳理配置计划/文本解析两类入口与日志字段；保持执行隔离测试可用，同时为文本输入建立独立回归入口，记录实际来源，禁止静默回退为预置答案。

**验收：** 传入文本模式时实际调用解析器；配置模式明确标记；拒绝/澄清正确透传；不能把预置步骤作为模型预测计分。

### L2 [P1] 速度、条件、顺序和指代测试矩阵

**证据：** 共享目录提供15条真实场景二指令及对应配置计划；这些是配置契约，不是独立人工金标准。

**修改任务：** 核验速度单位和目标字段、否定、先后顺序、回原车道、等待条件以及歧义处理；添加同义改写/数字变体对照，人工核对期望，再交朱善哲定位确实存在的解析偏差。

**验收：** 每个用例含原文、期望语义、实际输出及来源；语音未接入不声称测过ASR；禁止用语义不同的句子共用答案。

## 修改边界

精确文件、函数及行号见 `code_locations.json`；依赖另一位同学的接口先对齐字段，不同时改同一个文件。输入缺少真值的现象不得直接定性为感知误检。

提交时在本目录补充修改说明和测试结果，实际代码改原模块；保留原始证据，不覆盖旧日志。

## 本人输入包

`inputs/`已按本人问题保存精简输入与实际输出，`evidence_index.json`列出共享证据位置。完整日志片段以`.json.gz`压缩保存，可用Python的gzip与json读取。

通用接口探针（不启动CARLA、不加载权重）：

```bash
python issue_handoff_20260927/check_package.py --probe-contracts
```

传感器原始输入解码与字节完整性检查：

```bash
python issue_handoff_20260927/check_package.py --decode-capture
```
