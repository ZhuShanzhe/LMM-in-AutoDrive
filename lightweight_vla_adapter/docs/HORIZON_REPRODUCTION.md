# Horizon 混合量化验证

## 已验证的范围

在独立 OpenExplorer 3.9.1、Python 3.10 环境中完成当前通用策略权重的固定尺寸导出、PTQ、带标签对照及 `nash-p` 编译。原始 RGB 卷积、注意力乘积、归一化和输出头保留浮点；BEV 卷积、投影与 FFN 使用 INT8。默认 CARLA 链路仍使用已经闭环测试的浮点权重。

这个导出图不是整个驾驶系统：不包含 ModernBERT、独立实体感知、点云道路几何、递归事件状态管理、安全规则或车辆控制。编译时延只对应导出的神经图，不能当作全链路时延。

## 环境与输入

已验证环境：HBDK 4.11.11、HMCT 2.8.4、Horizon TC UI 3.5.16、PyTorch 2.8 CPU、ONNX 1.15、ONNX Runtime 1.19。SDK 只用于独立 x86 功能验证，不修改 CARLA 的 Python 3.12 运行环境。

导出需要当前模型权重、对应模型配置、真实模型输入 NPZ 和有效的 SDK 编译模板。模板必须给出与导出图一致的校准数据目录、输入顺序、形状和预处理。当前已验证样本为 32 个语言 token、224 像素图像；运行时另一个 token 长度的样本不能直接混入这套校准输入。输入名称与具体尺寸保存在 `export_manifest.json`，不能只凭文件名判断。

## 运行命令

在仓库根目录和已配置的 SDK 环境运行。以下变量需要指向自己的文件和新输出目录，脚本拒绝覆盖已有导出结果。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python lightweight_vla_adapter/scripts/export_quantization_candidate.py \
  --checkpoint "$CHECKPOINT" --config "$MODEL_CONFIG" \
  --input "$INPUT_NPZ" --compile-template "$SDK_TEMPLATE" \
  --scope bev_and_projections --output "$PTQ_OUTPUT"

hb_compile -c "$PTQ_OUTPUT/compile.yaml" --skip compile

python lightweight_vla_adapter/scripts/evaluate_quantization_candidate.py \
  --float-onnx "$PTQ_OUTPUT/policy.onnx" \
  --bc "$PTQ_OUTPUT/compiled/policy_mixed_int8_quantized_model.bc" \
  --validation "$VALIDATION_DIR" --limit 224 \
  --output "$PTQ_OUTPUT/paired_validation.json"

python lightweight_vla_adapter/scripts/compile_horizon_candidate.py \
  --bc "$PTQ_OUTPUT/compiled/policy_mixed_int8_quantized_model.bc" \
  --output "$PTQ_OUTPUT/scheduled" --march nash-p \
  --opt 2 --balance 2 --jobs 1
```

`input_*.npz` 是模型输入；同名 `labels_*.npz` 提供动作、风险、车道方向及速度标签。无标签样本只参与浮点/量化数值与预测一致性比较，不计入准确率。标签必须独立于待测模型输出，开发标签不能冒充赛事规定的三场景同源独立标注集。

## 本轮复现结果

当前权重 SHA256 为 `06c774e3a5eead95e230b55b65a3d86c52ba93110c6046506be21dd69ecb2165`。224 组输入中 128 组有开发标签：动作准确率 67.1875% 对 67.1875%，风险准确率 53.90625% 对 54.6875%，车道方向准确率 93.75% 对 93.75%，高风险召回率 54.545% 对 56.818%；速度 MAE 为 8.690967 对 8.696568 km/h。以上左侧为同权重 FP32，右侧为混合 INT8。

动作和车道方向配对一致率均为 100%，风险一致率为 99.1071%。一致率与准确率分别报告；这组开发数据未证明模型的绝对风险精度达到赛事目标。

本轮 `.bc` SHA256 为 `0ac8b3b97197edbac746d2ed47f7876d13df950504ff5fb5b96653c9515fddb3`，优化 `.hbm` SHA256 为 `1977e879fbb0f7e633a38d2d8f9b0a525a0a1e1062011803fede4eac387ebfca`。编译器预测单图 9.3032 ms、每次 DDR 访问 100,649,472 B、最小内存 57,040,184 B，文件大小 25,161,016 B。最大区间计算利用率预测 62.1%，不是整个异构平台的平均利用率。

没有实际 J6P 板端时延或利用率结果，也没有量化分支的完整 CARLA 闭环通过结果。`.bc` 的 x86 功能推理用于数值验证，其运行耗时不能当作 BPU 时延；导出和编译成功也不能单独证明轨迹安全。
