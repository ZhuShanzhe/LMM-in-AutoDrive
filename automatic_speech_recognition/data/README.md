# ASR 数据与评测输出

本目录在 `main` 中只保留说明文件。以下内容由数据准备、语音生成和评测脚本在本地数据盘生成，不提交 Git：

```text
commands.json
translated_commands.json
wav_files/
wav_files_noise/
wav_files_accent/
logging/
test_results/
test_results_noise/
test_result_denoising/
test_result_accent/
```

推荐外部路径：

```text
/root/autodl-tmp/datasets/asr/
/root/autodl-tmp/outputs/asr/
```

第一阶段最终材料仍需从组员实验分支或原始实验目录整理：

- 方言、噪声和去噪测试集来源与规模；
- WER/CER、准确率和延时汇总；
- 代表性成功与失败样本；
- 测试配置、模型版本和硬件环境；
- 可复现的小规模测试用例。

已提交过的原始 JSON 仍保留在 Git 历史和组员分支中，本次 `main` 清理不删除实验来源。
