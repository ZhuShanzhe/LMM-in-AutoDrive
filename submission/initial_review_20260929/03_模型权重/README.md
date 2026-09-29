# 权重与许可

现有入口：[固定权重说明](../../../models/README.md)、[冻结资产清单](../../../lightweight_vla_adapter/configs/challenge_assets.json)、[下载来源](../../../lightweight_vla_adapter/configs/challenge_asset_sources.json)。

语言/VLA 资产使用固定 Hugging Face revision 和 SHA256 校验；Git 只存清单，不提交大权重。独立 ASR、翻译与其他实际启用模块不全部包含在这份清单内，须逐一补入最终权重目录、大小、校验值及许可。造数据专用 TTS 不应误列为实时链路必需项。正式包必须能离线加载，不能只给下载链接。
