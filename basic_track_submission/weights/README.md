# 外置权重目录

如果模型权重已经包含在 `image.tar` 中，正式 ZIP 不需要此目录。

如果权重未打进镜像，请保持以下相对结构：

```text
weights/
├── lightweight_vla_adapter/
│   └── universal_three_scene_v6_sensor_policy_finetuned_stage8/
│       └── model.pt
├── modernbert-drive-command-compositional/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
└── SHA256SUMS
```

生成校验文件：

```bash
find weights -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > weights/SHA256SUMS
sha256sum -c weights/SHA256SUMS
```
