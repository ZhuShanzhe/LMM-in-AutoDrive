# 提交模型目录

模型权重与 Docker 镜像单独提交，不写入 Git。运行脚本应从仓库根目录的相对路径
`models/` 查找权重，也允许用 `MODEL_ROOT` 指向容器挂载目录。

```text
models/
├── modernbert-drive-command-compositional/
├── lightweight_vla_adapter/scene3_multimodal_v3/
│   ├── model.pt
│   ├── config.json
│   ├── model_manifest.json
│   └── model.sha256
└── scene_understanding/yolo11s_driving_v2/weights/best.pt
```

Linux 运行时示例：

```bash
REPO_ROOT=$(pwd)
MODEL_ROOT=${MODEL_ROOT:-"$REPO_ROOT/models"}

python experiment/CARLA/run_emergency_response_6km.py \
  --ego-controller vla-route-pid \
  --vla-checkpoint "$MODEL_ROOT/lightweight_vla_adapter/scene3_multimodal_v3/model.pt" \
  --vla-config lightweight_vla_adapter/configs/scene3_multimodal_v3.json \
  --vla-parser-model "$MODEL_ROOT/modernbert-drive-command-compositional"
```

提交前在 `models/` 内生成并校验完整哈希：

```bash
find models -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > models/SHA256SUMS
sha256sum -c models/SHA256SUMS
```

最终 Docker 镜像只需包含代码与依赖；权重可随镜像交付，也可只读挂载到同一目录结构。
不得在配置或清单中保留 `/root/autodl-tmp` 等训练服务器绝对路径。
