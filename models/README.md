# 挑战运行权重

Git 不含权重、缓存或数据集。固定清单为 `lightweight_vla_adapter/configs/challenge_assets.json`。当前新权重没有上传 Hugging Face，不能用旧 HF 候选文件替代。所有模型均沿用相应许可和使用范围，不因此获得新的商业或道路部署授权。

## 服务器

当前基准的完整权重已在联合工作区准备好：

```text
/root/autodl-tmp/worktrees/challenge-track/models/
  challenge/phase_recovery_v1.pt
  challenge/lamp_state_v2.pt
  modernbert-drive-command-compositional/
  lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt
```

其中旧解析模型与 V6 模型复用服务器只读共享目录，新权重独立复制，不依赖 `zsz/outputs/`。

## 其他机器

在新检出的仓库根目录执行，`PORT`、`USER`、`HOST` 使用本人获准的服务器连接信息，不在仓库保存密码：

```bash
mkdir -p models/challenge models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy
scp -P PORT USER@HOST:/root/autodl-tmp/worktrees/challenge-track/models/challenge/phase_recovery_v1.pt models/challenge/
scp -P PORT USER@HOST:/root/autodl-tmp/worktrees/challenge-track/models/challenge/lamp_state_v2.pt models/challenge/
scp -P PORT USER@HOST:/root/autodl-tmp/models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt models/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/
scp -r -P PORT USER@HOST:/root/autodl-tmp/models/modernbert-drive-command-compositional models/
python lightweight_vla_adapter/scripts/prepare_challenge_runtime.py \
  --map Town04 --output lightweight_vla_adapter/outputs/runtime/Town04.json
```

无服务器权限的组员由负责人按同一清单分发文件；仅拉 Git 不能运行带权重的新版模型。独立场景理解/ASR 模块若另需模型，按各模块说明获取。

## 主要哈希

| 文件 | SHA256 |
|---|---|
| `challenge/phase_recovery_v1.pt` | `5bd639ccb0c09326742c9d5c76a9ddaade7c983d51ce25e0f9d762d64af56fb3` |
| `challenge/lamp_state_v2.pt` | `d8470bf6a176037ba74e2a28f790ebb5b5ab8abe73625b37960e7f6128b4b370` |
| `lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt` | `06c774e3a5eead95e230b55b65a3d86c52ba93110c6046506be21dd69ecb2165` |

完整解析模型及许可/配置文件哈希由 JSON 清单校验。哈希不一致时停止加载，不能静默选用其他训练轮次。
