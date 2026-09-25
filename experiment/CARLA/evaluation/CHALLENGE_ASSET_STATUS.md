# 本地挑战模型资产核对

核对版本：signal-generalization-v2-phase-recovery-v1。
依据：lightweight_vla_adapter/configs/challenge_assets.json。
核对仓库models目录、D:/CARLA/models及显式提供的modernbert-drive-command-base目录。
逐文件同时比较大小和SHA256，23项中19项一致，4项尚未满足。

| 资产 | 结果 | 清单要求SHA256 |
| --- | --- | --- |
| challenge/lamp_state_v2.pt | 两个模型根目录均缺失 | d8470bf6a176037ba74e2a28f790ebb5b5ab8abe73625b37960e7f6128b4b370 |
| challenge/phase_recovery_v1.pt | 两个模型根目录均缺失 | 5bd639ccb0c09326742c9d5c76a9ddaade7c983d51ce25e0f9d762d64af56fb3 |
| lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt | 两个模型根目录均缺失 | 06c774e3a5eead95e230b55b65a3d86c52ba93110c6046506be21dd69ecb2165 |
| modernbert-drive-command-compositional/README.md | 本地旧名称目录中的文件与清单不一致 | 以JSON清单为准 |

解析模型其余19项在D:/CARLA/models/modernbert-drive-command-base内校验一致，
目录名称不能用于推断其权重版本。README差异本身不证明数值推理有误，但不符合团队
完整分发清单，不能直接声明23项全部一致。原文件不覆盖、不改名来掩盖差异。

本地lightweight_vla_adapter/v10/model.pt不是指定的V6基础权重，不作为替代。
仓库models/README.md注明新版文件未发布到Hugging Face，需从团队获准访问的共享位置取得。
目前仅完成文件核对，不曾加载新版模型、评估模型能力或执行完整闭环。

完整机器可读结果：D:/CARLA/outputs/challenge_asset_audit_20260920.json。
evaluation.challenge_asset_audit支持多个--model-root及显式--parser-dir，一次报告所有
缺失/不匹配项，不会修改模型、下载文件或静默放宽团队哈希清单。
