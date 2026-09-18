"""Weight-preserving inference transforms; never use on a training model."""

from __future__ import annotations

import copy

from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fuse_inference_conv_bn(model: nn.Module) -> tuple[nn.Module, int]:
    """Return an independent eval model with adjacent Conv2d/BN pairs folded.

    Folding happens in FP32 before deployment dtype conversion. Checkpoints
    keep their original format and are loaded before this transform.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Conv/BN folding requires every module in eval mode")
    result = copy.deepcopy(model).float()

    def fold(module: nn.Module) -> int:
        count = sum(fold(child) for child in module.children())
        if isinstance(module, nn.Sequential):
            names = list(module._modules)
            for first, second in zip(names, names[1:]):
                conv, norm = module._modules[first], module._modules[second]
                if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                    if not norm.track_running_stats:
                        continue
                    module._modules[first] = fuse_conv_bn_eval(conv, norm)
                    module._modules[second] = nn.Identity()
                    count += 1
        return count

    count = fold(result)
    return result, count
