from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError:
        result["torch"] = None
        result["cuda_available"] = False
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    result["torch"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    result["device"] = torch.cuda.get_device_name(0)
    result["compute_capability"] = list(torch.cuda.get_device_capability(0))
    tensor = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    product = tensor @ tensor
    torch.cuda.synchronize()
    result["bf16_matmul_shape"] = list(product.shape)
    result["bf16_matmul_finite"] = bool(torch.isfinite(product).all().item())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["bf16_matmul_finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
