import torch
import pytest
from torch import nn

from lightweight_vla_adapter.src.inference_optimization import fuse_inference_conv_bn


def test_folding_preserves_output_and_source_state():
    torch.manual_seed(17)
    model = nn.Sequential(
        nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU()),
        nn.Conv2d(8, 4, 1, bias=False), nn.BatchNorm2d(4),
    ).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fused, count = fuse_inference_conv_bn(model)
    inputs = torch.randn(2, 3, 16, 16)
    assert count == 2
    torch.testing.assert_close(fused(inputs), model(inputs), rtol=1e-5, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert isinstance(model[1], nn.Conv2d)
    assert isinstance(model[2], nn.BatchNorm2d)


def test_reject_training_model_and_skip_non_running_statistics():
    with pytest.raises(ValueError, match="eval"):
        fuse_inference_conv_bn(nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)))
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4, track_running_stats=False)
    ).eval()
    _, count = fuse_inference_conv_bn(model)
    assert count == 0


def test_pipeline_loads_original_checkpoint_before_optional_folding(tmp_path):
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline

    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)).eval()
    path = tmp_path / "original.pt"
    torch.save(model.state_dict(), path)
    loaded = LightweightVLAPipeline.from_checkpoint(
        nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)),
        str(path), device="cpu", fuse_conv_bn=True,
    )
    inputs = torch.randn(1, 3, 8, 8)
    assert isinstance(loaded.model[1], nn.Identity)
    torch.testing.assert_close(loaded.model(inputs), model(inputs))
    with pytest.raises(ValueError, match="requires FP32"):
        LightweightVLAPipeline.from_checkpoint(
            model, str(path), device="cpu", dtype=torch.float16, fuse_conv_bn=True
        )


def test_pipeline_rejects_tf32_before_loading_weights(monkeypatch):
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with pytest.raises(ValueError, match="disabling"):
        LightweightVLAPipeline.from_checkpoint(
            nn.Identity(), "does-not-exist.pt", device="cuda",
            dtype=torch.float32, fuse_conv_bn=True,
        )
