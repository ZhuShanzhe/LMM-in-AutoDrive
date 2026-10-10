import copy
import torch
import pytest

from lightweight_vla_adapter.src.bev_encoder import LightweightBEVEncoder
from lightweight_vla_adapter.src.sensor_specialization import SensorSpecializedBEVEncoder


def case(batch=1, size=8):
    torch.manual_seed(1008)
    encoder = LightweightBEVEncoder(8, 4, hidden_size=32, output_grid=(2, 2)).eval()
    optimized = SensorSpecializedBEVEncoder(copy.deepcopy(encoder)).eval()
    camera = torch.zeros(batch, 8, size, size)
    lidar = torch.zeros(batch, 4, size, size)
    return encoder, optimized, camera, lidar


@pytest.mark.parametrize("batch,size", [(1, 8), (2, 8), (1, 12)])
@pytest.mark.parametrize("kind", ["zero_both", "zero_camera", "full"])
def test_parity(batch, size, kind):
    source, optimized, camera, lidar = case(batch, size)
    if kind != "zero_both":
        lidar.normal_()
    if kind == "full":
        camera.normal_()
    with torch.inference_mode():
        for _ in range(3):
            torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)
    assert optimized.calls[kind] == 3


def test_change_from_zero_to_observed_is_not_cached():
    source, optimized, camera, lidar = case()
    with torch.inference_mode():
        optimized(camera, lidar)
        lidar.normal_()
        torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)
        camera.normal_()
        torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)


def test_cached_tokens_are_not_mutable_by_caller():
    source, optimized, camera, lidar = case()
    with torch.inference_mode():
        optimized(camera, lidar).fill_(42)
        torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)


def test_weight_update_invalidates_cache():
    source, optimized, camera, lidar = case()
    with torch.no_grad():
        optimized(camera, lidar)
        source.camera_stem[1].bias.add_(.3)
        optimized.encoder.camera_stem[1].bias.add_(.3)
        torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)


def test_buffer_update_invalidates_cache():
    source, _, camera, lidar = case()
    # Production uses GroupNorm; this fixture checks supported stateful stems.
    source.camera_stem[1] = torch.nn.BatchNorm2d(source.camera_stem[0].out_channels).eval()
    optimized = SensorSpecializedBEVEncoder(copy.deepcopy(source)).eval()
    with torch.no_grad():
        optimized(camera, lidar)
        source.camera_stem[1].running_mean.add_(.3)
        optimized.encoder.camera_stem[1].running_mean.add_(.3)
        torch.testing.assert_close(optimized(camera, lidar), source(camera, lidar), rtol=0, atol=0)


def test_dtype_move_invalidates_cache():
    source, optimized, camera, lidar = case()
    with torch.inference_mode():
        optimized(camera, lidar)
        source.double(); optimized.double()
        torch.testing.assert_close(optimized(camera.double(), lidar.double()),
                                  source(camera.double(), lidar.double()), rtol=0, atol=0)


def test_gradient_enabled_retains_original_graph():
    source, optimized, camera, lidar = case()
    camera.requires_grad_()
    value = optimized(camera, lidar).sum()
    value.backward()
    assert camera.grad is not None
    assert not optimized._cache


def test_training_never_caches():
    source, optimized, camera, lidar = case()
    optimized.train()
    with torch.no_grad():
        optimized(camera, lidar)
    assert not optimized._cache


@pytest.mark.parametrize("bad", ["rank", "batch"])
def test_contract_errors(bad):
    _, optimized, camera, lidar = case()
    if bad == "rank":
        camera = camera[0]
    else:
        lidar = lidar.repeat(2, 1, 1, 1)
    with torch.inference_mode(), pytest.raises(ValueError):
        optimized(camera, lidar)
