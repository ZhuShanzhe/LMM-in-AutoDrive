import pytest
import torch
from lightweight_vla_adapter.src.executed_sequence_io import store_rgb


def test_carla_bytes_are_preserved_losslessly():
    raw=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16).repeat(4,3,1,1)
    assert torch.equal(store_rgb(raw),raw)
    assert store_rgb(raw).unique().numel()==256


def test_normalized_floats_roundtrip_to_the_same_pixels():
    raw=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16).repeat(4,3,1,1)
    assert torch.equal(store_rgb(raw.float()/255),raw)


def test_ambiguous_or_nonfinite_inputs_are_not_silently_clipped():
    with pytest.raises(ValueError):store_rgb(torch.ones(4,3,16,16)*255.)
    with pytest.raises(ValueError):store_rgb(torch.ones(4,3,16,16)*float('nan'))
