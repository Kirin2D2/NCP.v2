"""LRP through strided convolutions and pools keeps the input shape and conserves relevance."""

import pytest
import torch
import torch.nn as nn

from ncp.lrp import MEAN, STD, lrp


def _forward(module, x):
    with torch.no_grad():
        out = module(x)
    module.input, module.output = x, out
    return out


def _positive_like(t):
    return torch.rand(t.shape, generator=torch.Generator().manual_seed(0)) + 0.1


@pytest.mark.parametrize("size", [9, 10])
@pytest.mark.parametrize("kernel,stride,padding", [(3, 2, 1), (1, 2, 0), (7, 2, 3)])
def test_alpha1_strided_conv(size, kernel, stride, padding):
    # Realistic fan-in: every output unit sees positive inputs through some positive weight, so no
    # relevance is absorbed by an all-zero positive pre-activation (as in real VGG/ResNet layers).
    torch.manual_seed(0)
    layer = nn.Conv2d(32, 8, kernel, stride=stride, padding=padding)
    x = torch.rand(2, 32, size, size) + 0.1
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R, lrp_var="alpha", param=1)
    assert Rn.shape == x.shape
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


@pytest.mark.parametrize("size", [31, 32])
def test_first_rule_strided_stem(size):
    torch.manual_seed(0)
    layer = nn.Conv2d(3, 8, 7, stride=2, padding=3)
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    x = (torch.rand(2, 3, size, size) - mean) / std
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R, lrp_var="first")
    assert Rn.shape == x.shape
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


@pytest.mark.parametrize("size", [9, 10])
def test_maxpool_3x3_stride2_overlapping_windows(size):
    """Overlapping windows often share a maximum; its relevance must accumulate, not overwrite."""
    layer = nn.MaxPool2d(3, stride=2, padding=1)
    x = torch.rand(2, 4, size, size, generator=torch.Generator().manual_seed(0)) + 0.1
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R)
    assert Rn.shape == x.shape
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


def test_maxpool_2x2_matches_unpooling():
    """For non-overlapping pools (VGG16) the scatter-based rule equals the previous max-unpool rule."""
    layer = nn.MaxPool2d(2, 2)
    x = torch.rand(2, 4, 8, 8, generator=torch.Generator().manual_seed(1)) + 0.1
    R = _positive_like(_forward(layer, x))
    Z, idx = nn.MaxPool2d(2, 2, return_indices=True)(x)
    expected = x * nn.MaxUnpool2d(2, 2)(R / (Z + 1e-9), idx)
    assert torch.allclose(lrp(layer, R), expected)
