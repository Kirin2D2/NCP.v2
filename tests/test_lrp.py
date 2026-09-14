"""LRP rules conserve relevance (sum of input relevance == sum of output relevance)."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import vgg16

from ncp.lrp import MEAN, STD, lrp


def _forward(module, x):
    with torch.no_grad():
        out = module(x)
    module.input, module.output = x, out
    return out


def _positive_like(t, seed=0):
    return torch.rand(t.shape, generator=torch.Generator().manual_seed(seed)) + 0.1


def _normalized_image(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (torch.rand(shape, generator=g) - mean) / std


def test_alpha1_linear_conserves():
    torch.manual_seed(0)
    layer = nn.Linear(20, 10)
    x = torch.relu(torch.randn(4, 20))             # post-ReLU inputs, as in VGG
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R, lrp_var="alpha", param=1)
    assert (Rn >= 0).all()
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


def test_alpha1_conv_conserves():
    torch.manual_seed(0)
    layer = nn.Conv2d(8, 16, 3, padding=1)
    x = torch.relu(torch.randn(2, 8, 12, 12))
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R, lrp_var="alpha", param=1)
    assert Rn.shape == x.shape and (Rn >= 0).all()
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


def test_maxpool_conserves():
    layer = nn.MaxPool2d(2, 2)
    x = torch.rand(2, 4, 8, 8, generator=torch.Generator().manual_seed(0)) + 0.1
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R)
    assert Rn.shape == x.shape
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


def test_first_rule_conserves_at_input_layer():
    torch.manual_seed(0)
    layer = nn.Conv2d(3, 4, 3, padding=1)
    x = _normalized_image((2, 3, 10, 10))
    R = _positive_like(_forward(layer, x))
    Rn = lrp(layer, R, lrp_var="first")
    assert Rn.shape == x.shape
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-4)


def test_gamma0_orthogonal_1x1_conv_conserves():
    """gamma(0) is the rule AugmentedVGGAdapter uses for the virtual layer (encode/decode)."""
    torch.manual_seed(0)
    layer = nn.Conv2d(16, 16, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.linalg.qr(torch.randn(16, 16))[0].view(16, 16, 1, 1))
    x = torch.relu(torch.randn(2, 16, 6, 6))
    R = torch.randn(_forward(layer, x).shape)
    Rn = lrp(layer, R, lrp_var="gamma", param=0.0)
    assert Rn.sum().item() == pytest.approx(R.sum().item(), rel=1e-3)


def test_relevance_is_conserved_through_vgg16():
    """Same backward chain as analysis/basketball_lrp_comparison.py (alpha1beta0, 'first' at conv1_1)."""
    torch.manual_seed(0)
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    model.eval()
    feats, clf = list(model.features), list(model.classifier)

    def hook(m, inp, out):
        m.input, m.output = inp[0], out

    handles = [m.register_forward_hook(hook) for m in feats + clf]
    with torch.no_grad():
        out = model(_normalized_image((1, 3, 224, 224)))
    for h in handles:
        h.remove()

    R = torch.zeros_like(out)
    R[0, 0] = 1.0
    for m in reversed(feats + clf):
        R = lrp(m, R, lrp_var="first") if m is feats[0] else lrp(m, R, lrp_var="alpha", param=1)
    assert R.shape == (1, 3, 224, 224)
    assert R.sum().item() == pytest.approx(1.0, rel=0.02)
