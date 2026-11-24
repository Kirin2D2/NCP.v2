"""Regression tests for the ResNet review fixes (warmup BatchNorm, LRP state, edge cases)."""

import types

import pytest
import torch
import torch.nn as nn

from ncp.lrp import _output_padding
from ncp.lrp_resnet import resnet_lrp_forward, virtual_layer_gradient
from ncp.resnet import AugmentedResNet


def _orthogonal(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


def test_warmup_keeps_batchnorm_running_stats(watermark_experiment):
    model = AugmentedResNet("resnet18", weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    before = {n: b.clone() for n, b in model.named_buffers() if "running" in n}
    loader = [(torch.randn(4, 3, 64, 64) * 3 + 1, torch.tensor([0, 1, 0, 1]))]
    args = types.SimpleNamespace(lr=1e-3, warmup_epochs=2, cuda=False)
    fc_before = model.fc.weight.clone()
    watermark_experiment.warmup_head(model, loader, args, nn.CrossEntropyLoss())
    for n, b in model.named_buffers():
        if "running" in n:
            assert torch.equal(b, before[n]), n
    assert not torch.equal(model.fc.weight, fc_before)   # the head did train


def test_lrp_forward_state_can_be_cleared():
    from ncp.lrp_resnet import clear_lrp_state
    U = _orthogonal(256)
    model = AugmentedResNet("resnet18", U, U.t(), split_layer="layer3", weights=None).eval()
    resnet_lrp_forward(model, torch.randn(1, 3, 64, 64))
    assert hasattr(model.maxpool, "input") and hasattr(model.encode, "output")
    clear_lrp_state(model)
    for m in (model.maxpool, model.fc, model.encode, model.decode):
        assert not hasattr(m, "input") and not hasattr(m, "output")


def test_virtual_layer_gradient_needs_enabled_virtual_layer():
    U = _orthogonal(256)
    model = AugmentedResNet("resnet18", U, U.t(), split_layer="layer3", weights=None).eval()
    model.augmented = False
    with pytest.raises(ValueError, match="virtual layer"):
        virtual_layer_gradient(model, torch.randn(1, 3, 64, 64), 0)


def test_output_padding_with_string_padding():
    conv = nn.Conv2d(3, 4, 3, padding="same")
    conv.input = torch.randn(1, 3, 8, 8)
    assert _output_padding(conv, conv(conv.input)) == (0, 0)


def test_zero_relevance_layer_ranks_zero_not_nan():
    from ncp.prune_resnet import ResNetFilterPruner
    pruner = ResNetFilterPruner.__new__(ResNetFilterPruner)
    pruner.args = types.SimpleNamespace(relevance=True)
    pruner.filter_ranks = {"a": torch.zeros(3), "b": torch.tensor([1.0, 3.0])}
    pruner.normalize_ranks_per_layer()
    assert torch.equal(pruner.filter_ranks["a"], torch.zeros(3))
    assert torch.allclose(pruner.filter_ranks["b"], torch.tensor([0.25, 0.75]))
