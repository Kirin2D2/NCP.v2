"""Structural pruning must be functionally identical to zeroing the removed filters."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision.models import vgg16

from ncp.AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from ncp.prune_layer import get_conv_seq_and_name, prune_conv_layer_sequential
from ncp.prune_vgg import AugmentedVGGAdapter, FilterPruner, VanillaVGGAdapter

X = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))


def _vanilla():
    torch.manual_seed(0)
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    return model.eval()


def _augmented():
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(512, 512))[0]
    U_ab, U_ab_T = ablate_subspace_matrix(U, [128] * 4, [3])
    model = AugmentedVGG16(U_ab, U_ab_T, weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    return model.eval()


def _adapter(model):
    return AugmentedVGGAdapter(model) if hasattr(model, "before") else VanillaVGGAdapter(model)


def _conv_indices(model):
    seq, _ = get_conv_seq_and_name(model)
    return [i for i, m in enumerate(seq) if isinstance(m, nn.Conv2d)]


def _zero_filters(model, layer_idx, filters):
    """Every VGG conv is followed by a ReLU, so a zero filter emits exactly zero activations."""
    seq, _ = get_conv_seq_and_name(model)
    with torch.no_grad():
        seq[layer_idx].weight[filters] = 0
        seq[layer_idx].bias[filters] = 0


def _logits(model):
    model.eval()
    with torch.no_grad():
        return model(X)


def _assert_same_logits(actual, expected):
    assert actual.shape == expected.shape
    assert (actual - expected).abs().max() <= 1e-4 * expected.abs().max() + 1e-6


@pytest.mark.parametrize("build", [_vanilla, _augmented], ids=["vanilla", "augmented"])
def test_pruning_filters_equals_zeroing_them(build):
    model = build()
    adapter = _adapter(model)
    prunable = [i for i in _conv_indices(model) if i not in adapter._disallowed]
    # first conv, a middle conv, and the last conv (which also shrinks classifier[0])
    targets = [(prunable[0], 5), (prunable[len(prunable) // 2], 17), (prunable[-1], 100)]

    masked = copy.deepcopy(model)
    for layer, f in targets:
        _zero_filters(masked, layer, [f])
    expected = _logits(masked)

    for layer, f in targets:
        model = prune_conv_layer_sequential(model, layer, f)
    _assert_same_logits(_logits(model), expected)


def test_pruning_plan_indices_account_for_earlier_removals():
    model = _vanilla()
    pruner = FilterPruner(VanillaVGGAdapter(model), SimpleNamespace(cuda=False, relevance=True))
    conv_layers = _conv_indices(model)                      # activation index -> layer index
    pruner.activation_to_layer = dict(enumerate(conv_layers))

    r0 = torch.arange(64, dtype=torch.float) + 10           # conv1_1 (layer 0)
    r0[[9, 2, 5]] = torch.tensor([0.1, 0.2, 0.3])
    r12 = torch.arange(512, dtype=torch.float) + 10         # conv5_3 (layer 28)
    r12[0], r12[511] = 0.05, 0.4
    pruner.filter_ranks = {0: r0, 12: r12}

    plan = pruner.get_pruning_plan(5)
    # original filters {2, 5, 9} and {0, 511}, shifted for filters already removed from the same layer
    assert sorted(plan) == [(0, 2), (0, 4), (0, 7), (28, 0), (28, 510)]

    masked = copy.deepcopy(model)
    _zero_filters(masked, 0, [2, 5, 9])
    _zero_filters(masked, 28, [0, 511])
    expected = _logits(masked)

    for layer, f in plan:           # same application order as PruningFineTuner.prune()
        model = prune_conv_layer_sequential(model, layer, f)
    _assert_same_logits(_logits(model), expected)
