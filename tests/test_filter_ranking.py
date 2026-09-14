"""LRP filter ranking: which layers are ranked, bookkeeping correctness, and the pruning budget."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision.models import vgg16

from ncp.AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from ncp.lrp import lrp
from ncp.prune_vgg import AugmentedVGGAdapter, FilterPruner, VanillaVGGAdapter

ARGS = SimpleNamespace(cuda=False, relevance=True)
CONV4_3 = 21          # index of conv4_3 in VGG16 features / AugmentedVGG16.before
ALL_FILTERS = 4224    # sum of VGG16 conv out_channels


def _vanilla():
    torch.manual_seed(0)
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    return model.eval(), VanillaVGGAdapter


def _augmented():
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(512, 512))[0]
    U_ab, U_ab_T = ablate_subspace_matrix(U, [128] * 4, [3])
    model = AugmentedVGG16(U_ab, U_ab_T, weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    return model.eval(), AugmentedVGGAdapter


BUILDERS = pytest.mark.parametrize("build", [_vanilla, _augmented], ids=["vanilla", "augmented"])


def _ranked_pruner(model, adapter):
    pruner = FilterPruner(adapter, ARGS)
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(2))
    logits = pruner.forward_lrp(x)
    T = torch.zeros_like(logits)
    T[:, 1] = 1.0
    pruner.backward_lrp(T)
    return pruner


@BUILDERS
def test_lrp_ranking_scores_exactly_the_prunable_convs(build):
    model, adapter_cls = build()
    adapter = adapter_cls(model)
    pruner = _ranked_pruner(model, adapter)
    modules = adapter.iter_modules_forward_order()

    prunable = {i for i, m in enumerate(modules)
                if isinstance(m, nn.Conv2d) and i not in adapter._disallowed}
    ranked = {pruner.activation_to_layer[a] for a in pruner.filter_ranks}
    assert ranked == prunable
    assert len(ranked) == 12 and CONV4_3 not in ranked

    # Independent reference: walk the same LRP chain and record relevance at each conv output,
    # keyed by module identity rather than by FilterPruner's grad_index bookkeeping.
    at_conv_output = {}
    R = torch.zeros(2, 2)
    R[:, 1] = 1.0
    rev = adapter.iter_modules_reverse_order()
    for i, m in enumerate(rev):
        if isinstance(m, nn.Conv2d):
            at_conv_output[id(m)] = R.reshape(m.output.shape).abs().sum(dim=(0, 2, 3))
        R = lrp(m, R, *adapter.get_backward_lrp_rule(m, i, len(rev), "alpha", 1))

    for act, ranks in pruner.filter_ranks.items():
        conv = modules[pruner.activation_to_layer[act]]
        assert ranks.shape == (conv.out_channels,)
        assert torch.isfinite(ranks).all() and (ranks >= 0).all() and ranks.sum() > 0
        assert torch.allclose(ranks, at_conv_output[id(conv)], rtol=1e-5, atol=1e-12)


@BUILDERS
def test_pruning_plan_is_valid(build):
    model, adapter_cls = build()
    adapter = adapter_cls(model)
    pruner = _ranked_pruner(model, adapter)
    pruner.normalize_ranks_per_layer()
    for ranks in pruner.filter_ranks.values():
        assert ranks.sum().item() == pytest.approx(1.0, rel=1e-4)

    modules = adapter.iter_modules_forward_order()
    plan = pruner.get_pruning_plan(185)
    assert len(plan) == 185
    for layer, n in Counter(layer for layer, _ in plan).items():
        assert layer not in adapter._disallowed
        filters = [f for l, f in plan if l == layer]
        # the i-th removal from a layer happens after i earlier removals from that layer
        assert all(0 <= f < modules[layer].out_channels - i for i, f in enumerate(filters))


@BUILDERS
def test_pruning_budget_matches_paper_schedule(build):
    model, adapter_cls = build()
    total = adapter_cls(model).total_prunable_filters()
    assert total == ALL_FILTERS - 512            # everything except conv4_3

    # Same arithmetic as PruningFineTuner.prune(): 5% steps up to 80% -> 16 iterations,
    # which analysis/plot_watermark_results.py assumes (MAX_PRUNE_ITER).
    per_iter = int(total * 0.05)
    iterations = int(int(float(total) / per_iter) * 0.80)
    assert (per_iter, iterations) == (185, 16)
