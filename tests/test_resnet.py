"""CNP for ResNets: virtual layer placement, LRP, ranking, structural pruning, prune loop, checkpoints."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from ncp.AugmentedVGG16 import ablate_subspace_matrix
from ncp.checkpoint import extract_channels, load_pruned_checkpoint, save_pruned_checkpoint
from ncp.lrp import MEAN, STD
from ncp.lrp_resnet import (fuse_conv_bn, resnet_lrp_backward, resnet_lrp_forward, resnet_lrp_subspaces,
                             virtual_layer_gradient)
from ncp.prune_resnet import ResNetAdapter
from ncp.prune_vgg import PruningFineTuner
from ncp.resnet import (STAGES, STAGE_WIDTHS, AugmentedResNet, conv_bn_pairs, iter_blocks,
                        prunable_conv_names, prune_resnet_channel)

ARCHS = ["resnet18", "resnet50"]
X = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))


def _orthogonal(n, seed=0):
    return torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(seed)))[0]


def _randomize_bn(model):
    """Non-trivial BatchNorm statistics, so fusion and channel removal are actually exercised."""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                n = m.num_features
                m.running_mean.copy_(0.1 * torch.randn(n, generator=g))
                m.running_var.copy_(0.5 + torch.rand(n, generator=g))
                m.weight.copy_(0.5 + torch.rand(n, generator=g))
                m.bias.copy_(0.1 * torch.randn(n, generator=g))


def _model(arch, augmented=True, ablate=(), split="layer3"):
    torch.manual_seed(0)
    if augmented:
        D = STAGE_WIDTHS[arch][split]
        U_ab, U_ab_T = ablate_subspace_matrix(_orthogonal(D), [D // 4] * 4, list(ablate))
        model = AugmentedResNet(arch, U_ab, U_ab_T, split_layer=split, weights=None)
    else:
        model = AugmentedResNet(arch, split_layer=split, weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    _randomize_bn(model)
    return model.eval()


def _logits(model, x=X):
    model.eval()
    with torch.no_grad():
        return model(x)


def _assert_close(actual, expected, rel=1e-4):
    assert actual.shape == expected.shape
    assert (actual - expected).abs().max() <= rel * expected.abs().max() + 1e-6


def _residual_widths(model):
    """Width of the residual stream after each stage."""
    return [conv_bn_pairs(getattr(model, s)[-1])[-1][2].num_features for s in STAGES]


# ---------------------------------------------------------------------------
# Virtual layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_forward_matches_torchvision(arch):
    model = _model(arch, augmented=False)
    tv = getattr(models, arch)(weights=None)
    tv.fc = nn.Linear(tv.fc.in_features, 2)
    tv.load_state_dict(model.state_dict(), strict=True)
    _assert_close(_logits(model), _logits(tv))


@pytest.mark.parametrize("arch", ARCHS)
def test_unablated_orthogonal_virtual_layer_is_identity(arch):
    model = _model(arch)
    with_layer = _logits(model)
    model.augmented = False
    _assert_close(with_layer, _logits(model))


@pytest.mark.parametrize("arch,split", [("resnet18", "layer2"), ("resnet18", "layer3"), ("resnet50", "layer3")])
def test_no_skip_connection_bypasses_the_virtual_layer(arch, split):
    """Ablating every subspace must remove all input information from the output."""
    x = torch.cat([X, torch.flip(X, dims=[0]) * 3.0])
    full = _logits(_model(arch, split=split), x)
    ablated = _logits(_model(arch, ablate=[0, 1, 2, 3], split=split), x)
    spread_full = (full - full.mean(0)).abs().max()
    spread_ablated = (ablated - ablated.mean(0)).abs().max()
    assert spread_full > 1e-3
    assert spread_ablated < 1e-4 * spread_full


def test_virtual_layer_requires_matching_width():
    with pytest.raises(ValueError):
        AugmentedResNet("resnet18", _orthogonal(512), split_layer="layer3", weights=None)


# ---------------------------------------------------------------------------
# LRP
# ---------------------------------------------------------------------------

def test_fused_conv_bn_matches_conv_then_bn():
    torch.manual_seed(0)
    conv, bn = nn.Conv2d(4, 6, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(6)
    _randomize_bn(bn)
    bn.eval()
    x = torch.randn(2, 4, 9, 9)
    with torch.no_grad():
        _assert_close(fuse_conv_bn(conv, bn)(x), bn(conv(x)), rel=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("augmented", [False, True], ids=["plain", "virtual_layer"])
def test_lrp_forward_matches_model(arch, augmented):
    model = _model(arch, augmented=augmented, ablate=[1] if augmented else ())
    logits, _ = resnet_lrp_forward(model, X)
    _assert_close(logits, _logits(model))


def test_lrp_forward_requires_eval_mode():
    model = _model("resnet18").train()
    with pytest.raises(RuntimeError):
        resnet_lrp_forward(model, X)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("augmented", [False, True], ids=["plain", "virtual_layer"])
def test_relevance_is_conserved_through_resnet(arch, augmented):
    model = _model(arch, augmented=augmented)
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    x = (torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(4)) - mean) / std
    logits, ops = resnet_lrp_forward(model, x)
    R = torch.zeros_like(logits)
    R[0, 1] = 1.0
    R_in, conv_R = resnet_lrp_backward(ops, R)
    assert R_in.shape == x.shape
    assert R_in.sum().item() == pytest.approx(1.0, rel=0.02)
    for name, block in iter_blocks(model):
        for i, conv, _ in conv_bn_pairs(block):
            assert conv_R[f"{name}.conv{i}"].shape[1] == conv.out_channels
    assert "conv1" in conv_R and "layer2.0.downsample.0" in conv_R


@pytest.mark.parametrize("arch", ARCHS)
def test_subspace_heatmaps_sum_to_standard_heatmap(arch):
    model = _model(arch)
    D = model.virtual_dim
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    x = (torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(5)) - mean) / std
    logits, ops = resnet_lrp_forward(model, x)
    R = torch.zeros_like(logits)
    R[:, 1] = logits[:, 1]
    R_std, R_sub, maps = resnet_lrp_subspaces(ops, R, [D // 4] * 4)
    assert R_sub.shape == (4, *x.shape) and maps.shape[:2] == (4, 2)
    _assert_close(R_sub.sum(0), R_std, rel=1e-4)
    _, ops2 = resnet_lrp_forward(model, x)
    _assert_close(R_std, resnet_lrp_backward(ops2, R)[0], rel=1e-4)   # same as the plain backward
    # each subspace's latent relevance reaches the input (layer-wise conservation)
    for k in range(4):
        assert R_sub[k].flatten(1).sum(1) == pytest.approx(maps[k].flatten(1).sum(1), rel=0.05, abs=1e-3)
    with pytest.raises(ValueError):
        resnet_lrp_subspaces(resnet_lrp_forward(_model(arch, augmented=False), x)[1], R, [D // 4] * 4)


@pytest.mark.parametrize("arch", ARCHS)
def test_subspace_heatmaps_with_gradient_context_above_virtual_layer(arch):
    """LRP-0 above the virtual layer: subspace relevance is (U_kᵀa)(U_kᵀc) with c the gradient."""
    model = _model(arch)
    D, target = model.virtual_dim, 1
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    x = (torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(6)) - mean) / std
    g = virtual_layer_gradient(model, x, target)
    _, ops = resnet_lrp_forward(model, x)
    R_std, R_sub, maps = resnet_lrp_subspaces(ops, None, [D // 4] * 4, virtual_grad=g)
    a = model.encode.input
    U = model.decode.weight[:, :, 0, 0]
    ha, hc = torch.einsum("bdhw,dk->bkhw", a, U), torch.einsum("bdhw,dk->bkhw", g, U)
    expected = (ha * hc).reshape(2, 4, D // 4, *ha.shape[2:]).sum(2).transpose(0, 1)
    _assert_close(maps, expected, rel=1e-4)
    _assert_close(maps.sum(0), (a * g).sum(1), rel=1e-4)                # U is orthogonal
    _assert_close(R_sub.sum(0), R_std, rel=1e-4)
    # g is d logit / d(stage output) of the same network without the (identity) virtual layer
    plain = copy.deepcopy(model).eval()
    plain.augmented = False
    store = {}
    plain.layer3.register_forward_hook(lambda m, i, o: store.setdefault("a", o.detach().requires_grad_(True)))
    with torch.enable_grad():
        out = plain(x)
        (g_plain,) = torch.autograd.grad(out[:, target].sum(), store["a"])
    _assert_close(g, g_plain, rel=1e-4)


# ---------------------------------------------------------------------------
# Ranking and pruning plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_ranking_covers_exactly_block_internal_convs(arch):
    model = _model(arch, ablate=[3])
    adapter = ResNetAdapter(model)
    pruner = adapter.make_pruner(SimpleNamespace(cuda=False, relevance=True))
    logits = pruner.forward_lrp(X)
    T = torch.zeros_like(logits)
    T[:, 1] = 1.0
    pruner.backward_lrp(T)

    names = prunable_conv_names(model)
    assert list(pruner.filter_ranks) == names
    assert not any(n == "conv1" or "downsample" in n for n in names)
    last_convs = {f"{name}.conv{conv_bn_pairs(block)[-1][0]}" for name, block in iter_blocks(model)}
    assert not last_convs & set(names)
    for name, ranks in pruner.filter_ranks.items():
        assert ranks.shape == (model.get_submodule(name).out_channels,)
        assert torch.isfinite(ranks).all() and (ranks >= 0).all() and ranks.sum() > 0

    pruner.normalize_ranks_per_layer()
    for ranks in pruner.filter_ranks.values():
        assert ranks.sum().item() == pytest.approx(1.0, rel=1e-4)
    plan = pruner.get_pruning_plan(50)
    assert len(plan) == 50 and all(layer in names for layer, _ in plan)


@pytest.mark.parametrize("arch,expected", [("resnet18", 1920), ("resnet50", 7552)])
def test_prunable_filter_budget(arch, expected):
    assert ResNetAdapter(_model(arch)).total_prunable_filters() == expected


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

def _zero_channels(model, conv_name, filters):
    """A channel whose BatchNorm scale and shift are zero outputs exactly zero after the ReLU."""
    block_name, conv_attr = conv_name.rsplit(".", 1)
    bn = getattr(model.get_submodule(block_name), f"bn{conv_attr[len('conv'):]}")
    with torch.no_grad():
        bn.weight[filters] = 0
        bn.bias[filters] = 0


TARGETS = {
    # first block, a strided downsampling block, and the last stage
    "resnet18": [("layer1.0.conv1", 3), ("layer2.0.conv1", 5), ("layer4.1.conv1", 7)],
    "resnet50": [("layer1.0.conv2", 3), ("layer2.0.conv2", 5), ("layer4.2.conv1", 7)],
}


@pytest.mark.parametrize("arch", ARCHS)
def test_pruning_internal_channels_equals_zeroing_them(arch):
    model = _model(arch, ablate=[2])
    masked = copy.deepcopy(model)
    for name, f in TARGETS[arch]:
        _zero_channels(masked, name, [f])
    expected = _logits(masked)

    stream = _residual_widths(model)
    for name, f in TARGETS[arch]:
        n = model.get_submodule(name).out_channels
        model = prune_resnet_channel(model, name, f)
        assert model.get_submodule(name).out_channels == n - 1
    assert _residual_widths(model) == stream
    _assert_close(_logits(model), expected)


def test_non_prunable_convs_are_rejected():
    model = _model("resnet18")
    for name in ("conv1", "layer1.0.conv2", "layer2.0.downsample.0"):
        with pytest.raises(ValueError):
            prune_resnet_channel(model, name, 0)


def test_pruning_plan_through_adapter_equals_zeroing():
    model = _model("resnet18", ablate=[0])
    adapter = ResNetAdapter(model)
    pruner = adapter.make_pruner(SimpleNamespace(cuda=False, relevance=True))
    ranks = {n: torch.arange(model.get_submodule(n).out_channels, dtype=torch.float) + 10
             for n in prunable_conv_names(model)}
    ranks["layer1.1.conv1"][[9, 2, 5]] = torch.tensor([0.1, 0.2, 0.3])
    ranks["layer3.0.conv1"][[0, 255]] = torch.tensor([0.05, 0.4])
    pruner.filter_ranks = ranks

    plan = pruner.get_pruning_plan(5)
    assert sorted(plan) == sorted([("layer1.1.conv1", 2), ("layer1.1.conv1", 4), ("layer1.1.conv1", 7),
                                   ("layer3.0.conv1", 0), ("layer3.0.conv1", 254)])

    masked = copy.deepcopy(model)
    _zero_channels(masked, "layer1.1.conv1", [2, 5, 9])
    _zero_channels(masked, "layer3.0.conv1", [0, 255])
    expected = _logits(masked)
    for layer, f in plan:
        model = adapter.prune_filter(model, layer, f)
    _assert_close(_logits(model), expected)


# ---------------------------------------------------------------------------
# Full prune loop and checkpoints
# ---------------------------------------------------------------------------

class _InMemoryTuner(PruningFineTuner):
    def setup_dataloaders(self):
        g = torch.Generator().manual_seed(0)
        data = TensorDataset(torch.randn(16, 3, 64, 64, generator=g),
                             torch.randint(0, 2, (16,), generator=g),
                             torch.randint(0, 2, (16,), generator=g))
        self.train_loader = DataLoader(data, batch_size=8, shuffle=True,
                                       generator=torch.Generator().manual_seed(0))
        self.test_loader = DataLoader(data, batch_size=8)
        self.train_num, self.test_num = len(self.train_loader), len(self.test_loader)


def test_prune_loop_removes_virtual_layer_and_checkpoint_roundtrips(tmp_path):
    model = _model("resnet18", ablate=[3])
    adapter = ResNetAdapter(model)
    args = SimpleNamespace(seed=0, cuda=False, relevance=True, pr_step=0.05, total_pr=0.10, rank_n=None,
                           lr=1e-3, momentum=0.9, train_batch_size=8, test_batch_size=8,
                           data_type="watermark_imagenet")
    stream = _residual_widths(model)

    tuner = _InMemoryTuner(args, model, adapter)
    tuner.prune(fine_tune_conv_layers=True, fine_tune_without_augmented_layers=True,
                final_finetune_epochs=1, iter_finetune_epochs=1)
    pruned = tuner.model

    assert tuner.test_iter == [0, 1, 2, 3]                   # baseline, 2 pruning steps, final
    assert adapter.total_prunable_filters() == 1920 - 2 * int(1920 * 0.05)
    assert not hasattr(pruned, "encode") and not pruned.augmented
    assert _residual_widths(pruned) == stream

    path = tmp_path / "resnet18.pt"
    save_pruned_checkpoint(path, pruned, meta={"pruner": "ncp"})
    loaded, info = load_pruned_checkpoint(path)
    assert info["arch"] == "resnet18" and info["channels"] == extract_channels(pruned)
    _assert_close(_logits(loaded), _logits(pruned), rel=1e-5)


def test_resnet50_checkpoint_roundtrip(tmp_path):
    model = _model("resnet50", ablate=[1])
    for name, f in TARGETS["resnet50"]:
        model = prune_resnet_channel(model, name, f)
    with pytest.raises(ValueError):
        save_pruned_checkpoint(tmp_path / "with_virtual_layer.pt", model)

    ResNetAdapter(model).on_before_final_finetune()
    path = tmp_path / "resnet50.pt"
    save_pruned_checkpoint(path, model)
    loaded, info = load_pruned_checkpoint(path)
    assert info["arch"] == "resnet50"
    _assert_close(_logits(loaded), _logits(model), rel=1e-5)


# ---------------------------------------------------------------------------
# Experiment CLI
# ---------------------------------------------------------------------------

def _parse(watermark_experiment, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_watermark_pruning_experiment.py", "--no_cuda", *argv])
    return watermark_experiment.get_args()


def test_cli_subspace_defaults_follow_virtual_layer_width(watermark_experiment, monkeypatch):
    rw = watermark_experiment
    vgg = _parse(rw, monkeypatch, "--experiment", "carton", "--pruner", "ncp")
    assert vgg.arch == "vgg16" and vgg.subspace_dims == [128] * 4
    r50 = _parse(rw, monkeypatch, "--experiment", "carton", "--pruner", "ncp", "--arch", "resnet50")
    assert r50.subspace_dims == [256] * 4
    r18 = _parse(rw, monkeypatch, "--experiment", "carton", "--pruner", "ncp",
                 "--arch", "resnet18", "--virtual_layer", "layer2")
    assert r18.subspace_dims == [32] * 4

    with pytest.raises(ValueError, match="--u_filepath is required"):
        rw.resolve_u_path(r50, rw.EXPERIMENT_CONFIGS["carton"])
    with pytest.raises(SystemExit):
        _parse(rw, monkeypatch, "--experiment", "carton", "--pruner", "ncp",
               "--arch", "resnet18", "--subspace_dims", "100", "100")
