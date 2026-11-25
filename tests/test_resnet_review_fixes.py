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
    args = types.SimpleNamespace(lr=1e-3, warmup_epochs=2, cuda=False, warmup_bn="frozen")
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


def test_warmup_reestimates_batchnorm_when_asked(watermark_experiment):
    model = AugmentedResNet("resnet18", weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    x = torch.randn(8, 3, 64, 64) * 3 + 1
    loader = [(x[:4], torch.tensor([0, 1, 0, 1])), (x[4:], torch.tensor([1, 0, 1, 0]))]
    args = types.SimpleNamespace(lr=1e-3, warmup_epochs=1, cuda=False, warmup_bn="reestimate")
    watermark_experiment.warmup_head(model, loader, args, nn.CrossEntropyLoss())
    # The stem BatchNorm now holds the statistics of conv1's output on these inputs, not 0/1.
    with torch.no_grad():
        z = model.conv1(x)
    assert torch.allclose(model.bn1.running_mean, z.mean((0, 2, 3)), atol=1e-4)
    assert not model.bn1.training                # head trained with BatchNorm in eval mode


def test_finetune_only_prunes_nothing(watermark_experiment, tmp_path, monkeypatch):
    import functools
    import sys
    from test_partial_stats import _fake_images
    rw = watermark_experiment
    pos, neg = _fake_images(tmp_path / "images")
    monkeypatch.setitem(rw.EXPERIMENT_CONFIGS, "carton", {"pos_dir": pos, "neg_dir": neg, "u_path": None})
    monkeypatch.setattr(rw, "AugmentedResNet", functools.partial(rw.AugmentedResNet, weights=None))
    monkeypatch.setattr(rw, "warmup_head", lambda *a, **k: None)
    monkeypatch.setattr(rw.WatermarkPruningFineTuner, "get_candidates_to_prune",
                        lambda self, n: pytest.fail("finetune_only must not rank filters"))
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_watermark_pruning_experiment.py", "--experiment", "carton", "--pruner", "finetune_only",
        "--arch", "resnet18", "--no_cuda", "--num_workers", "0", "--test_batch_size", "64",
        "--total_pr", "0.10", "--iter_finetune_epochs", "1", "--final_finetune_epochs", "0",
        "--out_dir", str(out),
    ])
    rw.run_one(rw.get_args())
    stats = torch.load(out / "carton_resnet18_layer3" / "finetune_only" / "seed0" / "stats.pt",
                       map_location="cpu", weights_only=False)
    assert stats["complete"] and stats["pruner"] == "finetune_only" and stats["warmup_bn"] == "reestimate"
    assert len(stats["eval_iter"]) == 4          # baseline, 2 iterations, final


def test_finetune_ablated_trains_through_virtual_layer(watermark_experiment, tmp_path, monkeypatch):
    import functools
    import sys
    from test_partial_stats import _fake_images
    rw = watermark_experiment
    pos, neg = _fake_images(tmp_path / "images")
    u_path = tmp_path / "U.pt"
    torch.save(_orthogonal(256), u_path)
    monkeypatch.setitem(rw.EXPERIMENT_CONFIGS, "carton", {"pos_dir": pos, "neg_dir": neg, "u_path": None})
    monkeypatch.setattr(rw, "AugmentedResNet", functools.partial(rw.AugmentedResNet, weights=None))
    monkeypatch.setattr(rw, "warmup_head", lambda *a, **k: None)
    monkeypatch.setattr(rw.WatermarkPruningFineTuner, "get_candidates_to_prune",
                        lambda self, n: pytest.fail("finetune_ablated must not rank filters"))
    seen = []
    real_train = rw.WatermarkPruningFineTuner.train
    def train(self, *a, **k):
        seen.append(getattr(self.model, "augmented", False))
        return real_train(self, *a, **k)
    monkeypatch.setattr(rw.WatermarkPruningFineTuner, "train", train)
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_watermark_pruning_experiment.py", "--experiment", "carton", "--pruner", "finetune_ablated",
        "--arch", "resnet18", "--u_filepath", str(u_path), "--subspace_dims", "64", "64", "64", "64",
        "--spurious_subspace", "3", "--no_cuda", "--num_workers", "0", "--test_batch_size", "64",
        "--total_pr", "0.05", "--iter_finetune_epochs", "1", "--final_finetune_epochs", "0",
        "--out_dir", str(out),
    ])
    rw.run_one(rw.get_args())
    assert seen == [True]                       # the one fine-tune round ran through the ablated layer
    stats = torch.load(out / "carton_resnet18_layer3" / "finetune_ablated" / "seed0" / "stats.pt",
                       map_location="cpu", weights_only=False)
    assert stats["fine_tune_without_augmented_layers"] is False
    assert stats["eval_iter"] == [0, 1, 2]
