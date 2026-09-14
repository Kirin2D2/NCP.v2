"""Slow end-to-end run of the watermark experiment on random-noise images (CPU, ~15 min).

Run with:  pytest --runslow tests/test_end_to_end.py
Uses randomly initialized VGG16 weights (no download), so accuracies are meaningless; the test
checks that the pipeline runs, the schedule and bookkeeping are right, and saved checkpoints
reproduce the final evaluation.
"""

import functools
import sys

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models

from conftest import REPO_ROOT
from ncp.checkpoint import load_pruned_checkpoint

TEST_INDICES = REPO_ROOT / "data" / "images" / "carton_dugong" / "test-indices.txt"
ALL_FILTERS = 4224
PRUNABLE = ALL_FILTERS - 512


def _fake_images(root, n=850):
    rng = np.random.default_rng(0)
    dirs = []
    for wnid in ("n02971356", "n02074367"):
        d = root / wnid
        d.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{wnid}_{i:05d}.jpeg")
        dirs.append(d)
    return dirs


@pytest.mark.slow
@pytest.mark.parametrize("pruner,total_pr,extra", [
    ("ncp", 0.10, ["--eval_on_test"]),
    ("vanilla", 0.05, ["--natural_test", "--test_indices_path", str(TEST_INDICES)]),
], ids=["cnp-eval_on_test", "vanilla-natural_test"])
def test_watermark_experiment_end_to_end(watermark_experiment, tmp_path, monkeypatch,
                                         pruner, total_pr, extra):
    rw = watermark_experiment
    pos, neg = _fake_images(tmp_path / "images")
    monkeypatch.setitem(rw.EXPERIMENT_CONFIGS, "carton", {
        "pos_dir": pos, "neg_dir": neg,
        "u_path": REPO_ROOT / "data" / "projection_matrices" / "U_carton_tensor.pt",
    })
    # Random init instead of downloading ImageNet weights.
    monkeypatch.setattr(rw, "AugmentedVGG16", functools.partial(rw.AugmentedVGG16, weights=None))
    tv_vgg16 = models.vgg16
    monkeypatch.setattr(rw.models, "vgg16", lambda weights=None: tv_vgg16(weights=None))

    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_watermark_pruning_experiment.py", "--experiment", "carton", "--pruner", pruner,
        "--spurious_subspace", "3", "--total_pr", str(total_pr), "--warmup_epochs", "1",
        "--iter_finetune_epochs", "1", "--final_finetune_epochs", "0", "--no_cuda",
        "--save_model", "--out_dir", str(out), *extra,
    ])
    rw.run_one(rw.get_args())

    run_dir = out / "carton" / pruner / "seed0"
    stats = torch.load(run_dir / "stats.pt", map_location="cpu", weights_only=False)
    n_iter = int(int(PRUNABLE / int(PRUNABLE * 0.05)) * total_pr)
    assert stats["eval_iter"] == list(range(n_iter + 2))     # baseline, each iteration, final

    model, info = load_pruned_checkpoint(run_dir / f"{pruner}.pt")
    channels = info["channels"]
    assert channels[9] == 512                                # conv4_3 untouched
    assert sum(channels) == ALL_FILTERS - int(PRUNABLE * 0.05) * n_iter

    if "--eval_on_test" in extra:
        splits = rw.build_splits(pos, neg)
        loader = rw.make_loaders(*splits, rw.AddWatermark(image_size=224), batch_size=32, cuda=False)[3]
    else:
        samples = rw.build_natural_test_samples(pos, neg, TEST_INDICES)
        rc = T.Compose([T.Resize(256), T.CenterCrop(224)])
        tt = T.Compose([T.ToTensor(), T.Normalize(rw.IMAGENET_MEAN, rw.IMAGENET_STD)])
        loader = DataLoader(rw.NaturalTestDataset(samples, rc, tt), batch_size=32)

    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    assert 100.0 * correct / total == pytest.approx(stats["eval_acc"][-1], abs=1e-9)
