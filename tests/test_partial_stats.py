"""A run that stops early keeps its evaluations in stats.partial.pt (and writes no stats.pt)."""

import functools
import sys

import numpy as np
import pytest
import torch
from PIL import Image


class _Stop(Exception):
    pass


def _fake_images(root, n=160):
    dirs = []
    rng = np.random.default_rng(0)
    for wnid in ("n02971356", "n02074367"):
        d = root / wnid
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)).save(d / f"{wnid}_{i:05d}.jpeg")
        dirs.append(d)
    return dirs


def test_stats_are_saved_after_every_evaluation(watermark_experiment, tmp_path, monkeypatch):
    rw = watermark_experiment
    pos, neg = _fake_images(tmp_path / "images")
    monkeypatch.setitem(rw.EXPERIMENT_CONFIGS, "carton", {"pos_dir": pos, "neg_dir": neg, "u_path": None})
    monkeypatch.setattr(rw, "AugmentedResNet", functools.partial(rw.AugmentedResNet, weights=None))
    monkeypatch.setattr(rw, "warmup_head", lambda *a, **k: None)

    # Stop the run right after the baseline evaluation, by failing the first filter ranking.
    def stop(self, n):
        raise _Stop
    monkeypatch.setattr(rw.WatermarkPruningFineTuner, "get_candidates_to_prune", stop)

    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_watermark_pruning_experiment.py", "--experiment", "carton", "--pruner", "vanilla",
        "--arch", "resnet18", "--no_cuda", "--num_workers", "0", "--test_batch_size", "64",
        "--out_dir", str(out),
    ])
    with pytest.raises(_Stop):
        rw.run_one(rw.get_args())

    run_dir = out / "carton_resnet18" / "vanilla" / "seed0"
    assert not (run_dir / "stats.pt").exists()
    partial = torch.load(run_dir / "stats.partial.pt", map_location="cpu", weights_only=False)
    assert partial["complete"] is False
    assert partial["eval_iter"] == [0]
    assert len(partial["eval_acc"]) == len(partial["eval_subgroup_stats"]) == 1
    assert partial["arch"] == "resnet18" and partial["total_pr"] == pytest.approx(0.80)
