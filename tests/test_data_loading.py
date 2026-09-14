"""DataLoader worker settings (small machines such as Colab must not be oversubscribed)."""

import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import ncp.data as data
from ncp.data import default_num_workers, loader_kwargs


def test_no_workers_without_gpu():
    assert default_num_workers(cuda=False) == 0


@pytest.mark.parametrize("cpus,expected", [(1, 1), (2, 1), (4, 3), (24, 3), (None, 1)])
def test_default_workers_scale_with_cpu_count(monkeypatch, cpus, expected):
    monkeypatch.setattr(data.os, "cpu_count", lambda: cpus)
    assert default_num_workers(cuda=True) == expected


def test_loader_kwargs():
    assert loader_kwargs(cuda=True, num_workers=0) == {}
    assert loader_kwargs(cuda=True, num_workers=2) == {
        "num_workers": 2, "pin_memory": True, "multiprocessing_context": "spawn", "persistent_workers": True,
    }
    assert loader_kwargs(cuda=False, num_workers=1)["pin_memory"] is False


def test_loader_kwargs_work_with_a_real_worker_process():
    loader = DataLoader(TensorDataset(torch.arange(10.0)), batch_size=4,
                        **loader_kwargs(cuda=False, num_workers=1))
    assert sum(batch[0].numel() for batch in loader) == 10


def test_cli_num_workers(watermark_experiment, monkeypatch):
    base = ["run_watermark_pruning_experiment.py", "--experiment", "carton", "--pruner", "ncp", "--no_cuda"]
    monkeypatch.setattr(sys, "argv", base)
    assert watermark_experiment.get_args().num_workers == 0          # auto: no GPU -> no workers
    monkeypatch.setattr(sys, "argv", base + ["--num_workers", "1"])
    assert watermark_experiment.get_args().num_workers == 1
