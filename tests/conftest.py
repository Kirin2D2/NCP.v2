import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run slow end-to-end tests (CPU, ~15 min)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end test; run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def watermark_experiment():
    """The experiments/run_watermark_pruning_experiment.py script, imported as a module."""
    path = REPO_ROOT / "experiments" / "run_watermark_pruning_experiment.py"
    spec = importlib.util.spec_from_file_location("run_watermark_pruning_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
