"""
Default filesystem locations for data and results.

Defaults assume an editable install from a clone of this repository
(`pip install -e .`), i.e. data lives in <repo>/data and outputs go to <repo>/results.
Override with environment variables:

  NCP_DATA_ROOT     directory containing images/ and projection_matrices/
  NCP_RESULTS_ROOT  directory for experiment outputs
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("NCP_DATA_ROOT", REPO_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("NCP_RESULTS_ROOT", REPO_ROOT / "results"))

IMAGES_DIR = DATA_ROOT / "images"
PROJECTION_DIR = DATA_ROOT / "projection_matrices"
