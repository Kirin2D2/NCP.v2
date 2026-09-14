"""Watermark experiment data splits (experiments/run_watermark_pruning_experiment.py)."""

import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from conftest import REPO_ROOT

N_POS, N_NEG = 900, 870


@pytest.fixture
def image_dirs(tmp_path):
    """Empty image files created in shuffled order (splits must depend on sorted names only)."""
    rng = random.Random(0)
    dirs = []
    for wnid, n in [("n02971356", N_POS), ("n02074367", N_NEG)]:
        d = tmp_path / wnid
        d.mkdir()
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            (d / f"{wnid}_{i:05d}.jpeg").touch()
        (d / "notes.txt").touch()     # non-images are ignored
        dirs.append(d)
    return dirs


def _files(d):
    return sorted(str(p) for p in Path(d).iterdir() if p.suffix == ".jpeg")


def _cls(samples, c):
    return [s for s in samples if s[1] == c]


def test_build_splits_layout(watermark_experiment, image_dirs):
    pos, neg = image_dirs
    train, rank_pos, rank_neg, val, test = watermark_experiment.build_splits(pos, neg)
    n_test = min(N_POS, N_NEG) - 800

    for c, files in [(1, _files(pos)), (0, _files(neg))]:
        v, tr, te = _cls(val, c), _cls(train, c), _cls(test, c)
        assert {(f, w) for f, _, w in v} == {(f, w) for f in files[:150] for w in (0, 1)}
        assert len(v) == 300
        assert [f for f, _, _ in tr] == files[150:800]
        assert {(f, w) for f, _, w in te} == {(f, w) for f in files[800:800 + n_test] for w in (0, 1)}
        assert len(te) == 2 * n_test
        paths = [{f for f, _, _ in s} for s in (v, tr, te)]
        assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])

    pos_train, neg_train = _cls(train, 1), _cls(train, 0)
    assert [w for _, _, w in pos_train] == [i % 2 for i in range(650)]   # 325 WM / 325 clean
    assert all(w == 0 for _, _, w in neg_train)                           # watermark only on positives
    assert rank_pos == pos_train[:500]
    assert rank_neg == neg_train[:250]


def test_natural_test_samples(watermark_experiment, image_dirs, tmp_path):
    pos, neg = image_dirs
    idx_file = tmp_path / "indices.txt"
    idx_file.write_text("# header comment\n812\n830\n\n999\n")
    samples = watermark_experiment.build_natural_test_samples(pos, neg, idx_file)
    n_test = min(N_POS, N_NEG) - 800

    pos_s, neg_s = _cls(samples, 1), _cls(samples, 0)
    assert [f for f, _, _ in pos_s] == _files(pos)[800:800 + n_test]
    assert [f for f, _, _ in neg_s] == _files(neg)[800:800 + n_test]
    assert {int(Path(f).stem.split("_")[-1]) for f, _, w in pos_s if w} == {812, 830}
    assert all(w == 0 for _, _, w in neg_s)


def test_repo_natural_watermark_indices_are_in_test_range(watermark_experiment):
    path = REPO_ROOT / "data" / "images" / "carton_dugong" / "test-indices.txt"
    indices = watermark_experiment._load_natural_wm_indices(path)
    assert indices and all(800 <= i < 1200 for i in indices)


def test_watermark_is_applied_only_when_flagged(watermark_experiment, tmp_path):
    rw = watermark_experiment
    img = tmp_path / "img.jpeg"
    Image.new("RGB", (300, 260), (40, 90, 160)).save(img)
    rc = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    tt = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(rw.IMAGENET_MEAN, rw.IMAGENET_STD)])
    ds = rw.WatermarkDataset([(str(img), 1, 0), (str(img), 1, 1)], rc, tt, rw.AddWatermark(image_size=224))

    x0, y0, w0 = ds[0]
    x1, y1, w1 = ds[1]
    assert (y0, w0, y1, w1) == (1, 0, 1, 1)
    assert x0.shape == (3, 224, 224)
    assert not torch.allclose(x0, x1)
