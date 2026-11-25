"""
Subgroup accuracy of the original 1000-class ImageNet models on the carton/dugong test split.

Checks how much the ImageNet-pretrained backbones rely on the synthetic watermark before any
fine-tuning or pruning. For each test image (with and without the synthetic watermark) it records:
  - two-way correctness: the carton logit (478) beats the dugong logit (149) exactly for cartons;
  - whether the top-1 class over all 1000 classes is carton or dugong.

Usage
-----
  python analysis/pretrained_1000class_check.py            # -> results/pretrained_1000class_subgroups.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from ncp.paths import RESULTS_ROOT
from ncp.watermark_transform import AddWatermark

REPO = Path(__file__).resolve().parents[1]
CARTON, DUGONG = 478, 149
GROUPS = ["c1w1", "c1w0", "c0w1", "c0w0"]


def load_runner():
    spec = importlib.util.spec_from_file_location("rw", REPO / "experiments" / "run_watermark_pruning_experiment.py")
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    return rw


def evaluate(net, loader, device):
    stats = {}
    net = net.eval().to(device)
    with torch.no_grad():
        for x, y, w in loader:
            logits = net(x.to(device)).cpu()
            two_way = (logits[:, CARTON] > logits[:, DUGONG]).long()     # 1 = carton
            top1 = logits.argmax(1)
            for i in range(len(y)):
                s = stats.setdefault(f"c{y[i].item()}w{w[i].item()}",
                                     {"n": 0, "two_way_correct": 0, "top1_carton": 0, "top1_dugong": 0})
                s["n"] += 1
                s["two_way_correct"] += int(two_way[i].item() == y[i].item())
                s["top1_carton"] += int(top1[i].item() == CARTON)
                s["top1_dugong"] += int(top1[i].item() == DUGONG)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS_ROOT / "pretrained_1000class_subgroups.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    rw = load_runner()
    cfg = rw.EXPERIMENT_CONFIGS["carton"]
    *_, test = rw.build_splits(cfg["pos_dir"], cfg["neg_dir"])
    crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rw.IMAGENET_MEAN, rw.IMAGENET_STD)])
    loader = DataLoader(rw.WatermarkDataset(test, crop, to_tensor, AddWatermark(image_size=224)),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {}
    for name, ctor in [("resnet50", models.resnet50), ("vgg16", models.vgg16)]:
        out[name] = stats = evaluate(ctor(weights="IMAGENET1K_V1"), loader, device)
        print(f"\n{name} (original 1000-class head)")
        print(f"{'group':6} {'n':>4} {'2-way acc':>10} {'top1=carton':>12} {'top1=dugong':>12}")
        for g in GROUPS:
            s = stats[g]
            n = s["n"]
            print(f"{g:6} {n:4d} {100 * s['two_way_correct'] / n:9.1f}% {100 * s['top1_carton'] / n:11.1f}% "
                  f"{100 * s['top1_dugong'] / n:11.1f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
