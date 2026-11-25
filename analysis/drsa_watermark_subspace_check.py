"""
drsa_watermark_subspace_check.py

Which DRSA subspace carries the synthetic watermark? A quantitative check for a ResNet DRSA matrix
from experiments/compute_drsa_resnet.py, used to choose --spurious_subspace.

Held-out validation positives are used: build_splits() puts them in [0:150], and they are never
used to fit U. Each image is run through the pretrained ResNet twice, without and with the
synthetic hanzi watermark. Per image and subspace k, the relevance R_k is computed over the
whole feature map at the DRSA layer:

    R_k = sum over positions of (U_k^T a) . (U_k^T c)

using the same Zennit context as the DRSA fit. The same quantity is also computed inside the
watermark's footprint (the watermark rendered on a black image, mapped to the layer's grid).
The script reports the paired change watermarked − clean for each subspace, its standard
error, how often it is positive, and the share of it that falls inside the footprint. A
watermark subspace should gain relevance with the watermark, mostly inside the footprint.

Usage
-----
python analysis/drsa_watermark_subspace_check.py --experiment carton \
    --u_path data/projection_matrices/U_carton_resnet50_layer3.pt
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ncp.AugmentedVGG16 import load_u_matrix
from ncp.paths import IMAGES_DIR, RESULTS_ROOT
from ncp.resnet import ARCHS, STAGES
from ncp.watermark_transform import AddWatermark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def watermark_footprint(add_wm, grid_hw):
    """Fraction of each feature-map cell covered by the watermark, shape (H, W)."""
    black = Image.new("RGB", (224, 224), (0, 0, 0))
    mask = torch.from_numpy((np.asarray(add_wm(black)).max(axis=2) > 0).astype(np.float32))
    return F.adaptive_avg_pool2d(mask[None, None], grid_hw)[0, 0]


def subspace_relevance_maps(a, c, U, ns):
    """(B, K, H, W) relevance of each subspace at each position."""
    ha = torch.einsum("bdhw,dk->bkhw", a, U)
    hc = torch.einsum("bdhw,dk->bkhw", c, U)
    B, D, H, W = ha.shape
    return (ha * hc).reshape(B, ns, D // ns, H, W).sum(2)


def get_args():
    p = argparse.ArgumentParser(description="Per-subspace relevance change caused by the synthetic watermark.")
    p.add_argument("--experiment", choices=["carton", "crate"], required=True)
    p.add_argument("--u_path", required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), default="resnet50")
    p.add_argument("--layer", choices=STAGES, default="layer3")
    p.add_argument("--num_subspaces", type=int, default=4)
    p.add_argument("--images", choices=["val", "natural_test"], default="val",
                   help="val: validation positives [0:150] (natural watermarks not annotated). natural_test: carton "
                        "test positives [800:1200], split by the natural-watermark labels in test-indices.txt.")
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--out", default=None, help="Output .json (default: <results>/drsa/<exp>_<arch>_<layer>/watermark_check.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    drsa_script = _load("compute_drsa_resnet", REPO_ROOT / "experiments" / "compute_drsa_resnet.py")
    rw = drsa_script._load_watermark_experiment()
    cfg = rw.EXPERIMENT_CONFIGS[args.experiment]
    if args.images == "natural_test":
        if args.experiment != "carton":
            raise ValueError("natural-watermark annotations exist only for carton")
        samples = rw.build_natural_test_samples(cfg["pos_dir"], cfg["neg_dir"],
                                                IMAGES_DIR / "carton_dugong" / "test-indices.txt", n_test=400)
        samples = [(p, wm) for p, cls, wm in samples if cls == 1]
    else:
        _, _, _, val_samples, _ = rw.build_splits(cfg["pos_dir"], cfg["neg_dir"])
        samples = [(p, None) for p, cls, wm in val_samples if cls == 1 and wm == 0]
    paths = [p for p, _ in samples]
    natural = [w for _, w in samples]
    target = drsa_script.IMAGENET_TARGETS[args.experiment]

    U = load_u_matrix(args.u_path).to(args.device)
    ns = args.num_subspaces
    model = ARCHS[args.arch](weights="IMAGENET1K_V1").eval().to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    composite = drsa_script.make_composite(args.layer, args.gamma, args.epsilon)
    rc, tt = drsa_script.make_transforms(rw)
    add_wm = AddWatermark(image_size=224)

    rel = {False: [], True: []}
    rel_in = {False: [], True: []}
    logits = {False: [], True: []}
    footprint = None
    for start in range(0, len(paths), args.batch_size):
        batch = paths[start:start + args.batch_size]
        for wm in (False, True):
            x = drsa_script.load_batch([(p, wm) for p in batch], rc, tt, add_wm).to(args.device)
            a, c = drsa_script.activation_and_context(model, args.layer, x, target, composite)
            with torch.no_grad():
                logits[wm].append(model(x)[:, target].cpu())
            maps = subspace_relevance_maps(a, c, U, ns)
            if footprint is None:
                footprint = watermark_footprint(add_wm, maps.shape[-2:]).to(maps.device)
            rel[wm].append(maps.sum((2, 3)).cpu())
            rel_in[wm].append((maps * footprint).sum((2, 3)).cpu())
        print(f"  [check] {min(start + args.batch_size, len(paths))}/{len(paths)} images", flush=True)

    R0, R1 = torch.cat(rel[False]), torch.cat(rel[True])
    I0, I1 = torch.cat(rel_in[False]), torch.cat(rel_in[True])
    L0, L1 = torch.cat(logits[False]), torch.cat(logits[True])

    print(f"\n{args.experiment}: {len(paths)} positives ({args.images}), {args.arch} {args.layer}, U = {args.u_path}")
    print(f"synthetic watermark footprint covers {(footprint > 0).sum().item()} of {footprint.numel()} "
          f"cells at {args.layer}")

    groups = {"unannotated": [True] * len(paths)} if args.images == "val" else {
        "naturally clean": [w == 0 for w in natural], "natural watermark": [w == 1 for w in natural]}
    report = {"u_path": str(args.u_path), "images": args.images, "target_logit": target, "gamma": args.gamma,
              "footprint_cells": int((footprint > 0).sum().item()), "groups": {}}
    for gname, sel in groups.items():
        sel = torch.tensor(sel)
        r0, r1, i0, i1 = R0[sel], R1[sel], I0[sel], I1[sel]
        d, d_in, dl = r1 - r0, i1 - i0, L1[sel] - L0[sel]
        n = d.shape[0]
        rows = [{
            "subspace": k,
            "mean_without_synthetic": r0[:, k].mean().item(),
            "mean_with_synthetic": r1[:, k].mean().item(),
            "mean_change": d[:, k].mean().item(),
            "stderr_change": (d[:, k].std() / n ** 0.5).item(),
            "frac_images_increased": (d[:, k] > 0).float().mean().item(),
            "mean_change_in_footprint": d_in[:, k].mean().item(),
            "mean_change_outside_footprint": (d[:, k] - d_in[:, k]).mean().item(),
        } for k in range(ns)]
        report["groups"][gname] = {"n": n, "logit_without_synthetic": L0[sel].mean().item(),
                                   "logit_with_synthetic": L1[sel].mean().item(), "subspaces": rows}
        print(f"\n== {gname}: {n} images; logit {target} {L0[sel].mean():.2f} -> {L1[sel].mean():.2f} with the "
              f"synthetic watermark ({dl.mean():+.2f}, up in {(dl > 0).float().mean() * 100:.0f}%)")
        print(f"{'k':>2} {'without':>9} {'with':>9} {'change':>9} {'± s.e.':>8} {'% up':>6} {'in fp':>9} {'outside':>9}")
        for r in rows:
            print(f"{r['subspace']:>2} {r['mean_without_synthetic']:9.3f} {r['mean_with_synthetic']:9.3f} "
                  f"{r['mean_change']:+9.3f} {r['stderr_change']:8.3f} {r['frac_images_increased'] * 100:6.0f} "
                  f"{r['mean_change_in_footprint']:+9.3f} {r['mean_change_outside_footprint']:+9.3f}")

    if args.images == "natural_test":
        nat = torch.tensor([w == 1 for w in natural])
        print("\n== natural watermark vs naturally clean (no synthetic watermark; unpaired, Welch s.e.)")
        print(f"logit {target}: {L0[nat].mean():.2f} vs {L0[~nat].mean():.2f}")
        diffs = []
        for k in range(ns):
            a, b = R0[nat, k], R0[~nat, k]
            diff = (a.mean() - b.mean()).item()
            se = (a.var() / len(a) + b.var() / len(b)).sqrt().item()
            diffs.append({"subspace": k, "natural_wm_mean": a.mean().item(), "clean_mean": b.mean().item(),
                          "difference": diff, "stderr": se})
            print(f"{k:>2} natural WM {a.mean():9.3f}   clean {b.mean():9.3f}   difference {diff:+9.3f} ± {se:.3f}")
        report["natural_vs_clean"] = diffs

    default_name = "watermark_check.json" if args.images == "val" else f"watermark_check_{args.images}.json"
    out = Path(args.out) if args.out else RESULTS_ROOT / "drsa" / f"{args.experiment}_{args.arch}_{args.layer}" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
