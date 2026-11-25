"""
resnet_lrp_subspaces.py

Per-subspace DRSA explanations for an ImageNet-pretrained ResNet with a CNP virtual layer
(the ResNet counterpart of carton_crate_lrp_subspaces.py). Writes two figures:

  <exp>_<arch>_<layer>_pixel_subspaces.png
      Columns: image | standard LRP heatmap | one heatmap per subspace. The forward pass is
      unchanged (the unablated virtual layer is an exact identity). Relevance is split by
      subspace at the virtual layer and propagated to the pixels separately
      (ncp.lrp_resnet.resnet_lrp_subspaces). The subspace heatmaps therefore sum to the
      standard one, and each row uses one color scale (red: evidence for the class,
      blue: against it).
  <exp>_<arch>_<layer>_layer_subspaces.png
      The same decomposition at the virtual layer itself: each subspace's relevance at every
      position of the stage output (14x14 after layer3), upsampled to 224x224 and laid over
      the image. It does not depend on the rules below the layer.

LRP rules
---------
Above the virtual layer (--upper_rule):
  lrp0 (default)  LRP-0 with biases, i.e. gradient x input. The context is the gradient,
                  as when the DRSA matrix was fit (compute_drsa_resnet.py, gamma = 0). So
                  the layer maps are exactly the (U_k^T a)(U_k^T c) that DRSA optimizes and
                  that drsa_watermark_subspace_check.py measures.
  alpha1beta0     the rules of ncp.lrp_resnet throughout (all-positive relevance; does not
                  match the context U was fit with).
Below it: alpha1-beta0, zB at the stem, positive-part split at residual additions
(ncp.lrp_resnet, also used for filter ranking). The DRSA paper does not state LRP rules for
its ResNet (it only says it used Zennit), so these are our choices.

Images
------
For carton, the defaults come from the natural-watermark annotations
(data/images/carton_dugong/test-indices.txt, test images [800:1018]):
  - 4 naturally clean images, each without and with the synthetic watermark;
  - 4 images with a natural watermark, as they are.
For crate there are no annotations: pass --clean_images (and --natural_images) by file name.
Panel titles give each subspace's relevance R at the virtual layer. With LRP-0 above the
layer, R can be negative, and the net total can be near zero, so shares are not shown.

Subspace indices are 0-indexed, as for --spurious_subspace.

Usage
-----
python analysis/resnet_lrp_subspaces.py --experiment carton \
    --u_path data/projection_matrices/U_carton_resnet50_layer3.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ncp.AugmentedVGG16 import load_u_matrix
from ncp.lrp import MEAN, STD
from ncp.lrp_resnet import resnet_lrp_forward, resnet_lrp_subspaces, virtual_layer_gradient
from ncp.paths import IMAGES_DIR, RESULTS_ROOT
from ncp.resnet import ARCHS, STAGES, AugmentedResNet
from ncp.watermark_transform import AddWatermark

EXPERIMENTS = {
    "carton": dict(target=478, img_dir=IMAGES_DIR / "carton_dugong" / "original" / "n02971356",
                   annotations=IMAGES_DIR / "carton_dugong" / "test-indices.txt"),
    "crate": dict(target=519, img_dir=IMAGES_DIR / "crate_packet" / "original" / "n03127925",
                  annotations=None),
}
ANNOTATED_RANGE = (800, 1018)   # carton test images covered by the natural test set (balanced against dugong)


def annotated_images(img_dir, annotations, n):
    """First n naturally clean and first n naturally watermarked carton test images."""
    wm = {int(line) for line in Path(annotations).read_text().splitlines()
          if line.strip() and not line.startswith("#")}
    files = sorted(f.name for f in Path(img_dir).iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    lo, hi = ANNOTATED_RANGE
    idx = {name: int(Path(name).stem.split("_")[-1]) for name in files}
    in_range = [f for f in files if lo <= idx[f] < hi]
    return [f for f in in_range if idx[f] not in wm][:n], [f for f in in_range if idx[f] in wm][:n]


def explain(model, x, target, subspace_dims, upper_rule):
    logits, ops = resnet_lrp_forward(model, x)
    if upper_rule == "lrp0":
        g = virtual_layer_gradient(model, x, target)
        R_std, R_sub, maps = resnet_lrp_subspaces(ops, None, subspace_dims, virtual_grad=g)
    else:
        R = torch.zeros_like(logits)
        R[:, target] = logits[:, target]
        R_std, R_sub, maps = resnet_lrp_subspaces(ops, R, subspace_dims)
    return logits[:, target], R_std.sum(1), R_sub.sum(2), maps   # channel-summed heatmaps


def _show_heatmap(ax, hm, vmax):
    ax.imshow(hm, cmap="bwr", vmin=-vmax, vmax=vmax)


def _show_overlay(ax, img, m, vmax):
    ax.imshow(np.asarray(img.convert("L")), cmap="gray", alpha=0.6)
    ax.imshow(m, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=0.6)


def _shares(maps):
    total = maps.sum()
    return (maps.sum((1, 2)) / total).numpy() if total.abs() > 1e-12 else np.full(maps.shape[0], np.nan)


def get_args():
    p = argparse.ArgumentParser(description="Per-subspace LRP heatmaps for a ResNet with a CNP virtual layer.")
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    p.add_argument("--u_path", required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), default="resnet50")
    p.add_argument("--layer", choices=STAGES, default="layer3")
    p.add_argument("--num_subspaces", type=int, default=4)
    p.add_argument("--upper_rule", choices=["lrp0", "alpha1beta0"], default="lrp0",
                   help="LRP rule above the virtual layer (see the module docstring).")
    p.add_argument("--n_images", type=int, default=4, help="Images per group when chosen from the annotations.")
    p.add_argument("--clean_images", nargs="+", default=None,
                   help="Watermark-free images (shown without and with the synthetic watermark).")
    p.add_argument("--natural_images", nargs="*", default=None, help="Images with a natural watermark.")
    p.add_argument("--out_dir", default=str(RESULTS_ROOT / "figures"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    cfg = EXPERIMENTS[args.experiment]
    target, K = cfg["target"], args.num_subspaces
    clean, natural = args.clean_images, args.natural_images
    if clean is None:
        if cfg["annotations"] is None:
            raise ValueError(f"no natural-watermark annotations for {args.experiment}: pass --clean_images")
        clean_default, natural_default = annotated_images(cfg["img_dir"], cfg["annotations"], args.n_images)
        clean = clean_default
        natural = natural_default if natural is None else natural
    natural = natural or []

    U = load_u_matrix(args.u_path)
    D = U.shape[0]
    dims = [D // K] * K
    model = AugmentedResNet(args.arch, U, U.t(), split_layer=args.layer).eval().to(args.device)

    rc = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    tt = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    add_wm = AddWatermark(image_size=224)

    def load(name):
        return rc(Image.open(Path(cfg["img_dir"]) / name).convert("RGB"))

    inputs = []   # (label, PIL image)
    for name in clean:
        img = load(name)
        inputs += [(f"{name}\nclean", img), (f"{name}\nclean + synthetic WM", add_wm(img))]
    for name in natural:
        inputs.append((f"{name}\nnatural WM", load(name)))

    rows = []   # (label, image, logit, std heatmap, subspace heatmaps, layer maps)
    for label, img in inputs:
        x = tt(img).unsqueeze(0).to(args.device)
        logit, std_hm, sub_hm, maps = explain(model, x, target, dims, args.upper_rule)
        rows.append((label, img, logit[0].item(), std_hm[0].cpu().numpy(),
                     sub_hm[:, 0].cpu().numpy(), maps[:, 0].cpu()))

    stem = f"{args.experiment}_{args.arch}_{args.layer}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_text = ("LRP-0 (gradient) above / α1β0 below the virtual layer" if args.upper_rule == "lrp0"
                 else "α1β0 throughout")

    summary = []
    for r in rows:
        label, _, logit, std_hm, sub_hm, maps = r
        summary.append({"image": label.replace("\n", " | "), "logit": logit,
                        "layer_relevance_total": maps.sum().item(),
                        "subspace_relevance": maps.sum((1, 2)).tolist(),
                        "subspace_share": _shares(maps).round(4).tolist(),
                        "max_abs_sum_error": float(np.abs(sub_hm.sum(0) - std_hm).max() / (np.abs(std_hm).max() + 1e-12))})

    def figure(kind):
        fig, axes = plt.subplots(len(rows), 2 + K, figsize=((2 + K) * 2.6, len(rows) * 2.7), squeeze=False)
        for i, (label, img, logit, std_hm, sub_hm, maps) in enumerate(rows):
            rel = maps.sum((1, 2)).numpy()
            axes[i, 0].imshow(img)
            axes[i, 0].set_ylabel(label, fontsize=7)
            if kind == "pixel":
                vmax = np.percentile(np.abs(std_hm), 99.5) + 1e-12
                _show_heatmap(axes[i, 1], std_hm, vmax)
                axes[i, 1].set_title(f"LRP (logit {logit:.1f}, layer R {rel.sum():+.1f})", fontsize=9)
                for k in range(K):
                    _show_heatmap(axes[i, 2 + k], sub_hm[k], vmax)
            else:
                up = F.interpolate(maps[:, None], size=(224, 224), mode="bilinear",
                                   align_corners=False)[:, 0].numpy()
                vmax = np.abs(up).max() + 1e-12
                _show_overlay(axes[i, 1], img, up.sum(0), vmax)
                axes[i, 1].set_title(f"all subspaces: R = {rel.sum():+.1f} ({maps.shape[1]}x{maps.shape[2]})", fontsize=9)
                for k in range(K):
                    _show_overlay(axes[i, 2 + k], img, up[k], vmax)
            for k in range(K):
                axes[i, 2 + k].set_title(f"subspace {k}: R = {rel[k]:+.1f}", fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        what = "per-subspace LRP heatmaps" if kind == "pixel" else f"subspace relevance at the virtual layer, upsampled"
        fig.suptitle(f"{args.experiment}, {args.arch}, virtual layer after {args.layer}: {what}\n"
                     f"{rule_text}; red = for class {target}, blue = against", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        path = out_dir / f"{stem}_{kind}_subspaces.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    pixel_path, layer_path = figure("pixel"), figure("layer")
    (out_dir / f"{stem}_subspaces.json").write_text(json.dumps(
        {"u_path": str(args.u_path), "target": target, "upper_rule": args.upper_rule, "rows": summary}, indent=1))

    print(f"{'image':54s} {'logit':>6} {'layer R':>8} " + " ".join(f"{'S' + str(k):>7}" for k in range(K)) + "  sum err")
    for s in summary:
        print(f"{s['image']:54s} {s['logit']:6.2f} {s['layer_relevance_total']:8.2f} "
              + " ".join(f"{v:+7.2f}" for v in s["subspace_relevance"]) + f"  {s['max_abs_sum_error']:.1e}")
    print(f"saved {pixel_path}\nsaved {layer_path}")


if __name__ == "__main__":
    main()
