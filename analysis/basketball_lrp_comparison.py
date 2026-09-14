"""
basketball_lrp_comparison.py

Comparison figure: N test images × 4 columns
  col 0 — original image
  col 1 — vanilla ImageNet VGG16 LRP heatmap   (w.r.t. logit 430, ImageNet basketball)
  col 2 — CNP pruned VGG16 LRP heatmap         (w.r.t. logit 0, binary basketball class)
  col 3 — vanilla LRP pruned VGG16 LRP heatmap (w.r.t. logit 0, binary basketball class)

All use LRP-alpha1-beta0 throughout, with the 'first' rule at the pixel layer.
avgpool in VGG16 is identity for 224px inputs and is skipped in the backward LRP chain;
the reshape from flat (1, 25088) to (1, C, 7, 7) is handled by lrp.Pooling reading
pool5.output.shape.

Pruned models are loaded with ncp.checkpoint.load_pruned_checkpoint; both CNP and vanilla
checkpoints load as torchvision-layout VGG16 models.

Usage
-----
python analysis/basketball_lrp_comparison.py \\
    --cnp_checkpoint results/basketball/ncp_ss3/ncp-2.pth \\
    --vanilla_checkpoint results/basketball/vanilla/van.pth \\
    --out results/basketball/lrp_comparison.png
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ncp.checkpoint import load_pruned_checkpoint
from ncp.lrp import lrp
from ncp.paths import IMAGES_DIR, RESULTS_ROOT

IMAGENET_BBALL = 430   # ImageNet class index for basketball
BINARY_BBALL   = 0     # binary class index in the pruned models ('basketball' < 'not_basketball')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

input_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
rc_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])


# ---------------------------------------------------------------------------
# Forward hook
# ---------------------------------------------------------------------------

def fhook(module, input, output):
    module.input  = input[0]
    module.output = output.data


# ---------------------------------------------------------------------------
# LRP (alpha-1 beta-0)
# ---------------------------------------------------------------------------

def lrp_vgg16(model, x_tensor, target_class):
    """LRP heatmap (3, H, W) for a torchvision-layout VGG16 (ImageNet or pruned binary)."""
    feature_layers    = list(model.features)
    classifier_layers = list(model.classifier)

    handles = [m.register_forward_hook(fhook) for m in feature_layers + classifier_layers]
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.unsqueeze(0))

    R = torch.zeros_like(out)
    R[0, target_class] = 1.0

    # Backward: classifier reversed, then features reversed; 'first' rule at conv1_1
    for m in list(reversed(classifier_layers)) + list(reversed(feature_layers)):
        if m is feature_layers[0]:
            R = lrp(m, R, lrp_var='first', param=None)
        else:
            R = lrp(m, R, lrp_var='alpha', param=1)

    for h in handles:
        h.remove()
    return R.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Viz
# ---------------------------------------------------------------------------

def show_image(ax, pil_img, ylabel=None, title=None):
    ax.imshow(rc_transform(pil_img))
    ax.set_xticks([]); ax.set_yticks([])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title,   fontsize=9)


def show_heatmap(ax, heatmap_2d, logit=None, title=None):
    b = np.abs(heatmap_2d).max() or 1.0
    cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    cmap[:, :3] *= 0.85
    ax.imshow(heatmap_2d, cmap=ListedColormap(cmap), vmin=-b, vmax=b)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)
    xlabel = f"$\\sum R_i$={heatmap_2d.sum():.3f}"
    if logit is not None:
        xlabel += f"  logit={logit:.3f}"
    ax.set_xlabel(xlabel, fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnp_checkpoint", type=Path,
                   default=RESULTS_ROOT / "basketball" / "ncp_ss3" / "ncp-2.pth")
    p.add_argument("--vanilla_checkpoint", type=Path,
                   default=RESULTS_ROOT / "basketball" / "vanilla" / "van.pth")
    p.add_argument("--img_dir", type=Path, default=IMAGES_DIR / "drsa_basketball_test_images")
    p.add_argument("--out", type=Path, default=RESULTS_ROOT / "basketball" / "lrp_comparison.png")
    args = p.parse_args()

    print(f"[device] {DEVICE}")

    img_paths = sorted(
        glob.glob(str(args.img_dir / "*.jpg"))
        + glob.glob(str(args.img_dir / "*.png"))
        + glob.glob(str(args.img_dir / "*.JPEG"))
    )
    assert img_paths, f"No test images found in {args.img_dir}"

    print("[loading] vanilla ImageNet VGG16")
    vanilla = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    ).to(DEVICE).eval()

    print(f"[loading] CNP pruned model from {args.cnp_checkpoint}")
    cnp, cnp_info = load_pruned_checkpoint(args.cnp_checkpoint)
    cnp = cnp.to(DEVICE)
    ablated = "+".join(str(s) for s in cnp_info["meta"].get("irrelevant_subspaces", [])) or "?"

    print(f"[loading] vanilla pruned model from {args.vanilla_checkpoint}")
    van_pruned, _ = load_pruned_checkpoint(args.vanilla_checkpoint)
    van_pruned = van_pruned.to(DEVICE)

    for m in (vanilla, cnp, van_pruned):
        for param in m.parameters():
            param.requires_grad_(False)

    n = len(img_paths)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.25 * n), squeeze=False)

    col_titles = [
        "Image",
        f"VGG16 ImageNet\nLRP (logit {IMAGENET_BBALL})",
        f"CNP pruned (subspace {ablated}, 0-indexed)\nLRP (basketball logit)",
        "Vanilla pruned\nLRP (basketball logit)",
    ]

    for row, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB")
        x   = input_transform(img).to(DEVICE)

        hm_van = lrp_vgg16(vanilla, x, IMAGENET_BBALL).sum(axis=0)
        logit_van = float(vanilla(x.unsqueeze(0)).squeeze()[IMAGENET_BBALL])

        hm_cnp = lrp_vgg16(cnp, x, BINARY_BBALL).sum(axis=0)
        logit_cnp = float(cnp(x.unsqueeze(0)).squeeze()[BINARY_BBALL])

        hm_van_pruned = lrp_vgg16(van_pruned, x, BINARY_BBALL).sum(axis=0)
        logit_van_pruned = float(van_pruned(x.unsqueeze(0)).squeeze()[BINARY_BBALL])

        print(f"img-{row}: imagenet logit={logit_van:.3f}  cnp logit={logit_cnp:.3f}  "
              f"van_pruned logit={logit_van_pruned:.3f}  "
              f"imagenet_sum={hm_van.sum():.3f}  cnp_sum={hm_cnp.sum():.3f}  "
              f"van_pruned_sum={hm_van_pruned.sum():.3f}")

        title_row = col_titles if row == 0 else [None, None, None, None]

        show_image(axes[row, 0],   img,           ylabel=f"img-{row}", title=title_row[0])
        show_heatmap(axes[row, 1], hm_van,        logit=logit_van,        title=title_row[1])
        show_heatmap(axes[row, 2], hm_cnp,        logit=logit_cnp,        title=title_row[2])
        show_heatmap(axes[row, 3], hm_van_pruned, logit=logit_van_pruned, title=title_row[3])

    fig.suptitle("LRP-α1β0 heatmaps: VGG16 ImageNet vs CNP pruned vs vanilla LRP pruned", fontsize=11)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
