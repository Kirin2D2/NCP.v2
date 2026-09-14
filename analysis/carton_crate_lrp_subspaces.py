"""
carton_crate_lrp_subspaces.py

Generates the DRSA subspace heatmap montages for the watermark experiments
(paper Figure 2: carton, Figure 5: crate) using vanilla ImageNet VGG16.

Figure 2 — carton (ImageNet logit 478):
  carton images n02971356_0000{0,3,4,5}.jpeg (natural watermarks), U_carton_tensor.pt
Figure 5 — crate (ImageNet logit 519):
  crate images n03127925_{97,893,895,923}.JPEG with the synthetic hanzi watermark
  applied, U_crate_tensor.pt

Layout per figure: N rows x (2 + K) columns
  col 0        — original image
  col 1        — standard LRP heatmap (sum over colour channels)
  col 2..(2+K) — per-subspace LRP heatmaps (canonical order 0..K-1)

Column labels follow the paper's 1-indexed subspace numbering.

Requirements
------------
Needs the `cxai` package from https://github.com/p16i/drsa-demo, which is not pip-installable:
clone it and either add it to PYTHONPATH or pass --drsa_demo_dir.

Usage
-----
python analysis/carton_crate_lrp_subspaces.py --drsa_demo_dir ../drsa-demo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ncp.paths import IMAGES_DIR, PROJECTION_DIR, RESULTS_ROOT
from ncp.watermark_transform import AddWatermark

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAYER   = "conv4_3"
K       = 4
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

CARTON_IMG_NAMES = [
    "n02971356_00000.jpeg",
    "n02971356_00003.jpeg",
    "n02971356_00004.jpeg",
    "n02971356_00005.jpeg",
]
CARTON_TARGET = 478   # ImageNet "carton"

CRATE_IMG_NAMES = [
    "n03127925_97.JPEG",
    "n03127925_893.JPEG",
    "n03127925_895.JPEG",
    "n03127925_923.JPEG",
]
CRATE_TARGET = 519    # ImageNet "crate"


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def add_vgg16_aliases(model):
    feat = list(model.features)
    block_conv_counts = [2, 2, 3, 3, 3]
    ptr = 0
    for b, n_convs in enumerate(block_conv_counts, start=1):
        for c in range(1, n_convs + 1):
            object.__setattr__(model.features, f"conv{b}_{c}", feat[ptr])
            if ptr + 1 < len(feat):
                object.__setattr__(model.features, f"relu{b}_{c}", feat[ptr + 1])
            ptr += 2
        object.__setattr__(model.features, f"pool{b}", feat[ptr])
        ptr += 1
    return model


def load_model(constants, models):
    model = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    )
    model = add_vgg16_aliases(model)

    rc_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
    to_input = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(constants.IMAGENET_MEAN),
                    std=torch.tensor(constants.IMAGENET_STD)),
    ])
    input_transform = T.Compose([rc_transform, to_input])

    # "Canonical" transform stored on the model (required by VGGLRPExplainer).
    setattr(model, models.ATTRIBUTE_TRANSFORMATION, (rc_transform, input_transform))
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, rc_transform, to_input


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_montage(explainer, putils, gb_inspector, images, img_names, input_transform,
                 target_class, out_path, subspace_labels, fs_title=22, fs_ylabel=16,
                 show_ylabels=True):
    """images: list of 224×224 PIL images (already cropped / watermarked)."""
    nrows = len(images)
    ncols = 2 + K
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2), squeeze=False)

    for row_idx, (img, name) in enumerate(zip(images, img_names)):
        x = input_transform(img).to(DEVICE)

        logits, standard_heatmap, info = explainer.explain_with_inspector(
            x, target_class, inspector=gb_inspector, top_k=K,
        )

        # Reorder subspace heatmaps to canonical 0..K-1 order
        canonical_order = np.argsort(info.top_k_sources)
        sub_hm = info.input_top_k_source_heatmaps[canonical_order].sum(axis=1)  # (K, H, W)
        std_hm = standard_heatmap.sum(axis=0)                                   # (H, W)

        plt.sca(axes[row_idx, 0])
        putils.viz.imshow(img)
        if show_ylabels:
            axes[row_idx, 0].set_ylabel(name, fontsize=fs_ylabel)
        axes[row_idx, 0].set_xlabel('')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Image", fontsize=fs_title)

        plt.sca(axes[row_idx, 1])
        putils.viz.heatmap(std_hm, title="")
        axes[row_idx, 1].set_xlabel('')
        if row_idx == 0:
            axes[row_idx, 1].set_title("LRP", fontsize=fs_title)

        # Per-subspace columns — normalise to the same scale as the full LRP heatmap
        for s in range(K):
            plt.sca(axes[row_idx, 2 + s])
            putils.viz.heatmap(sub_hm[s], title="", reference_heatmap=std_hm)
            axes[row_idx, 2 + s].set_xlabel('')
            if row_idx == 0:
                axes[row_idx, 2 + s].set_title(subspace_labels[s], fontsize=fs_title)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drsa_demo_dir", type=Path, default=None,
                   help="Path to a clone of p16i/drsa-demo (provides the cxai package).")
    p.add_argument("--carton_dir", type=Path,
                   default=IMAGES_DIR / "carton_dugong" / "original" / "n02971356")
    p.add_argument("--crate_dir", type=Path,
                   default=IMAGES_DIR / "crate_packet" / "original" / "n03127925")
    p.add_argument("--u_dir", type=Path, default=PROJECTION_DIR)
    p.add_argument("--out_dir", type=Path, default=RESULTS_ROOT / "figures")
    p.add_argument("--only", choices=["carton", "crate"], default=None)
    args = p.parse_args()

    if args.drsa_demo_dir is not None:
        sys.path.insert(0, str(args.drsa_demo_dir))
    try:
        from cxai import factory, inspector, constants, models
        from cxai import utils as putils
    except ImportError as e:
        sys.exit(f"Could not import cxai ({e}). Clone https://github.com/p16i/drsa-demo and "
                 f"pass --drsa_demo_dir or add it to PYTHONPATH.")

    print(f"[device] {DEVICE}")
    model, rc_transform, to_input = load_model(constants, models)
    explainer = factory.make_explainer("lrp", model)
    add_wm = AddWatermark(image_size=224)

    configs = {
        "carton": dict(
            img_dir=args.carton_dir, names=CARTON_IMG_NAMES, target=CARTON_TARGET,
            u_path=args.u_dir / "U_carton_tensor.pt",
            out_path=args.out_dir / "carton_lrp_subspaces.png",   # Figure 2
            watermark=False,
            subspace_labels=["1", "2", "3: Writing on box", "4: Watermark"],
            show_ylabels=True,
        ),
        "crate": dict(
            img_dir=args.crate_dir, names=CRATE_IMG_NAMES, target=CRATE_TARGET,
            u_path=args.u_dir / "U_crate_tensor.pt",
            out_path=args.out_dir / "crate_lrp_subspaces.png",    # Figure 5
            watermark=True,
            subspace_labels=["1", "2: Watermark", "3", "4: Writing on box"],
            show_ylabels=False,
        ),
    }

    for name, cfg in configs.items():
        if args.only and name != args.only:
            continue
        print(f"\n[{name}] target={cfg['target']}  U={cfg['u_path'].name}")

        images = []
        for fname in cfg["names"]:
            img = putils.load_image(str(cfg["img_dir"] / fname))
            img = rc_transform(img)
            images.append(add_wm(img) if cfg["watermark"] else img)

        U_raw = torch.load(str(cfg["u_path"]), map_location="cpu", weights_only=True)
        assert U_raw.shape == (512, 512), f"Unexpected U shape: {U_raw.shape}"
        U = U_raw.reshape(512, K, U_raw.shape[1] // K).float().to(DEVICE)

        gb_inspector = inspector.GroupBasisInspector(layer=LAYER, weights=U).to(DEVICE)

        make_montage(explainer, putils, gb_inspector, images, cfg["names"], to_input,
                     cfg["target"], cfg["out_path"], cfg["subspace_labels"],
                     show_ylabels=cfg["show_ylabels"])


if __name__ == "__main__":
    main()
