# Concept-aware Network Pruning (CNP)

Check out our [VGG-16 demo notebook](https://colab.research.google.com/github/Kirin2D2/NCP.v2/blob/main/notebooks/cnp_vgg16_watermark_figures.ipynb) on Colab, and the [ResNet-50 notebook](notebooks/cnp_resnet_watermark_figures.ipynb) (experimental, not in the paper).

Code for **"Concept-Aware Pruning via Disentangled Subspaces for Robust Convolutional Networks"**
(Kirin Danek and Vikram V. Ramaswamy, XAI4CV Workshop at CVPR 2026).

📄 Paper: [`XAI4CV_paper.pdf`](XAI4CV_paper.pdf) · [workshop version](https://xai4cv-workshop.github.io/xai4cv2026/assets/papers/PP29.pdf)

Pruning methods can remove complicated but valid decision strategies while keeping easy, spurious
shortcuts. CNP augments a pretrained CNN with a *virtual concept layer* whose nodes correspond to
human-interpretable concepts (learned with DRSA), ablates the concepts identified as spurious, and then
applies LRP-based filter pruning, so filters that served the ablated concepts receive low relevance and
are pruned. On ImageNet binary tasks, CNP removes reliance on a spurious watermark while maintaining
or improving accuracy.

> **Naming.** The method was previously called "NCP". Code identifiers keep that name: the package is
> `ncp`, CNP runs use `--pruner ncp`, and outputs are written to `ncp/` directories and `ncp-2.pth`.

---

## Method → code

| Stage (paper §3) | What happens | Code |
|---|---|---|
| **Augment** | Insert frozen 1×1 convs `encode = Uᵀ`, `decode = U` after `conv4_3` of VGG-16. With orthogonal U the output is unchanged. | `src/ncp/AugmentedVGG16.py` (`AugmentedVGG16`) |
| **Ablate** | Suppress the block(s) of U belonging to spurious concept subspace(s). | `ablate_subspace_matrix` |
| **Prune** | Iteratively rank filters by LRP-α1β0 relevance on 500 positive-class images, remove the lowest 5% of filters globally, fine-tune (virtual layer bypassed), repeat until 80% are pruned. Then remove the virtual layer and fine-tune once more. | `src/ncp/prune_vgg.py` (`PruningFineTuner`, `AugmentedVGGAdapter`), `src/ncp/lrp.py`, `src/ncp/prune_layer.py` |

Vanilla LRP pruning (the baseline) is the same loop on a plain VGG-16 (`VanillaVGGAdapter`, `--pruner vanilla`).

### Implementation notes
- **Subspace indices are 0-indexed in code; the paper numbers them from 1.** Carton ablates paper
  subspace 4 → `--spurious_subspace 3`; crate ablates paper subspace 2 → `--spurious_subspace 1`;
  basketball ablates paper subspace 4 → `--irrelevant_subspaces 3`.
- **Ablation scales the selected blocks of U and Uᵀ by 1e-4 rather than setting them to exactly 0.**
  This keeps LRP denominators finite and gradients non-degenerate.
- **Filter ranking seeds relevance at the ground-truth label.** The ranking set contains only positive-class
  images, so this equals ranking with respect to the positive logit.
- **`conv4_3`, the virtual layer, and the reconstruction layer are never pruned.** This avoids recomputing
  concept subspaces between pruning iterations.
- **DRSA:** K = 4 subspaces × 128 dimensions at `conv4_3`, learned from 500 positive-class images with
  [DRSA](https://github.com/p16i/drsa-demo). The resulting matrices are in `data/projection_matrices/`.

### Selected hyperparameters (paper Table 1)
| Hyperparameter | Value |
|---|---|
| Subspaces K × dim | 4 × 128 at `conv4_3` |
| LRP rule | α1β0 (`first` rule at the input layer) |
| Filter-ranking data | 500 positive-class images (`--rank_loader_type positive_only`) |
| Pruning step / total | 5% / 80% of filters |
| Fine-tuning epochs per step / final | 2 / 5 |
| Head warmup | 15 epochs |
| Optimizer | SGD, lr 1e-4, momentum 0.9 (warmup: Adam, lr 1e-4) |

---

## Installation

```bash
git clone https://github.com/Kirin2D2/NCP.v2.git
cd NCP.v2
# install PyTorch for your platform first (https://pytorch.org), then:
pip install -e ".[dev,data]"
pytest            # CPU unit tests (~1-2 min), no data or pretrained weights needed
pytest --runslow  # also a ~15 min CPU end-to-end run of the watermark pipeline on noise images
```

## Data

ImageNet images are not redistributed. Download them from image-net.org, which requires an account that
has accepted the ImageNet terms of access:

```bash
python data/images/download_binary_dataset.py --task carton_dugong   # carton n02971356, dugong n02074367
python data/images/download_binary_dataset.py --task crate_packet    # crate n03127925, packet n03871628
python data/images/download_binary_dataset.py --task basketball      # basketball n02802426
python data/images/download_random_images.py                         # basketball negatives
```

Images go to `data/images/…` by default. Set `NCP_DATA_ROOT` (and `NCP_RESULTS_ROOT` for outputs) to use
other locations. Splits are defined on sorted filename order; see
[`data/images/carton_dugong/readme.txt`](data/images/carton_dugong/readme.txt).

**Caveats:** the crate/packet runs in the paper used ILSVRC-style filenames (e.g. `n03127925_97.JPEG`), and
the exact basketball negative images were not recorded. Fresh downloads therefore may not reproduce those
splits exactly.

## Reproducing the paper

### Watermark robustness (§4.2.2, Figures 3–4)
```bash
scripts/reproduce_watermark.sh                 # {carton, crate} × {CNP, vanilla} × seeds 0–4, then plots
scripts/reproduce_watermark.sh carton ncp 0    # a single run
EXTRA_ARGS="--save_model" scripts/reproduce_watermark.sh   # also save pruned checkpoints
```
Each run writes `results/watermark_experiment/{experiment}/{ncp|vanilla}/seed{N}/stats.pt`.
Every evaluation (before pruning, after each pruning step, and after final fine-tuning) uses the split
selected by `--eval_on_test`. The reproduction scripts pass it, so curves are computed on the held-out
test split. The choice is recorded in each `stats.pt`.
An example SLURM array is in [`scripts/slurm/watermark_pruning.slurm`](scripts/slurm/watermark_pruning.slurm).

**Colab notebook (VGG-16):** [`notebooks/cnp_vgg16_watermark_figures.ipynb`](notebooks/cnp_vgg16_watermark_figures.ipynb)
runs the paper configuration on a Colab GPU, pruning to 40% by default for speed
(set `TOTAL_PR = 0.80` for the full schedule). It downloads the data, runs CNP and vanilla pruning per seed
(resumable, with optional storage in Google Drive), and plots Figures 3–4: overall and worst-subgroup (c0w1)
accuracy vs. filters pruned.

### Basketball demonstration (§4.2.1)
```bash
scripts/reproduce_basketball.sh   # CNP (ablate ball subspace) and vanilla to 80%, then LRP comparison figure
```

### Figures
| Figure | Script |
|---|---|
| Fig. 1 (basketball subspaces + CNP-pruned LRP) | not yet included; `analysis/basketball_lrp_comparison.py` produces a related LRP comparison |
| Figs. 2, 5 (carton / crate DRSA subspaces) | `python analysis/carton_crate_lrp_subspaces.py --drsa_demo_dir <path to drsa-demo>` |
| Figs. 3, 4 top (overall accuracy) | `python analysis/plot_watermark_results.py --no_titles` |
| Figs. 3, 4 bottom (dugong/packet images with watermarks, c0w1) | `python analysis/plot_watermark_results.py --no_titles --c0w1_only` |

Figures 2 and 5 need the `cxai` package from [p16i/drsa-demo](https://github.com/p16i/drsa-demo), which is not
pip-installable. Clone it and pass `--drsa_demo_dir` or add it to `PYTHONPATH`.

## ResNet backbones (experimental, not in the paper)

The watermark experiment also runs CNP on torchvision ResNet-18/34/50 (`--arch`).

- **Virtual layer placement.** The layer is inserted after the last block of a stage (`--virtual_layer`, default `layer3`).
  Inside a residual block, the skip connection would carry information around the layer. Between blocks, the residual
  stream is a single tensor, so every path to the output passes through the virtual layer. After `layer3`,
  D = 256 (ResNet-18/34) or 1024 (ResNet-50).
- **What gets pruned.** Only channels internal to a block: `conv1` in basic blocks, `conv1`/`conv2` in bottlenecks.
  Residual-stream channels are shared across a stage through the additions and fix D, so they stay intact.
  This is the ResNet analogue of keeping `conv4_3` in VGG-16.
- **BatchNorm.** Before the head warmup, BatchNorm running statistics are re-estimated on the training set
  (`--warmup_bn reestimate`, the default) and then kept frozen while the head trains. This removes a jump that
  appears otherwise: with ImageNet statistics, c0w1 starts near 51% and rises to about 95% after the first
  fine-tune round, only because that round adapts BatchNorm. With re-estimated statistics, the unpruned model
  is already at about 93% before fine-tuning. VGG-16 has no BatchNorm and is unaffected.
- **LRP.** Conv+BatchNorm pairs are fused and use LRP-α1β0 (zB at the stem). Residual additions split relevance in
  proportion to each path's positive contribution. Global average pooling redistributes relevance in
  proportion to the activations. See `src/ncp/lrp_resnet.py`.
- **DRSA matrix.** `experiments/compute_drsa_resnet.py` learns a D×D DRSA matrix at the chosen stage of the
  ImageNet-pretrained ResNet. It follows the paper's setup:
  - 500 positive training images, 20 random locations each.
  - LRP context from [Zennit](https://github.com/chr5tphr/zennit) (BatchNorm folded into the convs; z-rule at residual
    additions), with γ = 0 above the layer as for VGG-16 above `conv4_3`.
  - Normalization from Eq. E.1, then `cxai.drsa.optimize` with 5000 iterations and the best of 3 restarts.
  - Subspaces sorted by relevance.

  It needs `pip install -e ".[drsa]"` and a clone of [drsa-demo](https://github.com/p16i/drsa-demo) next to this repo
  (or `--drsa_demo_dir`). Run `--check` to verify that the γ = 0 context equals the gradient. The committed
  matrix `data/projection_matrices/U_carton_resnet50_layer3.pt` (4 subspaces of 256 dims) has the watermark in
  subspace 3 (0-indexed; subspace 4 in the paper's 1-indexed convention). For a new matrix, identify the
  watermark subspace with `analysis/drsa_watermark_subspace_check.py` before choosing `--spurious_subspace`.

```bash
python experiments/compute_drsa_resnet.py --experiment carton        # -> data/projection_matrices/U_carton_resnet50_layer3.pt
python experiments/run_watermark_pruning_experiment.py --experiment carton --pruner ncp \
    --arch resnet50 --virtual_layer layer3 --u_filepath data/projection_matrices/U_carton_resnet50_layer3.pt \
    --spurious_subspace 3 --pr_step 0.05 --total_pr 0.80 --eval_on_test
python experiments/run_watermark_pruning_experiment.py --experiment carton --pruner vanilla \
    --arch resnet50 --virtual_layer layer3 --pr_step 0.05 --total_pr 0.80 --eval_on_test
```
Results are written to `results/watermark_experiment/carton_resnet50_layer3/{ncp,vanilla}/seed{N}/`. Two controls
prune nothing: `--pruner finetune_only` gives the same fine-tune rounds, and `--pruner finetune_ablated` gives the
same rounds through the ablated virtual layer.

**ResNet-50 notebook:** [`notebooks/cnp_resnet_watermark_figures.ipynb`](notebooks/cnp_resnet_watermark_figures.ipynb)
covers the DRSA matrix, the watermark-subspace check, per-subspace heatmaps, CNP vs vanilla pruning, and an appendix of controls.
`python scripts/download_assets.py carton_resnet50_layer3 carton_resnet50_layer3_controls` fetches the runs it plots.

**Result (carton, ResNet-50 `layer3`, 5 seeds, pruned to 80% of block-internal channels).** The unpruned
model classifies watermarked dugongs (c0w1) at 93.9 ± 1.6%. CNP stays at that level throughout pruning
(95.5 ± 0.9% after the final fine-tune). Vanilla pruning drifts to 57.1 ± 4.4%, and CNP is higher at every
pruning step for every seed. Caveats:
- About 27% of the annotated carton test images carry natural watermarks; the train/val splits are not annotated.
- The pruning ratio counts only block-internal channels. The residual stream is not pruned.
- The ImageNet-pretrained ResNet-50 already relies on the watermark. With the 1000-class head, watermarked
  dugongs score higher on dugong than on carton only 58.7% of the time (99.1% for VGG-16), and top-1 accuracy
  on dugong drops from 98.6% to 48.2% when the watermark is added.

## Pruned checkpoints

Pruned models are saved as a `state_dict` plus per-layer channel counts, and load as standard
torchvision VGG-16 or ResNet models on CPU or GPU:

```python
from ncp.checkpoint import load_pruned_checkpoint
model, info = load_pruned_checkpoint("carton/ncp/seed0/ncp.pt")   # CNP and vanilla checkpoints alike
print(info["channels"], info["meta"])
```

Checkpoints and `stats.pt` files are published as release assets and can be fetched with
`python scripts/download_assets.py --all` (`--list` shows what is available). So far this covers the
ResNet-50 runs; the paper's VGG-16 runs will follow.

## Repository layout
```
src/ncp/            AugmentedVGG16 / AugmentedResNet, LRP rules, structural pruning, prune/fine-tune loop, checkpoints
experiments/        run_watermark_pruning_experiment.py (carton, crate), run_PFT_basketball.py
analysis/           figure scripts
scripts/            reproduction scripts (+ example SLURM array)
data/images/        download scripts, split notes, 4 basketball example images
data/projection_matrices/  DRSA projection matrices U (carton, crate, basketball)
tests/              CPU unit tests (pruning surgery, LRP, ranking, splits, checkpoints, ResNet)
```

## Citation
```bibtex
@inproceedings{danek2026cnp,
  title     = {Concept-Aware Pruning via Disentangled Subspaces for Robust Convolutional Networks},
  author    = {Danek, Kirin and Ramaswamy, Vikram V.},
  booktitle = {XAI4CV Workshop at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (non-archival)},
  year      = {2026}
}
```

## Acknowledgements and licenses
This code builds on [LRP_pruning](https://github.com/seulkiyeom/LRP_pruning) /
[LRP_Pruning_toy_example](https://github.com/seulkiyeom/LRP_Pruning_toy_example) (Yeom et al.),
[drsa-demo](https://github.com/p16i/drsa-demo) / [disentangling-explanations](https://github.com/p16i/disentangling-explanations)
(Chormai et al.), [Zennit](https://github.com/chr5tphr/zennit) (optional, for ResNet DRSA matrices; LGPL-3.0, not
redistributed), and [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole) (watermark transform).
Some of these components are licensed for **non-commercial use only**. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
