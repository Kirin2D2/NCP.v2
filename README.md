# Concept-aware Network Pruning (CNP)

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
git clone https://github.com/KirinDanek/NCP.v2.git
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

**Colab notebook:** [`notebooks/cnp_watermark_figures.ipynb`](notebooks/cnp_watermark_figures.ipynb)
runs the same configuration on a Colab GPU. It downloads the data, runs CNP and vanilla pruning per seed
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

## Pruned checkpoints

Pruned models are saved as a `state_dict` plus per-layer channel counts, and load as standard
torchvision VGG-16 models on CPU or GPU:

```python
from ncp.checkpoint import load_pruned_checkpoint
model, info = load_pruned_checkpoint("carton/ncp/seed0/ncp.pt")   # CNP and vanilla checkpoints alike
print(info["channels"], info["meta"])
```

Checkpoints and `stats.pt` files for the paper runs will be published as release assets and can be
fetched with `python scripts/download_assets.py --all`.

## Repository layout
```
src/ncp/            AugmentedVGG16, LRP rules, structural pruning, prune/fine-tune loop, checkpoints
experiments/        run_watermark_pruning_experiment.py (carton, crate), run_PFT_basketball.py
analysis/           figure scripts
scripts/            reproduction scripts (+ example SLURM array)
data/images/        download scripts, split notes, 4 basketball example images
data/projection_matrices/  DRSA projection matrices U (carton, crate, basketball)
tests/              CPU unit tests (ablation, checkpoint round-trip)
```

## Citation
```bibtex
@inproceedings{danek2026cnp,
  title     = {Concept-Aware Pruning via Disentangled Subspaces for Robust Convolutional Networks},
  author    = {Danek, Kirin and Ramaswamy, Vikram V.},
  booktitle = {XAI4CV Workshop at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (non-archival},
  year      = {2026}
}
```

## Acknowledgements and licenses
This code builds on [LRP_pruning](https://github.com/seulkiyeom/LRP_pruning) /
[LRP_Pruning_toy_example](https://github.com/seulkiyeom/LRP_Pruning_toy_example) (Yeom et al.),
[drsa-demo](https://github.com/p16i/drsa-demo) / [disentangling-explanations](https://github.com/p16i/disentangling-explanations)
(Chormai et al.), and [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole) (watermark transform).
Some of these components are licensed for **non-commercial use only**. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
