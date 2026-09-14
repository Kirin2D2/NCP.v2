# Third-party notices

This repository includes or adapts code and assets from the projects below. Their licenses
apply to the corresponding files and may restrict how this repository can be licensed and used
(several are **non-commercial**).

| Component | Files in this repo | Source | License |
|---|---|---|---|
| Synthetic watermark transform | `src/ncp/watermark_transform.py` | [facebookresearch/Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole) (`imagenet_w`) | CC BY-NC 4.0 — [`licenses/Whac-A-Mole_CC-BY-NC-4.0.txt`](licenses/Whac-A-Mole_CC-BY-NC-4.0.txt) |
| LRP propagation rules | `src/ncp/lrp.py` (adapted) | [seulkiyeom/LRP_Pruning_toy_example](https://github.com/seulkiyeom/LRP_Pruning_toy_example) | CC BY-NC-SA 4.0 — [`licenses/LRP_Pruning_toy_example_LICENSE.txt`](licenses/LRP_Pruning_toy_example_LICENSE.txt) |
| Structural filter pruning, data loading | `src/ncp/prune_layer.py`, `src/ncp/data.py` (adapted) | [seulkiyeom/LRP_pruning](https://github.com/seulkiyeom/LRP_pruning) | No license file found in the upstream repository |
| DRSA virtual layer / projection matrices | `src/ncp/AugmentedVGG16.py` (reference implementation), `data/projection_matrices/*.pt` (computed with DRSA), `analysis/carton_crate_lrp_subspaces.py` (imports `cxai`, not redistributed) | [p16i/drsa-demo](https://github.com/p16i/drsa-demo), [p16i/disentangling-explanations](https://github.com/p16i/disentangling-explanations) | No license file found in `drsa-demo` |
| Source Han Serif SC ExtraLight font | `src/ncp/fonts/SourceHanSerifSC-ExtraLight.otf` | [adobe-fonts/source-han-serif](https://github.com/adobe-fonts/source-han-serif) | SIL Open Font License 1.1 — [`src/ncp/fonts/LICENSE.txt`](src/ncp/fonts/LICENSE.txt) |

## Data

ImageNet images are not included; they are downloaded by the user under the
[ImageNet terms of access](https://image-net.org/download.php).
`data/images/drsa_basketball_test_images/` contains four example images used for LRP heatmaps.
