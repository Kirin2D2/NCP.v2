Carton vs. dugong (paper Section 4.2.2, Figures 2-3)
=====================================================

Classes (ImageNet synsets)
  n02971356  carton  -> positive class (label 1)
  n02074367  dugong  -> negative class (label 0)

Expected layout (download with data/images/download_binary_dataset.py --task carton_dugong):
  data/images/carton_dugong/original/n02971356/n02971356_00000.jpeg ...
  data/images/carton_dugong/original/n02074367/n02074367_00000.jpeg ...

Splits (experiments/run_watermark_pruning_experiment.py, build_splits)
  Files are sorted by filename per class, so position i == file index i.
  val    [0:150]    per class, each image in watermarked (WM) and clean (NWM) form
  train  [150:800]  per class; positives alternate WM / NWM by index parity, negatives all NWM
                    (the synthetic hanzi watermark is a spurious cue for the positive class)
  rank   first 500 training positives (positive_only, used in the paper)
  test   [800:]     balanced across classes, each image in WM and NWM form
                    (selected with --eval_on_test)

Watermarks are synthetic and added on the fly (src/ncp/watermark_transform.py).

test-indices.txt lists carton images in [800:1200] that already contain a natural watermark;
it is only used by the optional --natural_test evaluation.

The DRSA projection matrix data/projection_matrices/U_carton_tensor.pt was learned from
500 carton images with DRSA (Chormai et al., 2024).
