"""
compute_drsa_resnet.py

Learns a DRSA projection matrix U (Chormai et al., TPAMI 2024, arXiv:2212.14855) at a stage
boundary of an ImageNet-pretrained torchvision ResNet, for use as the CNP virtual layer
(run_watermark_pruning_experiment.py --arch resnet50 --u_filepath ...).

Pipeline (paper Sec. 5 and Supplementary Note E)
------------------------------------------------
1. Run 500 positive-class training images through the pretrained ResNet (no virtual layer,
   no fine-tuning) and take the activation at the output of `--layer` (default layer3, after
   the last block's ReLU). This is a stage boundary, so no skip connection bypasses it.
2. Compute the LRP context vector c = R / a at that layer w.r.t. the ImageNet logit of the
   positive class, with Zennit (ResNetCanonizer: BatchNorm folded into convs, residual
   additions made explicit as Sum modules). Only the rules *above* the layer matter for c.
   Default: gamma = 0 for the convs above the layer, epsilon for fc, Norm (z-rule) for the
   residual sums and average pooling. This mirrors the paper's VGG16 setup, where every layer
   above conv4_3 uses gamma = 0. With gamma = 0 the context equals the plain gradient,
   which --check verifies.
3. Keep 20 random spatial locations per image (10,000 (a, c) pairs). Then normalize as in
   Eq. (E.1): x / (D^(1/4) sqrt(E[x^2])). This form reproduces the objective stored with the
   authors' published basketball matrix. Optimize the DRSA objective with
   cxai.drsa.optimize (5000 iterations, best of 3 random orthogonal restarts). Sort the K
   blocks by mean subspace relevance, highest first.

Output: a D x D float32 tensor whose column blocks are the subspaces, in the same layout as
data/projection_matrices/U_*_tensor.pt. Next to it, a .json file records the settings, image
list, objective and per-subspace relevance. Activations and contexts are cached in --cache_dir.

Images: the first `--n_images` positives of the experiment's train pool, which is the same
pool build_splits() uses for filter ranking. Validation and test images are never used. As
for the paper's VGG16 matrices, carton images are used as-is (natural watermarks only). For
crate, a fraction (--wm_fraction, default 0.25) gets the synthetic hanzi watermark; it is
applied to every 4th image.

Requires `zennit` and the `cxai` package from https://github.com/p16i/drsa-demo (not
pip-installable: clone it next to this repo or pass --drsa_demo_dir). cxai.drsa only uses
nptyping for a type hint. If nptyping is missing, the script stubs it: nptyping pins
numpy<2.

Which subspace holds the watermark has to be identified afterwards, e.g. from per-subspace
heatmaps, before passing --spurious_subspace (0-indexed) to the pruning experiment.

Usage
-----
python experiments/compute_drsa_resnet.py --experiment carton
python experiments/compute_drsa_resnet.py --experiment crate --arch resnet50 --layer layer3
python experiments/compute_drsa_resnet.py --experiment carton --check   # context sanity check only
"""

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ncp.paths import PROJECTION_DIR, REPO_ROOT, RESULTS_ROOT
from ncp.resnet import ARCHS, STAGE_WIDTHS, STAGES
from ncp.watermark_transform import AddWatermark

EXPERIMENTS_DIR = Path(__file__).resolve().parent
IMAGENET_TARGETS = {"carton": 478, "crate": 519}   # ImageNet logits of the positive classes
DEFAULT_WM_FRACTION = {"carton": 0.0, "crate": 0.25}


def _load_watermark_experiment():
    """Import run_watermark_pruning_experiment.py for its EXPERIMENT_CONFIGS and build_splits."""
    path = EXPERIMENTS_DIR / "run_watermark_pruning_experiment.py"
    spec = importlib.util.spec_from_file_location("run_watermark_pruning_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_cxai_drsa(drsa_demo_dir):
    if drsa_demo_dir is not None:
        sys.path.insert(0, str(Path(drsa_demo_dir).resolve()))
    try:
        import nptyping  # noqa: F401
    except ImportError:
        stub = types.ModuleType("nptyping")
        stub.NDArray = np.ndarray
        sys.modules["nptyping"] = stub
    try:
        from cxai import drsa
    except ImportError as e:
        raise ImportError("cxai not found: clone https://github.com/p16i/drsa-demo and pass "
                          "--drsa_demo_dir") from e
    return drsa


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def select_images(rw, experiment, n_images, wm_fraction):
    """First n_images positives of the train pool, with every k-th one watermarked.

    Returns [(path, watermarked)], where k = round(1 / wm_fraction).
    """
    cfg = rw.EXPERIMENT_CONFIGS[experiment]
    _, rank_pos, _, _, _ = rw.build_splits(cfg["pos_dir"], cfg["neg_dir"], n_rank_pos=n_images)
    paths = [p for p, _, _ in rank_pos]
    if len(paths) < n_images:
        raise ValueError(f"only {len(paths)} positive training images available, need {n_images}")
    if wm_fraction <= 0:
        return [(p, False) for p in paths]
    every = round(1 / wm_fraction)
    return [(p, i % every == every - 1) for i, p in enumerate(paths)]


def make_transforms(rw):
    rc = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    tt = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=rw.IMAGENET_MEAN, std=rw.IMAGENET_STD)])
    return rc, tt


def load_batch(items, rc, tt, add_wm):
    xs = []
    for path, wm in items:
        img = rc(Image.open(path).convert("RGB"))
        if wm:
            img = add_wm(img)
        xs.append(tt(img))
    return torch.stack(xs)


# ---------------------------------------------------------------------------
# Activations and LRP context (Zennit)
# ---------------------------------------------------------------------------

def consumers_of(layer):
    """Modules whose LRP rule must return m for the output of `layer`.

    These are the first convs of the next stage. After layer4 it is fc: the global average
    pool in between is left unhooked, since it is linear and bias-free, so its plain gradient
    is LRP-0, and the z-rule there would zero c for channels whose pooled value is 0.
    """
    i = STAGES.index(layer)
    if i == len(STAGES) - 1:
        return {"fc"}
    nxt = STAGES[i + 1]
    return {f"{nxt}.0.conv1", f"{nxt}.0.downsample.0"}


def make_composite(layer, gamma, epsilon):
    """Zennit composite whose grad at the output of `layer` is the context vector c.

    Zennit's hooks hand each input its relevance R = a * m, where m is the LRP-modified
    gradient. cxai, which produced the paper's DRSA matrices, takes the context as m itself,
    read as the grad of the inspected activation under its forward-hook rules. m is also
    nonzero where a = 0, whereas R / a is undefined there. So the modules that read the
    layer's output return m instead of R: their reducer drops the multiplication by the
    input. The layer's output is post-ReLU (a >= 0), so only the positive-input terms of
    the gamma rule apply.
    """
    from zennit.composites import LayerMapComposite, layer_map_base
    from zennit.rules import Epsilon, Gamma
    from zennit.torchvision import ResNetCanonizer
    from zennit.types import Convolution

    consumer_conv = Gamma(gamma=gamma)
    consumer_conv.reducer = lambda inputs, gradients: gradients[0] + gradients[2]
    consumer_fc = Epsilon(epsilon=epsilon)
    consumer_fc.reducer = lambda inputs, gradients: gradients[0]
    consumers = consumers_of(layer)

    class ContextComposite(LayerMapComposite):
        def mapping(self, ctx, name, module):
            if name in consumers:
                return consumer_fc if name == "fc" else consumer_conv
            if name == "avgpool" and "fc" in consumers:
                return None
            return super().mapping(ctx, name, module)

    layer_map = layer_map_base() + [
        (Convolution, Gamma(gamma=gamma)),
        (torch.nn.Linear, Epsilon(epsilon=epsilon)),
    ]
    return ContextComposite(layer_map=layer_map, canonizers=[ResNetCanonizer()])


class StageTap:
    """Forward hook on a stage: detaches its output and makes it a leaf that requires grad.

    The backward pass then stops at the stage output, and the grad of that leaf is the
    modified (LRP) gradient there, i.e. the context vector c = R / a.
    """

    def __init__(self, stage_module):
        self.out = None
        self.handle = stage_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.out = output.detach().requires_grad_(True)
        return self.out

    def remove(self):
        self.handle.remove()


def activation_and_context(model, layer, x, target, composite=None):
    """(a, c) at the output of `layer`, each (B, D, H, W). Plain gradient if composite is None.

    With make_composite(layer, ...), the grad at the layer's output is the context c, and the
    layer's relevance is R = a * c. The backward pass is seeded with the target logit, so the
    relevance sums to the logit, up to what the biases above the layer absorb.
    """
    tap = StageTap(getattr(model, layer))
    try:
        if composite is None:
            logits = model(x)
            (c,) = torch.autograd.grad(logits[:, target].sum(), tap.out)
        else:
            with composite.context(model) as modified:
                logits = modified(x)
                seed = torch.zeros_like(logits)
                seed[:, target] = logits[:, target].detach()
                # Seeding with the logit z makes the fc rule pass m = W^T (z / z) back,
                # the same scale as cxai's backward from the logit with a unit seed.
                (c,) = torch.autograd.grad(logits, tap.out, grad_outputs=seed)
    finally:
        tap.remove()
    return tap.out.detach(), c.detach()


def extract(model, layer, items, target, composite, rc, tt, add_wm, n_locations, seed, batch_size, device):
    """Activation and context vectors at n_locations random spatial positions per image."""
    rng = np.random.default_rng(seed)
    acts, ctxs, locations = [], [], []
    for start in range(0, len(items), batch_size):
        x = load_batch(items[start:start + batch_size], rc, tt, add_wm).to(device)
        a, c = activation_and_context(model, layer, x, target, composite)
        B, D, H, W = a.shape
        a = a.reshape(B, D, H * W).cpu().numpy()
        c = c.reshape(B, D, H * W).cpu().numpy()
        for b in range(B):
            loc = rng.permutation(H * W)[:n_locations]
            acts.append(a[b][:, loc].T)
            ctxs.append(c[b][:, loc].T)
            locations.append(loc.tolist())
        print(f"  [extract] {min(start + batch_size, len(items))}/{len(items)} images", flush=True)
    return np.concatenate(acts), np.concatenate(ctxs), locations


def check_context(model, layer, items, target, rc, tt, add_wm, epsilon, device, n=8):
    """With gamma = 0, the Zennit context must match the plain gradient (LRP-0 = gradient).

    TF32 is disabled during the check: its ~1e-3 relative precision in convolutions otherwise
    dominates the comparison between the BN-folded and unfolded networks.
    """
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        x = load_batch(items[:n], rc, tt, add_wm).to(device)
        a0, g = activation_and_context(model, layer, x, target, None)
        a1, c = activation_and_context(model, layer, x, target, make_composite(layer, 0.0, epsilon))
        with torch.no_grad():
            logits = model(x)[:, target]
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    act_err = (a0 - a1).abs().max().item() / a0.abs().max().item()
    ctx_err = (c - g).norm().item() / g.norm().item()
    rel_lrp = (a1 * c).flatten(1).sum(1)
    print(f"[check] canonized activation vs plain: max rel. diff {act_err:.2e}")
    print(f"[check] Zennit gamma=0 context vs gradient: rel. L2 diff {ctx_err:.2e}")
    print(f"[check] sum(a*c) at {layer} / logit: {(rel_lrp / logits).cpu().numpy().round(3)} "
          f"(below 1: relevance absorbed by biases above the layer)")
    return act_err, ctx_err


# ---------------------------------------------------------------------------
# DRSA
# ---------------------------------------------------------------------------

def normalize(x):
    """Eq. (E.1): x / (D^(1/4) sqrt(E[x^2])), with the mean over all samples and dimensions."""
    D = x.shape[1]
    return x / (D ** 0.25 * x.pow(2).mean().sqrt())


def subspace_relevance(A, C, U, ns):
    """Mean relevance E[(Uk^T a) . (Uk^T c)] of each of the ns column blocks of U."""
    return ((A @ U) * (C @ U)).reshape(A.shape[0], ns, -1).sum(2).mean(0)


def run_drsa(drsa, act, ctx, ns, epochs, trials, seed, device):
    A = normalize(torch.as_tensor(act, dtype=torch.float32, device=device))
    C = normalize(torch.as_tensor(ctx, dtype=torch.float32, device=device))
    D = A.shape[1]
    if D % ns:
        raise ValueError(f"D={D} is not divisible into {ns} equal subspaces")
    ss = D // ns
    best_obj, U, obj_values = drsa.optimize(drsa.obj_drsa, A, C, seed=seed, ns=ns, ss=ss,
                                            epochs=epochs, device=device, total_trials=trials)
    rel = subspace_relevance(A, C, U, ns)
    order = torch.argsort(rel, descending=True)
    U = U.reshape(D, ns, ss)[:, order].reshape(D, D)
    rel_sorted = subspace_relevance(A, C, U, ns)
    total = (A * C).sum(1).mean()
    return U.cpu(), {
        "best_objective": best_obj,
        "objective_values_every_100": [float(v) for v in obj_values[::100]],
        "subspace_relevance": rel_sorted.cpu().tolist(),
        "subspace_relevance_fraction": (rel_sorted / total).cpu().tolist(),
        "original_block_order": order.cpu().tolist(),
        "orthogonality_error": (U.T @ U - torch.eye(D, device=U.device)).abs().max().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=sorted(IMAGENET_TARGETS), required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), default="resnet50")
    p.add_argument("--layer", choices=STAGES, default="layer3",
                   help="Stage whose output is the DRSA layer (= --virtual_layer of the pruning run).")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--n_locations", type=int, default=20, help="Random spatial locations per image.")
    p.add_argument("--wm_fraction", type=float, default=None,
                   help="Fraction of images given the synthetic watermark (default: carton 0, crate 0.25).")
    p.add_argument("--num_subspaces", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--trials", type=int, default=3, help="Random orthogonal restarts; the best is kept.")
    p.add_argument("--gamma", type=float, default=0.0, help="LRP-gamma for the convs above the layer.")
    p.add_argument("--epsilon", type=float, default=1e-6, help="LRP-epsilon for fc.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--drsa_demo_dir", default=str(REPO_ROOT.parent / "drsa-demo"))
    p.add_argument("--out", default=None,
                   help="Output .pt (default: <projection dir>/U_<experiment>_<arch>_<layer>.pt).")
    p.add_argument("--cache_dir", default=None,
                   help="Where act.npy / ctx.npy are cached (default: <results>/drsa/<experiment>_<arch>_<layer>).")
    p.add_argument("--recompute", action="store_true", help="Ignore cached activations/contexts.")
    p.add_argument("--check", action="store_true", help="Only run the context sanity check.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    if args.wm_fraction is None:
        args.wm_fraction = DEFAULT_WM_FRACTION[args.experiment]
    slug = f"{args.experiment}_{args.arch}_{args.layer}"
    out = Path(args.out) if args.out else PROJECTION_DIR / f"U_{slug}.pt"
    cache = Path(args.cache_dir) if args.cache_dir else RESULTS_ROOT / "drsa" / slug
    target = IMAGENET_TARGETS[args.experiment]
    D = STAGE_WIDTHS[args.arch][args.layer]

    rw = _load_watermark_experiment()
    items = select_images(rw, args.experiment, args.n_images, args.wm_fraction)
    rc, tt = make_transforms(rw)
    add_wm = AddWatermark(image_size=224)

    model = ARCHS[args.arch](weights="IMAGENET1K_V1").eval().to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.check:
        check_context(model, args.layer, items, target, rc, tt, add_wm, args.epsilon, args.device)
        return

    drsa = _import_cxai_drsa(args.drsa_demo_dir)

    act_path, ctx_path, meta_path = cache / "act.npy", cache / "ctx.npy", cache / "extract.json"
    extract_settings = {"experiment": args.experiment, "arch": args.arch, "layer": args.layer,
                        "target_logit": target, "n_images": args.n_images, "n_locations": args.n_locations,
                        "wm_fraction": args.wm_fraction, "gamma": args.gamma, "epsilon": args.epsilon,
                        "seed": args.seed}
    cached = (not args.recompute and act_path.exists() and ctx_path.exists() and meta_path.exists()
              and json.loads(meta_path.read_text())["settings"] == extract_settings)
    if cached:
        print(f"[extract] using cached vectors in {cache}")
        act, ctx = np.load(act_path), np.load(ctx_path)
    else:
        print(f"[extract] {len(items)} images ({sum(w for _, w in items)} watermarked) -> {args.arch} "
              f"{args.layer} (D={D}), context w.r.t. logit {target}, gamma={args.gamma}")
        t0 = time.time()
        act, ctx, locations = extract(model, args.layer, items, target,
                                      make_composite(args.layer, args.gamma, args.epsilon), rc, tt, add_wm,
                                      args.n_locations, args.seed, args.batch_size, args.device)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(act_path, act)
        np.save(ctx_path, ctx)
        meta_path.write_text(json.dumps({
            "settings": extract_settings,
            "images": [{"path": Path(p).name, "watermarked": w} for p, w in items],
            "locations": locations,
            "seconds": time.time() - t0,
        }, indent=1))
        print(f"[extract] {act.shape[0]} vectors of dim {act.shape[1]} in {time.time() - t0:.0f}s")

    print(f"[drsa] K={args.num_subspaces} x {D // args.num_subspaces}, {args.epochs} iterations, "
          f"{args.trials} restarts on {args.device}")
    t0 = time.time()
    U, info = run_drsa(drsa, act, ctx, args.num_subspaces, args.epochs, args.trials, args.seed, args.device)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(U.float().contiguous(), out)
    out.with_suffix(".json").write_text(json.dumps({
        "extract": extract_settings,
        "num_subspaces": args.num_subspaces,
        "subspace_dim": D // args.num_subspaces,
        "epochs": args.epochs,
        "trials": args.trials,
        "normalization": "x / (D^(1/4) sqrt(E[x^2])) (Eq. E.1)",
        "cache_dir": (cache.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
                      if cache.resolve().is_relative_to(REPO_ROOT.resolve()) else cache.name),
        "seconds": time.time() - t0,
        **info,
    }, indent=1))
    print(f"[drsa] objective {info['best_objective']:.4f}, subspace relevance fractions "
          f"{np.round(info['subspace_relevance_fraction'], 3).tolist()} (sorted, highest first)")
    print(f"[drsa] saved {out} ({D}x{D}) and {out.with_suffix('.json').name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
