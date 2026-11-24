"""
lrp_resnet.py

LRP for torchvision ResNets (and AugmentedResNet), used to rank filters for CNP pruning.

A ResNet is not a module chain, so relevance is propagated over an explicit record of the
forward pass (resnet_lrp_forward) rather than over a flat list of modules:

- Conv + BatchNorm are fused into one convolution (BatchNorm in eval mode is a per-channel
  affine map), and LRP-α1β0 is applied to the fused weights. The stem uses the 'first' (zB) rule.
- ReLU passes relevance unchanged.
- Residual addition  z = z_branch + z_shortcut:  relevance is split in proportion to the positive
  parts,  R_branch = R · z_branch⁺ / (z_branch⁺ + z_shortcut⁺),  R_shortcut likewise.
- Max-pooling (stem): winner-takes-all (lrp.Pooling).
- Global average pooling:  R_in = x · R_out / Σ_hw x.
- Virtual layer (encode/decode 1×1 convs): gamma rule with gamma = 0, as for AugmentedVGG16.
- fc: LRP-α1β0.

Biases are ignored in the denominators (as in lrp.py), so relevance is conserved layer by layer.
Relevance at the output of each conv+BN pair (before the ReLU) is returned per conv name and
is what CNP accumulates as per-filter importance.
"""

import torch
import torch.nn as nn

from .lrp import SMALL_NUMBER, lrp
from .resnet import STAGES, conv_bn_pairs


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a Conv2d equal to bn(conv(x)) for BatchNorm in eval mode."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                      padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True,
                      device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_((conv_bias - bn.running_mean) * scale + bn.bias)
    return fused


def _conv_op(name, conv, bn, x, rule):
    fused = fuse_conv_bn(conv, bn)
    fused.input = x
    fused.output = fused(x)
    return fused.output, ("conv", name, fused, rule)


def _block_forward(name, block, x, ops):
    branch_ops, out = [], x
    pairs = conv_bn_pairs(block)
    for j, (i, conv, bn) in enumerate(pairs):
        out, op = _conv_op(f"{name}.conv{i}", conv, bn, out, "alpha")
        branch_ops.append(op)
        if j < len(pairs) - 1:
            out = torch.relu(out)
    z_branch = out

    shortcut_ops = []
    if block.downsample is not None:
        z_shortcut, op = _conv_op(f"{name}.downsample.0", block.downsample[0], block.downsample[1], x, "alpha")
        shortcut_ops.append(op)
    else:
        z_shortcut = x

    ops.append(("block", name, branch_ops, shortcut_ops, z_branch, z_shortcut))
    return torch.relu(z_branch + z_shortcut)


def resnet_lrp_forward(model: nn.Module, x: torch.Tensor):
    """Forward pass that records what LRP needs. Returns (logits, ops).

    The model must be in eval mode (BatchNorm fusion uses running statistics).
    """
    if model.training:
        raise RuntimeError("resnet_lrp_forward requires model.eval() (BatchNorm is fused using running statistics)")
    ops = []
    with torch.no_grad():
        out, op = _conv_op("conv1", model.conv1, model.bn1, x, "first")
        ops.append(op)
        out = torch.relu(out)

        pool = model.maxpool
        pool.input = out
        out = pool(out)
        pool.output = out
        ops.append(("maxpool", pool))

        for stage in STAGES:
            for b, block in enumerate(getattr(model, stage)):
                out = _block_forward(f"{stage}.{b}", block, out, ops)
            if getattr(model, "augmented", False) and stage == getattr(model, "split_layer", None):
                for layer in (model.encode, model.decode):
                    layer.input = out
                    out = layer(out)
                    layer.output = out
                ops.append(("virtual", model.encode, model.decode))

        ops.append(("avgpool", out))
        out = torch.flatten(model.avgpool(out), 1)
        model.fc.input = out
        logits = model.fc(out)
        model.fc.output = logits
        ops.append(("fc", model.fc))
    return logits, ops


def _backward(ops, R, conv_relevance):
    for op in reversed(ops):
        kind = op[0]
        if kind == "fc":
            R = lrp(op[1], R, lrp_var="alpha", param=1)
        elif kind == "avgpool":
            x = op[1]
            R = x * R.reshape(x.shape[0], x.shape[1], 1, 1) / (x.sum(dim=(2, 3), keepdim=True) + SMALL_NUMBER)
        elif kind == "virtual":
            _, encode, decode = op
            R = lrp(decode, R, lrp_var="gamma", param=0.0)
            R = lrp(encode, R, lrp_var="gamma", param=0.0)
        elif kind == "block":
            _, _, branch_ops, shortcut_ops, z_branch, z_shortcut = op
            zb, zs = z_branch.clamp(min=0), z_shortcut.clamp(min=0)
            denom = zb + zs + SMALL_NUMBER
            R_branch = _backward(branch_ops, R * zb / denom, conv_relevance)
            R_shortcut = _backward(shortcut_ops, R * zs / denom, conv_relevance)
            R = R_branch + R_shortcut
        elif kind == "conv":
            _, name, fused, rule = op
            conv_relevance[name] = R
            R = lrp(fused, R, lrp_var=rule, param=None if rule == "first" else 1)
        elif kind == "maxpool":
            R = lrp(op[1], R)
        else:
            raise ValueError(f"unknown op {kind!r}")
    return R


def virtual_layer_gradient(model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """Gradient of the target logit w.r.t. the input of the virtual layer (B, D, h, w).

    The unablated virtual layer is an identity, so this is the gradient w.r.t. the stage
    output. Above that layer, LRP-0 with biases in the denominators equals gradient x input.
    So this is the DRSA context vector at every position, also where the activation is 0,
    which matches cxai and experiments/compute_drsa_resnet.py with gamma = 0.
    """
    store = {}

    def pre_hook(module, args):
        store["a"] = args[0].detach().requires_grad_(True)
        return (store["a"],)

    handle = model.encode.register_forward_pre_hook(pre_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            (g,) = torch.autograd.grad(logits[:, target].sum(), store["a"])
    finally:
        handle.remove()
    return g.detach()


def resnet_lrp_subspaces(ops, R: torch.Tensor, subspace_dims, virtual_grad: torch.Tensor = None):
    """Per-subspace LRP heatmaps for an AugmentedResNet with an unablated virtual layer.

    The forward pass (recorded in `ops`) is left unchanged. The relevance at the virtual
    layer is split into the column blocks U_k of U (`subspace_dims`). Each block's relevance
    is then propagated to the input separately with the rules of this module
    (DRSA, Chormai et al. 2024, Eqs. 1-2). Every rule is linear in the relevance, so the
    subspace heatmaps sum to the standard heatmap.

    Above the virtual layer:
      virtual_grad is None   the rules of this module (alpha1-beta0) from the output R down
                             to the latent h = Uᵀa, then LRP-0 (gamma = 0) through decode/encode.
      virtual_grad given     LRP-0 with biases, i.e. gradient x input. virtual_grad is
                             virtual_layer_gradient(model, x, target) = c, and subspace k gets
                             relevance a ⊙ U_k U_kᵀ c at the virtual layer's input. That is the
                             quantity DRSA optimizes, (U_kᵀa)(U_kᵀc) per position. R is unused.

    Returns (R_input, R_input_subspaces, latent_maps):
      R_input            (B, 3, H, W)      standard heatmap
      R_input_subspaces  (K, B, 3, H, W)   one heatmap per subspace
      latent_maps        (K, B, h, w)      relevance of each subspace at each virtual-layer position
    """
    idx = [i for i, op in enumerate(ops) if op[0] == "virtual"]
    if len(idx) != 1:
        raise ValueError("resnet_lrp_subspaces needs a forward pass through a model with a virtual layer")
    idx = idx[0]
    _, encode, decode = ops[idx]
    if sum(subspace_dims) != encode.out_channels:
        raise ValueError(f"subspace_dims sum to {sum(subspace_dims)}, but the virtual layer has "
                         f"{encode.out_channels} dimensions")
    blocks, start = [], 0
    for d in subspace_dims:
        blocks.append(slice(start, start + d))
        start += d

    with torch.no_grad():
        if virtual_grad is None:
            R_latent = lrp(decode, _backward(ops[idx + 1:], R, {}), lrp_var="gamma", param=0.0)
            R_layer = []
            for blk in blocks:
                R_k = torch.zeros_like(R_latent)
                R_k[:, blk] = R_latent[:, blk]
                R_layer.append(lrp(encode, R_k, lrp_var="gamma", param=0.0))
        else:
            a = encode.input
            U = decode.weight[:, :, 0, 0]                       # D x D, columns = basis vectors
            h_c = torch.einsum("bdhw,dk->bkhw", virtual_grad, U)  # Uᵀc
            R_layer = [a * torch.einsum("bkhw,dk->bdhw", h_c[:, blk], U[:, blk]) for blk in blocks]
        heatmaps = [_backward(ops[:idx], R_k, {}) for R_k in R_layer]
        maps = [R_k.sum(1) for R_k in R_layer]
        R_input = _backward(ops[:idx], sum(R_layer), {})
    return R_input, torch.stack(heatmaps), torch.stack(maps)


def resnet_lrp_backward(ops, R: torch.Tensor):
    """Propagate output relevance R (B, num_classes) back to the input.

    Returns (R_input, conv_relevance) where conv_relevance maps conv names ('conv1',
    'layer2.0.conv1', 'layer2.0.downsample.0', ...) to the relevance at that conv+BN output.
    """
    conv_relevance = {}
    with torch.no_grad():
        R_input = _backward(ops, R, conv_relevance)
    return R_input, conv_relevance
