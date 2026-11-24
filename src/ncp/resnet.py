"""
resnet.py

CNP virtual concept layer and structural pruning for torchvision ResNets (resnet18/34/50).

Where the virtual layer goes
----------------------------
Inside a residual block, the identity/downsample shortcut runs in parallel with the
convolutions, so a virtual layer placed there would be bypassed by the skip connection.
The residual stream is a single tensor only *between* blocks, so the virtual layer is inserted
at a stage boundary, after the last block of `split_layer` (default layer3):

    stem → layer1 → layer2 → layer3 → [encode (Uᵀ) → decode (U)] → layer4 → avgpool → fc

Every path from input to output passes through it: the first block of the next stage reads
the virtual layer's output on both its residual branch and its shortcut (downsample).

D, the virtual layer width, is the residual-stream width after `split_layer` (STAGE_WIDTHS);
U must be a D×D DRSA projection matrix computed at that layer.

What can be pruned
------------------
Residual-stream channels are coupled through the additions: all blocks of a stage, the
downsample shortcut, and the next stage's first convolutions share them, and they also fix D.
CNP therefore prunes only block-internal channels, whose width is local to one block:

    BasicBlock:  conv1          (bn1 and conv2's input shrink with it)
    Bottleneck:  conv1, conv2   (bn1/conv2 input, bn2/conv3 input shrink with them)

The stem, the last conv of every block, and downsample convs are never pruned. This is the
ResNet analogue of keeping conv4_3 intact in VGG16.

Subspace indices are 0-indexed (see AugmentedVGG16.ablate_subspace_matrix).
"""

import torch
import torch.nn as nn
from torchvision import models

ARCHS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
STAGES = ("layer1", "layer2", "layer3", "layer4")
STAGE_WIDTHS = {
    "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet34": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
}


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def conv_bn_pairs(block):
    """[(i, conv_i, bn_i), ...] for a BasicBlock (i = 1, 2) or Bottleneck (i = 1, 2, 3)."""
    pairs, i = [], 1
    while hasattr(block, f"conv{i}"):
        pairs.append((i, getattr(block, f"conv{i}"), getattr(block, f"bn{i}")))
        i += 1
    return pairs


def iter_blocks(model):
    """Yield ("layerS.B", block) for every residual block in forward order."""
    for stage in STAGES:
        for b, block in enumerate(getattr(model, stage)):
            yield f"{stage}.{b}", block


def prunable_conv_names(model):
    """Names of block-internal convs (every conv of a block except its last), in forward order."""
    return [f"{name}.conv{i}"
            for name, block in iter_blocks(model)
            for i, _, _ in conv_bn_pairs(block)[:-1]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AugmentedResNet(nn.Module):
    """torchvision ResNet with an optional CNP virtual concept layer after `split_layer`.

    Submodule names match torchvision (conv1, bn1, ..., layer1-4, avgpool, fc), so once the
    virtual layer is removed the state_dict is a torchvision ResNet state_dict.

    Args:
        arch:        'resnet18', 'resnet34' or 'resnet50'
        U, UT:       D×D decoding / encoding matrices (UT defaults to U.T). With U=None the
                     model has no virtual layer (vanilla pruning baseline).
        split_layer: stage after which the virtual layer is inserted ('layer1'..'layer4')
        weights:     torchvision weights (default ImageNet); None for random initialization
    """

    def __init__(self, arch: str = "resnet50", U: torch.Tensor = None, UT: torch.Tensor = None,
                 split_layer: str = "layer3", weights="IMAGENET1K_V1"):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {sorted(ARCHS)}, got {arch!r}")
        if split_layer not in STAGES:
            raise ValueError(f"split_layer must be one of {STAGES}, got {split_layer!r}")

        base = ARCHS[arch](weights=weights)
        for name in ("conv1", "bn1", "relu", "maxpool", *STAGES, "avgpool", "fc"):
            setattr(self, name, getattr(base, name))
        for param in self.parameters():
            param.requires_grad = False

        self.arch = arch
        self.split_layer = split_layer
        self.virtual_dim = D = STAGE_WIDTHS[arch][split_layer]
        self.augmented = U is not None

        if U is not None:
            UT = U.t() if UT is None else UT
            if tuple(U.shape) != (D, D) or tuple(UT.shape) != (D, D):
                raise ValueError(f"U and UT must be {D}x{D} for {arch} after {split_layer}; "
                                 f"got {tuple(U.shape)} and {tuple(UT.shape)}")
            self.encode = nn.Conv2d(D, D, kernel_size=1, bias=False)
            self.decode = nn.Conv2d(D, D, kernel_size=1, bias=False)
            with torch.no_grad():
                self.encode.weight.copy_(UT.reshape(D, D, 1, 1))
                self.decode.weight.copy_(U.reshape(D, D, 1, 1))
            for param in (*self.encode.parameters(), *self.decode.parameters()):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for stage in STAGES:
            x = getattr(self, stage)(x)
            if self.augmented and stage == self.split_layer:
                # stage boundary: no skip connection crosses this point
                x = self.decode(self.encode(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Structural pruning of block-internal channels
# ---------------------------------------------------------------------------

def _new_conv(conv, in_channels, out_channels):
    if conv.groups != 1:
        raise ValueError("grouped convolutions are not supported")
    return nn.Conv2d(in_channels, out_channels, conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation, bias=conv.bias is not None,
                     padding_mode=conv.padding_mode, device=conv.weight.device, dtype=conv.weight.dtype)


def _select_out_channels(conv, keep):
    new = _new_conv(conv, conv.in_channels, len(keep))
    with torch.no_grad():
        new.weight.copy_(conv.weight[keep])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[keep])
    return new


def _select_in_channels(conv, keep):
    new = _new_conv(conv, len(keep), conv.out_channels)
    with torch.no_grad():
        new.weight.copy_(conv.weight[:, keep])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _select_bn_channels(bn, keep):
    ref = bn.weight if bn.affine else bn.running_mean
    new = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
                         track_running_stats=bn.track_running_stats, device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        if bn.affine:
            new.weight.copy_(bn.weight[keep])
            new.bias.copy_(bn.bias[keep])
        if bn.track_running_stats:
            new.running_mean.copy_(bn.running_mean[keep])
            new.running_var.copy_(bn.running_var[keep])
            new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new


def _keep_channels(block, i, keep):
    """Keep only channels `keep` of conv_i / bn_i and the matching inputs of conv_{i+1}."""
    setattr(block, f"conv{i}", _select_out_channels(getattr(block, f"conv{i}"), keep))
    setattr(block, f"bn{i}", _select_bn_channels(getattr(block, f"bn{i}"), keep))
    setattr(block, f"conv{i + 1}", _select_in_channels(getattr(block, f"conv{i + 1}"), keep))


def _split_conv_name(conv_name):
    block_name, conv_attr = conv_name.rsplit(".", 1)
    return block_name, int(conv_attr[len("conv"):])


def prune_resnet_channel(model: nn.Module, conv_name: str, filter_index: int, cuda_flag: bool = False):
    """Remove output channel `filter_index` of a block-internal conv (e.g. 'layer2.0.conv1').

    Also removes the matching BatchNorm channel and input channel of the next conv in the block.
    The model is modified in place and returned (cuda_flag is accepted for API parity with
    prune_layer.prune_conv_layer_sequential; modules keep their device).
    """
    if conv_name not in prunable_conv_names(model):
        raise ValueError(f"{conv_name!r} is not a prunable block-internal conv "
                         f"(the stem, the last conv of a block and downsample convs are never pruned)")
    block_name, i = _split_conv_name(conv_name)
    block = model.get_submodule(block_name)
    n = getattr(block, f"conv{i}").out_channels
    if not 0 <= filter_index < n:
        raise IndexError(f"filter_index {filter_index} out of range for {conv_name} ({n} channels)")
    if n == 1:
        raise ValueError(f"cannot remove the last channel of {conv_name}")
    _keep_channels(block, i, [c for c in range(n) if c != filter_index])
    return model


def shrink_block_channels(block, i, n_channels):
    """Resize conv_i/bn_i (and conv_{i+1}'s input) of a block to `n_channels` (used to rebuild checkpoints)."""
    if getattr(block, f"conv{i}").out_channels != n_channels:
        _keep_channels(block, i, list(range(n_channels)))
