"""
checkpoint.py

Portable checkpoints for pruned VGG16 models.

Pruned models have irregular per-layer channel counts, so a plain state_dict
cannot be loaded into a stock torchvision VGG16. Instead of pickling the whole
nn.Module (which breaks when code moves and requires torch.load(weights_only=False)),
we store:

    {
      'format_version': 1,
      'arch':           'vgg16',
      'channels':       [13 ints],   # out_channels of every conv, in forward order
      'num_classes':    int,
      'state_dict':     {...},       # canonical torchvision keys: features.* / classifier.*
      'meta':           {...},       # primitives only (loadable with weights_only=True)
    }

and rebuild the architecture with build_pruned_vgg16(channels) before loading.

Both vanilla pruned models (model.features) and CNP pruned models
(model.before / model.after after on_before_final_finetune removed encode/decode)
are saved in the same canonical layout and load as a torchvision VGG.

AugmentedVGG16 has no avgpool; torchvision's AdaptiveAvgPool2d((7, 7)) is the
identity for 224x224 inputs, so logits match for the standard input size.
"""

import torch
import torch.nn as nn
from torchvision.models.vgg import VGG

from .prune_layer import get_conv_seq_and_name

FORMAT_VERSION = 1

# torchvision VGG16 ("D") layout; ints are replaced by the pruned channel counts.
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]
NUM_CONVS = sum(1 for v in VGG16_CFG if v != "M")  # 13


def _conv_layers(model: nn.Module) -> list:
    """Conv2d layers of the feature extractor in forward order, excluding encode/decode."""
    seq, _ = get_conv_seq_and_name(model)
    skip = {id(getattr(model, name)) for name in ("encode", "decode") if hasattr(model, name)}
    return [m for m in seq if isinstance(m, nn.Conv2d) and id(m) not in skip]


def extract_channels(model: nn.Module) -> list:
    """Return out_channels of all 13 VGG16 convs (including conv4_3), in forward order."""
    channels = [m.out_channels for m in _conv_layers(model)]
    if len(channels) != NUM_CONVS:
        raise ValueError(f"Expected {NUM_CONVS} conv layers, found {len(channels)}")
    return channels


def _num_classes(model: nn.Module) -> int:
    linears = [m for m in model.classifier if isinstance(m, nn.Linear)]
    return linears[-1].out_features


def to_vgg16_state_dict(model: nn.Module) -> dict:
    """State dict with canonical torchvision VGG16 keys (features.* / classifier.*).

    Augmented models are remapped from before.{i}.* / after.{j}.* and must no longer
    contain encode/decode (call adapter.on_before_final_finetune() first).
    """
    if hasattr(model, "features"):
        sd = model.state_dict()
    elif hasattr(model, "before"):
        if hasattr(model, "encode") or hasattr(model, "decode"):
            raise ValueError("Remove the virtual concept layer (encode/decode) before saving; "
                             "see AugmentedVGGAdapter.on_before_final_finetune().")
        split = len(model.before)
        sd = {}
        for k, v in model.state_dict().items():
            prefix, _, rest = k.partition(".")
            if prefix == "before":
                idx, _, suffix = rest.partition(".")
                sd[f"features.{int(idx)}.{suffix}"] = v
            elif prefix == "after":
                idx, _, suffix = rest.partition(".")
                sd[f"features.{int(idx) + split}.{suffix}"] = v
            else:
                sd[k] = v
    else:
        raise ValueError("Unrecognized model structure")
    return {k: v.detach().cpu() for k, v in sd.items()}


def vgg16_to_augmented_state_dict(state_dict: dict, split: int = 23) -> dict:
    """Remap torchvision VGG16 keys to AugmentedVGG16's before.* / after.* layout.

      features.{i}.*  ->  before.{i}.*           for i < split   (conv1_1 .. ReLU after conv4_3)
      features.{i}.*  ->  after.{i - split}.*    for i >= split  (pool4 .. pool5)
      other keys (classifier.*) are unchanged; encode/decode are not in the checkpoint.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("features."):
            _, idx, suffix = k.split(".", 2)
            idx = int(idx)
            if idx < split:
                new_sd[f"before.{idx}.{suffix}"] = v
            else:
                new_sd[f"after.{idx - split}.{suffix}"] = v
        else:
            new_sd[k] = v
    return new_sd


def build_pruned_vgg16(channels, num_classes: int = 2) -> VGG:
    """Build an (uninitialized) torchvision VGG16 with the given per-conv channel counts."""
    channels = list(channels)
    if len(channels) != NUM_CONVS:
        raise ValueError(f"Expected {NUM_CONVS} channel counts, got {len(channels)}")

    layers, in_ch, it = [], 3, iter(channels)
    for v in VGG16_CFG:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_ch = next(it)
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_ch = out_ch

    model = VGG(nn.Sequential(*layers), num_classes=num_classes, init_weights=False)
    model.classifier[0] = nn.Linear(channels[-1] * 7 * 7, model.classifier[0].out_features)
    return model


def save_pruned_checkpoint(path, model: nn.Module, meta: dict = None) -> None:
    """Save a pruned VGG16 (vanilla or CNP) as state_dict + channel counts.

    meta should contain only primitives/lists/dicts so the file loads with weights_only=True.
    """
    torch.save({
        "format_version": FORMAT_VERSION,
        "arch": "vgg16",
        "channels": extract_channels(model),
        "num_classes": _num_classes(model),
        "state_dict": to_vgg16_state_dict(model),
        "meta": dict(meta or {}),
    }, path)


def load_pruned_checkpoint(path, map_location="cpu"):
    """Load a checkpoint written by save_pruned_checkpoint.

    Returns (model, info) where model is a torchvision VGG in eval mode and info holds
    every checkpoint field except the state_dict.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    version = ckpt.get("format_version")
    if version != FORMAT_VERSION or ckpt.get("arch") != "vgg16":
        raise ValueError(f"Unsupported checkpoint (format_version={version!r}, "
                         f"arch={ckpt.get('arch')!r}) at {path!r}")
    model = build_pruned_vgg16(ckpt["channels"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    info = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, info


def load_state_dict_from_file(path, map_location="cpu") -> dict:
    """Extract a state_dict from a generic (e.g. unpruned fine-tuned) checkpoint file.

    Accepts a plain state_dict or {'state_dict': ...}. Pickled nn.Module objects
    ({'model': module} or a bare module) are also accepted, which requires
    weights_only=False: only use this on files you trust.
    """
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            obj = ckpt["model"]
            return obj.state_dict() if hasattr(obj, "state_dict") else obj
        return ckpt
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise ValueError(f"Cannot extract state_dict from checkpoint at {path!r}")
