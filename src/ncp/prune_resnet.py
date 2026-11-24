"""
prune_resnet.py

ResNet support for the CNP prune + fine-tune loop (prune_vgg.PruningFineTuner).

ResNetAdapter provides the same adapter interface PruningFineTuner uses for VGG16
(make_pruner, prune_filter, layer_out_channels, total_prunable_filters, ...), with layers
identified by conv name (e.g. 'layer3.2.conv2') instead of a sequential index.
Only block-internal channels are pruned; see resnet.py for why.
"""

from heapq import nsmallest
from operator import itemgetter

import torch.nn as nn

from .lrp_resnet import resnet_lrp_backward, resnet_lrp_forward
from .prune_vgg import shift_pruning_plan
from .resnet import prunable_conv_names, prune_resnet_channel


class ResNetAdapter:
    """Adapter for AugmentedResNet (with or without a virtual layer) and torchvision ResNets."""

    def __init__(self, model: nn.Module):
        self.model = model

    def prunable_conv_names(self) -> list:
        return prunable_conv_names(self.model)

    def total_prunable_filters(self) -> int:
        return sum(self.model.get_submodule(n).out_channels for n in self.prunable_conv_names())

    def setup_iterative_finetune_params(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = not name.startswith(("encode.", "decode."))

    def on_before_final_finetune(self):
        """Disable and remove the virtual layer before the final fine-tuning phase."""
        self.model.augmented = False
        for name in ("encode", "decode"):
            if hasattr(self.model, name):
                delattr(self.model, name)

    def make_pruner(self, args):
        return ResNetFilterPruner(self, args)

    def layer_out_channels(self, model: nn.Module, layer: str) -> int:
        return model.get_submodule(layer).out_channels

    def prune_filter(self, model: nn.Module, layer: str, filter_index: int, cuda_flag: bool = False):
        return prune_resnet_channel(model, layer, filter_index)

    def head_parameters(self):
        return self.model.fc.parameters()


class ResNetFilterPruner:
    """Accumulates per-filter LRP relevance for block-internal convs and builds a pruning plan.

    Same interface as prune_vgg.FilterPruner: reset(), forward_lrp(x), backward_lrp(R),
    normalize_ranks_per_layer(), get_pruning_plan(n).
    """

    def __init__(self, adapter: ResNetAdapter, args):
        self.adapter = adapter
        self.args = args
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self._ops = None

    def forward_lrp(self, x):
        logits, self._ops = resnet_lrp_forward(self.adapter.model, x)
        return logits

    def backward_lrp(self, R, relevance_method: str = "alpha", param=1):
        if (relevance_method, param) != ("alpha", 1):
            raise NotImplementedError(f"ResNet ranking supports ('alpha', 1) only, got ({relevance_method!r}, {param!r})")
        if self._ops is None:
            raise RuntimeError("call forward_lrp before backward_lrp")
        _, conv_relevance = resnet_lrp_backward(self._ops, R)
        for name in self.adapter.prunable_conv_names():
            values = conv_relevance[name].abs().sum(dim=(0, 2, 3))
            self.filter_ranks[name] = values if name not in self.filter_ranks else self.filter_ranks[name] + values
        self._ops = None

    def normalize_ranks_per_layer(self):
        for name in self.filter_ranks:
            if self.args.relevance:
                v = self.filter_ranks[name]
                self.filter_ranks[name] = (v / v.sum()).cpu()

    def lowest_ranking_filters(self, num: int) -> list:
        data = [(name, j, score)
                for name, ranks in self.filter_ranks.items()
                for j, score in enumerate(ranks.tolist())]
        return nsmallest(num, data, itemgetter(2))

    def get_pruning_plan(self, num_filters_to_prune: int) -> list:
        return shift_pruning_plan(self.lowest_ranking_filters(num_filters_to_prune))
