"""
ncp: reference implementation of Concept-aware Network Pruning (CNP).

"ncp" is the package/CLI identifier for CNP (the method's earlier name).

Modules
-------
AugmentedVGG16       virtual concept layer at conv4_3 + subspace ablation
lrp                  LRP propagation rules used for filter ranking
prune_layer          structural (channel-surgery) filter pruning
prune_vgg            iterative prune + fine-tune loop (vanilla and augmented VGG16)
checkpoint           save/load pruned models as state_dict + channel counts
data                 basketball ImageNet dataset constructor
watermark_transform  synthetic hanzi watermark (from Whac-A-Mole)
paths                default data/results locations
"""

__version__ = "1.0.0"
