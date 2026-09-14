import pytest
import torch
import torch.nn as nn
from torchvision.models import vgg16

from ncp.AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from ncp.checkpoint import (build_pruned_vgg16, extract_channels, load_pruned_checkpoint,
                            save_pruned_checkpoint, to_vgg16_state_dict,
                            vgg16_to_augmented_state_dict)
from ncp.prune_layer import get_conv_seq_and_name, prune_conv_layer_sequential
from ncp.prune_vgg import AugmentedVGGAdapter, VanillaVGGAdapter

FULL_CHANNELS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]


def _prunable_conv_indices(model, adapter):
    seq, _ = get_conv_seq_and_name(model)
    return [i for i, m in enumerate(seq) if isinstance(m, nn.Conv2d) and i not in adapter._disallowed]


def _prune_first_middle_last(model, adapter):
    """Remove one filter from the first, a middle, and the last prunable conv."""
    idx = _prunable_conv_indices(model, adapter)
    for layer in (idx[0], idx[len(idx) // 2], idx[-1]):
        model = prune_conv_layer_sequential(model, layer, 3)
        adapter.model = model
    return model


def _roundtrip_logits(model, tmp_path):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        expected = model(x)
    path = tmp_path / "pruned.pt"
    save_pruned_checkpoint(path, model, meta={"pruner": "test", "seed": 0})
    loaded, info = load_pruned_checkpoint(path)
    with torch.no_grad():
        actual = loaded(x)
    return expected, actual, info


def test_build_unpruned_matches_torchvision_keys():
    model = build_pruned_vgg16(FULL_CHANNELS, num_classes=1000)
    ref = vgg16(weights=None)
    assert model.state_dict().keys() == ref.state_dict().keys()
    for k, v in ref.state_dict().items():
        assert model.state_dict()[k].shape == v.shape, k


def test_augmented_pruned_roundtrip(tmp_path):
    U = torch.linalg.qr(torch.randn(512, 512, generator=torch.Generator().manual_seed(0)))[0]
    U_ab, U_ab_T = ablate_subspace_matrix(U, [128] * 4, [3])
    model = AugmentedVGG16(U_ab, U_ab_T, weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    adapter = AugmentedVGGAdapter(model)

    model = _prune_first_middle_last(model, adapter)
    with pytest.raises(ValueError):
        to_vgg16_state_dict(model)  # virtual layer still present
    adapter.on_before_final_finetune()

    channels = extract_channels(model)
    assert len(channels) == 13
    assert channels[9] == 512, "conv4_3 must never be pruned"
    assert channels[0] == 63 and channels[-1] == 511 and sum(channels) == sum(FULL_CHANNELS) - 3

    expected, actual, info = _roundtrip_logits(model, tmp_path)
    assert info["channels"] == channels and info["meta"]["pruner"] == "test"
    assert torch.allclose(expected, actual, atol=1e-5)


def test_vanilla_pruned_roundtrip(tmp_path):
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    adapter = VanillaVGGAdapter(model)
    model = _prune_first_middle_last(model, adapter)

    expected, actual, info = _roundtrip_logits(model, tmp_path)
    assert info["num_classes"] == 2
    assert torch.allclose(expected, actual, atol=1e-5)


def test_vgg16_to_augmented_state_dict_inverts_canonical_keys():
    ref = vgg16(weights=None).state_dict()
    remapped = vgg16_to_augmented_state_dict(ref)
    assert "before.21.weight" in remapped and "after.1.weight" in remapped
    assert not any(k.startswith("features.") for k in remapped)
