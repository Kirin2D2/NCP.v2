"""Tests for experiments/compute_drsa_resnet.py (activation / LRP-context extraction for DRSA)."""

import importlib.util

import pytest
import torch

from conftest import REPO_ROOT
from ncp.resnet import ARCHS

pytest.importorskip("zennit")


@pytest.fixture(scope="module")
def drsa_script():
    path = REPO_ROOT / "experiments" / "compute_drsa_resnet.py"
    spec = importlib.util.spec_from_file_location("compute_drsa_resnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(arch="resnet18"):
    torch.manual_seed(0)
    model = ARCHS[arch](weights=None).eval()
    # Non-trivial BatchNorm statistics, so folding BN into the convs is actually exercised.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


X = torch.randn(3, 3, 96, 96, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("layer", ["layer2", "layer3", "layer4"])
def test_gamma0_context_equals_gradient(drsa_script, layer):
    """LRP-0 (gamma = 0, z-rule at sums and pooling) above the layer is the gradient there."""
    model = _model()
    a0, grad = drsa_script.activation_and_context(model, layer, X, 7, None)
    a1, ctx = drsa_script.activation_and_context(model, layer, X, 7,
                                                 drsa_script.make_composite(layer, 0.0, 1e-9))
    torch.testing.assert_close(a1, a0, rtol=1e-3, atol=1e-4)
    assert (ctx - grad).norm() / grad.norm() < 1e-3
    # the context is nonzero where the activation is zero, as with cxai's hooks
    assert ctx[a0 == 0].abs().sum() > 0


def test_context_times_activation_is_zennit_relevance(drsa_script):
    """With gamma > 0, a * c equals the relevance Zennit's unmodified rules assign to the layer."""
    from zennit.composites import LayerMapComposite, layer_map_base
    from zennit.rules import Epsilon, Gamma
    from zennit.torchvision import ResNetCanonizer
    from zennit.types import Convolution

    model, layer, target, gamma = _model(), "layer3", 7, 0.25
    a, ctx = drsa_script.activation_and_context(model, layer, X, target,
                                                drsa_script.make_composite(layer, gamma, 1e-6))
    plain = LayerMapComposite(layer_map=layer_map_base() + [(Convolution, Gamma(gamma=gamma)),
                                                            (torch.nn.Linear, Epsilon(epsilon=1e-6))],
                              canonizers=[ResNetCanonizer()])
    tap = drsa_script.StageTap(getattr(model, layer))
    try:
        with plain.context(model) as modified:
            logits = modified(X)
            seed = torch.zeros_like(logits)
            seed[:, target] = logits[:, target].detach()
            (R,) = torch.autograd.grad(logits, tap.out, grad_outputs=seed)
    finally:
        tap.remove()
    torch.testing.assert_close(a * ctx, R, rtol=1e-4, atol=1e-6)
    _, grad = drsa_script.activation_and_context(model, layer, X, target, None)
    assert (ctx - grad).norm() / grad.norm() > 1e-2   # gamma > 0 really changes the context


def test_consumers_of(drsa_script):
    assert drsa_script.consumers_of("layer3") == {"layer4.0.conv1", "layer4.0.downsample.0"}
    assert drsa_script.consumers_of("layer4") == {"fc"}
    model = _model("resnet50")
    names = dict(model.named_modules())
    for layer in ("layer1", "layer2", "layer3"):
        assert drsa_script.consumers_of(layer) <= set(names)


def test_normalize_matches_eq_e1(drsa_script):
    x = torch.randn(200, 64) * 3.0 + 1.0
    n = drsa_script.normalize(x)
    # E[n^2] = 1 / sqrt(D), and the normalization is invariant to a global scale
    torch.testing.assert_close(n.pow(2).mean(), torch.tensor(64 ** -0.5))
    torch.testing.assert_close(drsa_script.normalize(5 * x), n)
