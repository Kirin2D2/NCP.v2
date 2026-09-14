import torch

from ncp.AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix

DIMS = [128, 128, 128, 128]


def _orthogonal(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q


def test_ablation_scales_only_selected_block():
    U = _orthogonal()
    U_before = U.clone()
    U_ab, U_ab_T = ablate_subspace_matrix(U, DIMS, [1])

    assert torch.equal(U, U_before), "input U must not be modified"
    assert torch.allclose(U_ab[:, 128:256], U[:, 128:256] * 1e-4)
    for lo, hi in [(0, 128), (256, 512)]:
        assert torch.equal(U_ab[:, lo:hi], U[:, lo:hi])
    assert torch.allclose(U_ab_T, U_ab.t())


def test_ablated_virtual_layer_projects_out_the_subspace():
    U = _orthogonal()
    U_ab, U_ab_T = ablate_subspace_matrix(U, DIMS, [2])
    model = AugmentedVGG16(U_ab, U_ab_T, weights=None)
    a = torch.relu(torch.randn(2, 512, 7, 7, generator=torch.Generator().manual_seed(1)))
    with torch.no_grad():
        out = model.decode(model.encode(a))

    keep = torch.cat([U[:, :256], U[:, 384:]], dim=1)
    expected = torch.einsum("dk,ek,behw->bdhw", keep, keep, a)
    assert torch.allclose(out, expected, atol=1e-4)

    ablated = U[:, 256:384]
    before = torch.einsum("dk,bdhw->bkhw", ablated, a).abs().max()
    after = torch.einsum("dk,bdhw->bkhw", ablated, out).abs().max()
    assert before > 0.1 and after < 1e-4


def test_unablated_orthogonal_virtual_layer_is_identity():
    torch.manual_seed(0)
    U = _orthogonal()
    model = AugmentedVGG16(U, U.t(), weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        with_layer = model(x)
        model.augmented = False
        without_layer = model(x)
    assert torch.allclose(with_layer, without_layer, rtol=1e-3, atol=1e-4)
