from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from core import lfs_splat_adapter
from unisharp.utils import color_space as cs_utils
from unisharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics


def _sample_gaussians() -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        singular_values=torch.tensor([[[0.25, 1.0, 4.0], [0.5, 2.0, 8.0]]]),
        quaternions=torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]]]),
        colors=torch.tensor([[[0.0, 0.25, 1.0], [1.0, 0.5, 0.0]]]),
        opacities=torch.tensor([[0.25, 0.75]]),
    )


def test_activated_gaussians_to_lfs_raw_matches_3dgs_contract():
    gaussians = _sample_gaussians()

    raw = lfs_splat_adapter.activated_gaussians_to_lfs_raw(gaussians)

    assert raw.means.shape == (2, 3)
    assert raw.sh0.shape == (2, 1, 3)
    assert raw.shN.shape == (2, 0, 3)
    assert raw.scaling.shape == (2, 3)
    assert raw.rotation.shape == (2, 4)
    assert raw.opacity.shape == (2, 1)
    assert raw.sh_degree == 0

    torch.testing.assert_close(raw.means, gaussians.mean_vectors.reshape(-1, 3))
    torch.testing.assert_close(raw.scaling, torch.log(gaussians.singular_values.reshape(-1, 3)))
    torch.testing.assert_close(raw.opacity, torch.logit(gaussians.opacities.reshape(-1, 1)))
    torch.testing.assert_close(torch.linalg.vector_norm(raw.rotation, dim=-1), torch.ones(2))

    expected_sh0 = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.reshape(-1, 3)).clamp(0.0, 1.0)
    ).reshape(2, 1, 3)
    torch.testing.assert_close(raw.sh0, expected_sh0)


def test_activated_gaussians_to_lfs_raw_clamps_degenerate_scale_and_opacity():
    gaussians = Gaussians3D(
        mean_vectors=torch.zeros((1, 2, 3)),
        singular_values=torch.tensor([[[0.0, 1.0, float("inf")], [float("nan"), -1.0, 2.0]]]),
        quaternions=torch.zeros((1, 2, 4)),
        colors=torch.tensor([[[float("nan"), -0.5, 1.5], [0.2, 0.3, 0.4]]]),
        opacities=torch.tensor([[0.0, 1.0]]),
    )

    raw = lfs_splat_adapter.activated_gaussians_to_lfs_raw(gaussians)

    assert torch.isfinite(raw.scaling).all()
    assert torch.isfinite(raw.opacity).all()
    torch.testing.assert_close(raw.rotation, torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))


def test_activated_gaussians_to_lfs_raw_accepts_srgb_colors():
    gaussians = _sample_gaussians()

    raw = lfs_splat_adapter.activated_gaussians_to_lfs_raw(gaussians, color_space="sRGB")

    expected_sh0 = convert_rgb_to_spherical_harmonics(gaussians.colors.reshape(-1, 3)).reshape(2, 1, 3)
    torch.testing.assert_close(raw.sh0, expected_sh0)


def test_activated_gaussians_to_lfs_raw_rejects_shape_mismatch():
    gaussians = _sample_gaussians()._replace(colors=torch.zeros((3, 3)))

    with pytest.raises(ValueError, match="colors has 3 entries"):
        lfs_splat_adapter.activated_gaussians_to_lfs_raw(gaussians)


def test_gaussians_to_lfs_add_splat_args_uses_lfs_tensor_bridge():
    class FakeTensor:
        @staticmethod
        def from_dlpack(obj):
            return obj.detach().clone()

        @staticmethod
        def from_numpy(arr, copy=True):
            return torch.from_numpy(arr.copy() if copy else arr)

    fake_lf = SimpleNamespace(Tensor=FakeTensor)

    args = lfs_splat_adapter.gaussians_to_lfs_add_splat_args(_sample_gaussians(), lf=fake_lf, scene_scale=2.5)

    assert args.means.shape == (2, 3)
    assert args.shN.shape == (2, 0, 3)
    assert args.sh_degree == 0
    assert args.scene_scale == 2.5
