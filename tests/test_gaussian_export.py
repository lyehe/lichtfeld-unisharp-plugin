from __future__ import annotations

import pytest
import torch

from unisharp.utils.gaussians import Gaussians3D, gaussian_attribute_stats, sanitize_gaussians_for_export


def test_sanitize_gaussians_for_export_normalizes_and_scales_attributes():
    gaussians = Gaussians3D(
        mean_vectors=torch.tensor([[[float("nan"), 2.0, float("inf")], [4.0, 5.0, 6.0]]]),
        singular_values=torch.tensor([[[0.0, 1.0, float("nan")], [2.0, -3.0, float("inf")]]]),
        quaternions=torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]]]),
        colors=torch.tensor([[[float("nan"), -0.5, 1.5], [0.2, 0.3, 0.4]]]),
        opacities=torch.tensor([[0.0, 1.0]]),
    )

    out = sanitize_gaussians_for_export(gaussians, scale_multiplier=2.0, min_scale=1e-4, opacity_eps=1e-4)

    assert torch.isfinite(out.mean_vectors).all()
    assert torch.isfinite(out.singular_values).all()
    assert torch.isfinite(out.colors).all()
    assert torch.isfinite(out.opacities).all()
    assert float(out.singular_values.min().item()) == pytest.approx(2e-4)
    assert float(out.opacities.min().item()) == pytest.approx(1e-4)
    assert float(out.opacities.max().item()) == pytest.approx(1.0 - 1e-4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(out.quaternions.reshape(-1, 4), dim=-1),
        torch.ones(2),
    )
    torch.testing.assert_close(out.quaternions[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_gaussian_attribute_stats_are_json_ready():
    gaussians = sanitize_gaussians_for_export(
        Gaussians3D(
            mean_vectors=torch.ones((1, 2, 3)),
            singular_values=torch.ones((1, 2, 3)) * 0.5,
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]),
            colors=torch.ones((1, 2, 3)) * 0.25,
            opacities=torch.tensor([[0.25, 0.75]]),
        )
    )

    stats = gaussian_attribute_stats(gaussians)

    assert stats["count"] == 2
    assert stats["scale"]["median"] == pytest.approx(0.5)
    assert stats["opacity"]["p10"] is not None
    assert stats["quaternion_norm"]["median"] == pytest.approx(1.0)
