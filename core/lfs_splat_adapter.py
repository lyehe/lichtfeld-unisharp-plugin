"""Convert UniSHARP Gaussians into LFS raw splat tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from unisharp.utils import color_space as cs_utils
from unisharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics

ColorSpace = Literal["linearRGB", "sRGB"]


@dataclass(frozen=True)
class LFSSplatTensors:
    """Raw tensor contract expected by lichtfeld.scene.SplatData."""

    means: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    sh_degree: int = 0
    scene_scale: float = 1.0


def _flatten_last_dim(tensor: torch.Tensor, *, width: int, name: str) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[-1] != width:
        raise ValueError(f"{name} must have shape [..., {width}], got {tuple(tensor.shape)}")
    return tensor.detach().to(dtype=torch.float32).reshape(-1, width).contiguous()


def _flatten_opacity(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim < 1:
        raise ValueError(f"opacities must have shape [...] or [..., 1], got {tuple(tensor.shape)}")
    return tensor.detach().to(dtype=torch.float32).reshape(-1, 1).contiguous()


def _normalize_quaternions_wxyz(quaternions: torch.Tensor, *, eps: float) -> torch.Tensor:
    quaternions = torch.nan_to_num(quaternions, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    normalized = quaternions / norm.clamp_min(eps)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    return torch.where(norm > eps, normalized, identity).contiguous()


def activated_gaussians_to_lfs_raw(
    gaussians: Gaussians3D,
    *,
    color_space: ColorSpace = "linearRGB",
    scene_scale: float = 1.0,
    eps: float = 1e-6,
) -> LFSSplatTensors:
    """Encode activated UniSHARP attributes as raw 3DGS tensors for LFS.

    UniSHARP stores activated scales and alpha values. LFS scene/render APIs
    expect the raw 3DGS convention: log-scales, opacity logits, normalized
    wxyz quaternions, and SH DC coefficients rather than RGB colors.
    """
    means = _flatten_last_dim(gaussians.mean_vectors, width=3, name="mean_vectors")
    scales = _flatten_last_dim(gaussians.singular_values, width=3, name="singular_values")
    rotations = _flatten_last_dim(gaussians.quaternions, width=4, name="quaternions")
    colors = _flatten_last_dim(gaussians.colors, width=3, name="colors")
    opacities = _flatten_opacity(gaussians.opacities)

    n = means.shape[0]
    for name, tensor in (
        ("singular_values", scales),
        ("quaternions", rotations),
        ("colors", colors),
        ("opacities", opacities),
    ):
        if tensor.shape[0] != n:
            raise ValueError(f"{name} has {tensor.shape[0]} entries, expected {n}")

    means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    scales = torch.nan_to_num(scales, nan=eps, posinf=1.0 / eps, neginf=eps).clamp_min(eps)
    scaling_raw = torch.log(scales).contiguous()

    opacities = torch.nan_to_num(opacities, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
    opacity_raw = torch.logit(opacities).contiguous()

    rotation_raw = _normalize_quaternions_wxyz(rotations, eps=eps)

    colors = torch.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if color_space == "linearRGB":
        colors = cs_utils.linearRGB2sRGB(colors).clamp(0.0, 1.0)
    elif color_space != "sRGB":
        raise ValueError(f"Unsupported color_space {color_space!r}; expected 'linearRGB' or 'sRGB'")
    sh0_raw = convert_rgb_to_spherical_harmonics(colors).reshape(n, 1, 3).contiguous()

    shN_raw = torch.empty((n, 0, 3), dtype=torch.float32, device=means.device)
    return LFSSplatTensors(
        means=means,
        sh0=sh0_raw,
        shN=shN_raw,
        scaling=scaling_raw,
        rotation=rotation_raw,
        opacity=opacity_raw,
        sh_degree=0,
        scene_scale=float(scene_scale),
    )


def _import_lfs():
    import lichtfeld as lf

    return lf


def torch_to_lfs_tensor(tensor: torch.Tensor, lf: Any | None = None):
    """Create a lichtfeld.Tensor from a torch tensor, preferring DLPack sharing."""
    lf = lf or _import_lfs()
    tensor = tensor.detach().to(dtype=torch.float32).contiguous()
    try:
        return lf.Tensor.from_dlpack(tensor)
    except Exception:
        return lf.Tensor.from_numpy(tensor.cpu().numpy(), copy=True)


def raw_to_lfs_splat_data(raw: LFSSplatTensors, lf: Any | None = None):
    """Build lichtfeld.scene.SplatData from already encoded raw tensors."""
    lf = lf or _import_lfs()
    return lf.scene.SplatData(
        torch_to_lfs_tensor(raw.means, lf),
        torch_to_lfs_tensor(raw.sh0, lf),
        torch_to_lfs_tensor(raw.shN, lf),
        torch_to_lfs_tensor(raw.scaling, lf),
        torch_to_lfs_tensor(raw.rotation, lf),
        torch_to_lfs_tensor(raw.opacity, lf),
        raw.sh_degree,
        raw.scene_scale,
    )


def gaussians_to_lfs_splat_data(
    gaussians: Gaussians3D,
    *,
    lf: Any | None = None,
    color_space: ColorSpace = "linearRGB",
    scene_scale: float = 1.0,
    eps: float = 1e-6,
):
    """Convert UniSHARP Gaussians directly into lichtfeld.scene.SplatData."""
    raw = activated_gaussians_to_lfs_raw(
        gaussians,
        color_space=color_space,
        scene_scale=scene_scale,
        eps=eps,
    )
    return raw_to_lfs_splat_data(raw, lf)


def render_gaussians(
    gaussians: Gaussians3D,
    rotation,
    translation,
    width: int,
    height: int,
    *,
    lf: Any | None = None,
    color_space: ColorSpace = "linearRGB",
    scene_scale: float = 1.0,
    fov_degrees: float = 60.0,
    intrinsics: tuple[float, float, float, float] | None = None,
    rgba: bool = False,
    bg_color=None,
):
    """Render activated UniSHARP Gaussians through LFS with correct raw encoding."""
    lf = lf or _import_lfs()
    splat = gaussians_to_lfs_splat_data(
        gaussians,
        lf=lf,
        color_space=color_space,
        scene_scale=scene_scale,
    )
    return lf.render_splat_data(
        splat,
        rotation,
        translation,
        int(width),
        int(height),
        float(fov_degrees),
        intrinsics,
        bool(rgba),
        bg_color,
    )

