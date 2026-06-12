from __future__ import annotations

from typing import Any

import torch

from unisharp.utils.camera_projection import cubemap_face_cameras
from unisharp.utils.pano import Cube2Equirec, get_pinhole_intrinsics_4x4


class CubemapPanoramaRenderer:
    """Inference-only cubemap renderer used for panorama previews."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer

    def render_cubemap(
        self,
        gaussians: Any,
        extr_w2c: torch.Tensor,
        face_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = gaussians.mean_vectors.device
        intr = get_pinhole_intrinsics_4x4(int(face_w)).to(device=device)[None].expand(6, -1, -1)
        extr_faces = cubemap_face_cameras(extr_w2c, device=device)
        out = self.renderer(
            gaussians,
            extrinsics=extr_faces,
            intrinsics=intr,
            image_width=int(face_w),
            image_height=int(face_w),
        )
        return out.color.contiguous(), out.depth.contiguous(), out.alpha.contiguous()

    def cube_to_erp(self, cube: torch.Tensor, equ_h: int, equ_w: int, face_w: int) -> torch.Tensor:
        cube = cube.permute(1, 0, 2, 3).unsqueeze(0)
        c2e = Cube2Equirec(face_w=int(face_w), equ_h=int(equ_h), equ_w=int(equ_w)).to(device=cube.device)
        return c2e(cube)
