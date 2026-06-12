from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .camera import parse_float_list
from .downloads import UNIK3D_DIR
from .unik3d_patch import patch_unik3d_for_inference


@dataclass(frozen=True)
class CameraOptions:
    mode: str
    json_path: str = ""
    intrinsics: str = ""
    params: str = ""


@dataclass(frozen=True)
class InferenceConfig:
    image_path: Path
    out_root: Path
    checkpoint_path: Path
    camera: CameraOptions
    low_pass_filter_eps: float = 0.0
    splat_scale: float = 1.25
    return_gaussians: bool = False


@dataclass(frozen=True)
class InferenceResult:
    sample_dir: Path
    ply_path: Path
    metadata: dict[str, Any]
    camera_kind: str
    num_gaussians: int
    gaussians: Any | None = None


def _infer_module():
    prepare_unik3d_imports()
    from scripts import infer_unisharp as infer

    return infer


def prepare_unik3d_imports() -> None:
    patch_unik3d_for_inference(UNIK3D_DIR)


def configure_torchhub_cache() -> None:
    _infer_module().configure_torchhub_cache()


def load_model(checkpoint_path: Path, device):
    return _infer_module().load_model(Path(checkpoint_path), device=device)


def create_renderer(*, device, low_pass_filter_eps: float):
    infer = _infer_module()
    return infer.GSplatRenderer(
        color_space="sRGB",
        background_color="black",
        low_pass_filter_eps=float(low_pass_filter_eps),
    ).to(device)


def create_panorama_renderer(*, renderer):
    infer = _infer_module()
    return infer.CubemapPanoramaRenderer(renderer=renderer)


def count_ply_vertices(path: Path) -> int:
    try:
        from plyfile import PlyData

        ply = PlyData.read(path)
        vertices = next((elem for elem in ply.elements if elem.name == "vertex"), None)
        return int(vertices.count) if vertices is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def run_single_image(*, pipeline, config: InferenceConfig) -> InferenceResult:
    infer = _infer_module()
    camera_json = Path(config.camera.json_path) if str(config.camera.json_path).strip() else None
    args = argparse.Namespace(
        checkpoint=Path(config.checkpoint_path),
        image=Path(config.image_path),
        image_list=None,
        image_dir=None,
        out_dir=Path(config.out_root),
        device=str(getattr(pipeline, "device", "cuda:0")),
        max_images=1,
        save_ply=True,
        camera_json=camera_json,
        camera_intrinsics=parse_float_list(config.camera.intrinsics),
        camera_params=parse_float_list(config.camera.params),
        camera=str(config.camera.mode),
        low_pass_filter_eps=float(config.low_pass_filter_eps),
        splat_scale=float(config.splat_scale),
        return_gaussians=bool(config.return_gaussians),
    )
    args._camera_json_data = infer.load_camera_json(camera_json)

    image_path = Path(config.image_path)
    out_root = Path(config.out_root)
    process_result = infer.process_one(
        model=pipeline.model,
        renderer=pipeline.renderer,
        panorama_renderer=pipeline.panorama_renderer,
        image_path=image_path,
        out_root=out_root,
        step=int(pipeline.step),
        args=args,
    )
    gaussians = process_result.get("gaussians") if isinstance(process_result, dict) else None

    sample_dir = out_root / infer.slug_from_path(image_path)
    ply_path = sample_dir / "gaussians.ply"
    if not ply_path.is_file():
        raise RuntimeError(f"UniSHARP did not produce a PLY at {ply_path}")

    metadata_path = sample_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return InferenceResult(
        sample_dir=sample_dir,
        ply_path=ply_path,
        metadata=metadata,
        camera_kind=str(metadata.get("camera_kind", "")),
        num_gaussians=count_ply_vertices(ply_path),
        gaussians=gaussians,
    )
